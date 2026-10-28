# 基于 rax 的 KV 前缀缓存（slot pool / SGLang 式）设计

日期：2026-10-28
状态：Draft，待确认

## 1. 背景与目标

server 模式在每个请求开始时会 `reset` 会话，然后重新 prefill 整段 prompt
（`src/server.c`）。如果多个请求共享同一段前缀（相同的 system prompt、相同的
多轮历史、或完全相同的 prompt），这段前缀的 KV 会被重复计算。

本设计用 rax 放射树记录 **token 前缀 → 物理 KV slot**，并把 KV 存储从
“逻辑位置就是槽位”改成“全局 slot pool + 每个请求的 slot 索引表”。这样：

- 命中前缀时直接复用 KV，只对后缀执行 forward；
- 共享前缀的多个分支可以同时保留；
- 不同分支分配不同 slot，互不覆盖；
- 用 LRU + 引用计数回收不再使用的 slot。

目标：

- 相同/部分相同 prompt 的 prefill 跳过重复计算。
- 支持多分支共享前缀，行为接近 SGLang 的 RadixAttention。
- rax 只保存元数据；KV 数据仍然只有一份。
- 与现有 graph executor（`src/graph.c`）、`GraphPlan.cache`、CPU/CUDA/Metal
  后端兼容。

非目标（第一版不做）：

- page_size > 1 / paged block table。
- device / host 分层 KV。
- 混合架构（Gated DeltaNet / Mamba2）的 recurrent state 快照。
- MLA 前缀缓存。
- 多 session / 并发共享同一个 pool。
- “全命中零 forward”：KV 里没有最后一个 token 的 hidden state / logits。
- CUDA/Metal 的 indexed attention kernel（第一版先做 CPU gather；device 层在
  kernel 落地前不启用前缀缓存）。

## 2. 现状与约束

以下事实决定了本方案的形状：

1. **KV 缓存在 `GraphPlan` 里，不在 `Session` 里**
   - `GraphPlan.cache` 是 `KvCache *`（`src/def.h`）。
   - 标准注意力由 `cache_build_attn_kvcache()` 分配（`src/graph.c`），每层一份
     `AttnKvCache.k/v`，容量为 `ctx_size`，布局为
     `[kv_head][position][kv_head_dim]`。
   - `cache_build()` 在 `graph_plan()` 内被调用，plan 是懒加载的。

2. **`op_attn` 目前把逻辑位置直接当 KV 槽位**
   - `op_attn` 使用 `c->pos`、`c->n` 和 `akc->cap`，读写
     `akc->k + head * cap * kv_head_dim + pos * kv_head_dim`。
   - CPU/CUDA/Metal 三份实现使用同样的布局。
   - 本方案要打破这个假设：逻辑位置通过请求表映射到物理 slot。

3. **`graph_execute` 以 `s->n_tokens` 作为起始位置**
   - 内部按 `n_ubatch` 分块，每块调用 `graph_compute()`。
   - 结束时 `s->n_tokens = pos + n_tokens`。
   - 前缀匹配和请求表准备必须发生在进入分块循环之前。

4. **`reset` 当前只清逻辑长度**
   - Llama/Qwen2/Falcon/Gemma3 的 reset 只把 `s->n_tokens = 0`。
   - Qwen3.5 / Falcon-H1 的 reset 还会清零 SSM/conv recurrent state。
   - server 每个请求前都会 reset，这是前缀缓存生效的入口。

5. **rax 只有精确查找**
   - `raxFind()` 只能精确匹配，不支持“最长前缀”。
   - 内部 `raxLowWalk()` 是 static 的。
   - 需要新增一个公开的 `raxFindPrefix()`。

## 3. 默认决策

本 spec 按以下默认值编写；如需调整，先改本节再改后续设计。

| 决策 | 默认值 | 理由 |
|---|---|---|
| 支持范围 | 纯 attention 架构 | Llama / Qwen2 / Falcon / Gemma3 |
| 混合架构 | 关闭 | Qwen3.5 / Falcon-H1 的 recurrent state 无法只靠 KV 复用 |
| MLA | 关闭 | 压缩 KV 不是逐 token 的干净切片 |
| 分支能力 | 多分支共存 | slot pool 允许同前缀的不同后缀同时保留 |
| KV 池容量 | `n_slots = max(ctx_size, kv_pool_tokens)`，默认 `kv_pool_tokens = ctx_size` | 至少容纳一个满上下文请求；调大可保留更多分支 |
| 位置与槽位 | 解耦 | 逻辑位置通过请求表映射到物理 slot |
| 全命中 | 最后一个 token 私有重算 | KV 里没有 logits；私有 slot 也避免写共享 KV |
| 缓存内容 | prompt + 生成结果，但树不含当前最后 token | 保证最后一个 token 总能重算 |
| 驱逐 | 叶子 LRU + `lock_ref` | 多分支必须避免覆盖旧分支 |
| 并发 | 单 session / 单请求 | server 当前是单线程单 session |
| 后端 | 第一版仅全 CPU attention 启用 | device indexed kernel 落地前不放开 |
| 满上下文 | 由 `graph_execute` 拒绝 | 不做“静默 flush 后继续生成” |

## 4. 总体方案

```
rax 树
  "A"       -> slot 7
  "AB"      -> slot 3
  "ABC"     -> slot 9
  "ABD"     -> slot 12     <- 与 ABC 共享 A、B，后缀各用自己的 slot
  ...

请求表（每个请求一份）
  pos:   0    1    2    3
  slot:  7    3    9   12

KV pool（每层一份连续缓冲，按 slot 索引）
  k/v[head][slot][kv_head_dim]
  slot 0,1,2,... 都是可分配的物理位置
```

请求流程：

```
begin
  |
  +--> 最长前缀匹配
  +--> 沿 SlotMeta.parent 收集前缀 slot
  +--> lock 共享前缀
  +--> 为后缀分配私有 slot
  |
forward
  |
  +--> op_attn 用请求表读写 KV
  |
finish
  |
  +--> 把非最后 token 挂树
  +--> 解锁共享前缀
  +--> 释放最后一个私有 slot
```

### 4.1 与旧版“单缓冲覆盖”的区别

旧版：

```
逻辑位置 i == KV 槽位 i
分叉时覆盖 [match_end, ...)，旧分支被驱逐
tree 实际只剩一条活动路径
```

本版：

```
逻辑位置 i --请求表--> 任意物理 slot
分叉时新分支分配新 slot，旧分支不动
tree 可以同时保留多条分支
驱逐由叶子 LRU 决定，而不是写入覆盖
```

rax 树本身不需要换；变化的是节点值语义、slot 生命周期、请求表和
`op_attn` 的寻址方式。

## 5. 数据结构与接口

### 5.1 slot pool 与元数据

```c
/* src/kvcache.h */
typedef struct {
    u32 token;        /* 该 slot 对应的 token id */
    u32 parent;       /* 前一个 token 的 slot；根为 U32_MAX */
    u32 child_count;  /* 直接子节点数，用于识别叶子 */
    u32 lock_ref;     /* 正在被多少请求使用 */
    u64 last_used;    /* LRU 时间戳 */
    bool in_tree;     /* 是否已经挂到 rax */
} KvSlotMeta;

typedef struct {
    u32  n_slots;
    u32  n_free;
    u32 *free_stack;
} KvSlotPool;

typedef struct {
    u32 *slots;        /* [ctx_size]，逻辑位置 -> 物理 slot */
    u32  len;          /* 当前序列长度 */
    u32  shared_len;   /* 匹配并 lock 的共享前缀长度 */
    bool active;
} KvRequestState;

typedef struct KvRadixCache {
    rax          *tree;        /* token 前缀 -> slot + 1 */
    KvSlotPool    pool;
    KvSlotMeta   *meta;        /* [n_slots] */
    KvRequestState req;
    KvCache      *kv;          /* 指向 GraphPlan.cache */
    u32           n_layer;
    u32           ctx_size;
    bool          enabled;
    unsigned char *keybuf;     /* key 编码 scratch */
    size_t         keybuf_cap;
    u64            clock;      /* 单调递增的 LRU 计数器 */
    /* stats */
    u64 hit_tokens, computed_tokens, evictions, allocations;
} KvRadixCache;
```

- slot id 在所有层之间通用：`slot 5` 表示每层缓冲里的第 5 个位置。
- 每层 `AttnKvCache.cap = n_slots`，布局仍是
  `[kv_head][slot][kv_head_dim]`。
- `n_slots >= ctx_size`，保证一个满上下文请求至少能放下。

### 5.2 接口

```c
bool kvcache_init(KvRadixCache *c, KvCache *kv, u32 n_layer,
                  u32 ctx_size, u32 n_slots, bool enabled);
void kvcache_free(KvRadixCache *c);
void kvcache_flush(KvRadixCache *c);

/* 请求生命周期 */
bool kvcache_begin(KvRadixCache *c, const u32 *tokens, u32 n, u32 *skip);
bool kvcache_finish(KvRadixCache *c, const u32 *tokens);
void kvcache_abort(KvRadixCache *c);
void kvcache_reset(KvRadixCache *c, const u32 *tokens);

/* decode：为下一个 token 分配一个私有 slot，并追加到请求表 */
u32  kvcache_decode_alloc(KvRadixCache *c);

/* token 序列之后可能用于诊断 */
void kvcache_stats(const KvRadixCache *c);
```

`kvcache_reset()` 的行为：

1. 如果有 active request，先 `kvcache_finish()`；
2. 清空 `req.len / req.shared_len / req.active`；
3. **不清树，不清 pool**。

### 5.3 key 编码与 value 编码

每个 token 编码成 4 字节大端：

```
t0          -> [00 00 00 01]
t0 t1       -> [00 00 00 01][00 00 00 02]
t0 t1 t2    -> [00 00 00 01][00 00 00 02][00 00 00 03]
```

每个前缀都插入一个 key 节点：

```
key   = tokens[0..i]
value = (void *)(uintptr_t)(slot_i + 1)
```

存 `slot + 1` 是为了避免 `slot == 0` 与 `NULL` 混淆；`raxNotFound` 仍然是
唯一表示“没有命中”的返回值。

### 5.4 所有权

`KvRadixCache` 的生命周期绑定到 `GraphPlan`，因为它的 KV 指针指向
`GraphPlan.cache`：

```c
/* def.h */
typedef struct KvRadixCache KvRadixCache;

typedef struct GraphPlan {
    ...
    KvRadixCache *radix;   /* 与 cache 同生命周期 */
} GraphPlan;
```

- 在 `graph_plan()` 完成 `cache_build()` 和 `graph_warmup()` 之后创建。
- 在 `cache_destroy()` 中、释放 KV arena 之前销毁。
- 不放进 `Session`，避免 graph 重建/释放后出现悬空指针。

对外访问辅助：

```c
KvRadixCache *graph_radix_cache(Session *s);
```

返回 `s->graph && s->graph->plan ? s->graph->plan->radix : NULL`。

## 6. rax 扩展与路径收集

### 6.1 `raxFindPrefix`

在 `src/rax.h` 增加：

```c
void *raxFindPrefix(rax *rt, unsigned char *s, size_t len, size_t *matchlen);
```

语义：

- 返回查询路径上**最深的 key 节点**的数据；
- `*matchlen` 为该 key 的字节长度；
- 没有命中返回 `raxNotFound`，`*matchlen = 0`。

实现要点：

- 复用 `raxLowWalk` 的遍历思路，但需要记录“路径上最后一个 iskey 节点”。
- **每到达一个节点先检查 `n->iskey`，再消费字符/下跳。**
- 非压缩节点：节点自身的字符是子边，消费后下跳。
- 压缩节点：节点存储的字符串属于其下的虚拟节点，节点自己的 key 是到达其
  起点的路径；所以必须先检查 iskey，再比较/消费字符串。
- 部分匹配压缩节点时，只能匹配到该压缩节点起点之前的 key，不能继续深入。

返回值用 `slot + 1` 编码，不会是数据值 `NULL`。

### 6.2 用 parent 链收集整条前缀

`raxFindPrefix` 只返回最深节点的 slot。为了得到整条前缀的 slot 列表，沿
`KvSlotMeta.parent` 往回走：

```c
u32 s = deepest_slot;
for (u32 i = matchlen; i-- > 0; ) {
    c->req.slots[i] = s;
    s = c->meta[s].parent;
}
```

前提是每个 token 都是一个 key 节点，所以 parent 链长度恰好等于匹配的 token
数。这样不需要新增“收集路径”的 rax API。

## 7. 生命周期

### 7.1 init

由 `graph_plan()` 在 plan 成功后调用：

```c
bool enabled = kvcache_arch_supported(s->en->model->arch) &&
               all_attn_layers_on_cpu(s);   /* 第一阶段限制 */
g->plan->radix = smalloc(sizeof(*g->plan->radix));
if (!g->plan->radix ||
    !kvcache_init(g->plan->radix, g->plan->cache, s->cfg.n_layer,
                  s->ctx_size, s->kv_pool_tokens, enabled)) {
    /* 失败：释放 radix，plan_reset，返回 false */
}
```

`kvcache_arch_supported()` 默认返回：

```
Llama / Qwen2 / Falcon / Gemma3 -> true
Qwen3.5 / Falcon-H1 / DeepSeek -> false
```

`all_attn_layers_on_cpu()` 是第一阶段限制：在 device indexed kernel 落地前，
只要有任何 attention 层在 CUDA/Metal，就关闭前缀缓存。关闭后 slot 表仍然可以
退化使用（逻辑位置 == slot），但不会做前缀复用。

`kvcache_init()` 初始化时：

- `free_stack` 填入 `0 .. n_slots-1`，`n_free = n_slots`；
- `meta[]` 清零，所有 slot `in_tree = false`；
- 请求表 `slots` 分配为 `ctx_size` 项，`req.active = false`；
- 校验 `kv->std[i].cap == n_slots`，不一致直接失败。

### 7.2 free

`cache_destroy()` 中：

```c
if (p->radix) {
    kvcache_free(p->radix);
    sfree(p->radix);
    p->radix = NULL;
}
/* 然后才释放 p->cache 和 cache_arena */
```

`kvcache_free()` 释放 rax 树、`meta`、`free_stack`、请求表、`keybuf`，不释放
KV 数据。

### 7.3 reset

各架构 reset：

- Llama / Qwen2 / Falcon / Gemma3：
  `kvcache_reset(graph_radix_cache(s), s->tokens); s->n_tokens = 0;`
- Qwen3.5 / Falcon-H1 / DeepSeek：
  `kvcache_reset(...)`（disabled 时是空操作），保留原有 recurrent state
  清零逻辑，`s->n_tokens = 0`。

`kvcache_reset()` 会先 finish 当前请求，把已经算好的非最后 token 挂树，然后
清空请求状态。树和 pool 保留。

### 7.4 flush

`kvcache_flush()` 是异常路径：先 `kvcache_abort()`（如果有 active request），
然后释放整棵树（`raxFree`），把所有 slot 归还 `free_stack`，清空 `meta` 和
请求状态，但保留 KV 缓冲本身。用于：

- 元数据损坏（例如检测到 slot/parent 不一致）；
- 测试或显式重置；
- OOM 后无法保证树一致性。

正常运行不调用 flush；上下文满由 `graph_execute` 的 `ctx_size` 检查拒绝。

## 8. 请求流程

### 8.1 begin：匹配 + 锁定 + 分配

```c
bool kvcache_begin(KvRadixCache *c, const u32 *tokens, u32 n, u32 *skip) {
    *skip = 0;
    if (!c || !c->enabled) return true;
    if (c->req.active || n == 0) return false;

    /* 1. 编码完整 token 序列 */
    if (!build_key(c, tokens, n)) return false;

    /* 2. 最长前缀匹配 */
    size_t klen = 0;
    void *data = raxFindPrefix(c->tree, c->keybuf, (size_t)n * 4, &klen);
    u32 matchlen = (data == raxNotFound) ? 0 : (u32)(klen / 4);

    /* 3. 最后一个 token 始终重算 / 私有 */
    u32 use = (matchlen >= n) ? n - 1 : matchlen;

    /* 4. 沿 parent 链收集前缀 slot */
    if (matchlen > 0) {
        u32 s = (u32)(uintptr_t)data - 1;
        for (u32 i = matchlen; i-- > 0; ) {
            c->req.slots[i] = s;
            s = c->meta[s].parent;
        }
    }

    /* 5. 锁共享前缀，并刷新 LRU */
    u64 now = ++c->clock;
    for (u32 i = 0; i < use; i++) {
        u32 s = c->req.slots[i];
        c->meta[s].lock_ref++;
        c->meta[s].last_used = now;
    }

    /* 6. 为 [use, n) 分配私有 slot；必要时驱逐叶子 */
    if (!kvcache_reserve(c, n - use)) {
        for (u32 i = 0; i < use; i++)
            c->meta[c->req.slots[i]].lock_ref--;
        return false;
    }
    for (u32 i = use; i < n; i++)
        c->req.slots[i] = kvcache_slot_alloc(c);

    c->req.len = n;
    c->req.shared_len = use;
    c->req.active = true;
    *skip = use;
    return true;
}
```

注意：

- 收集时会填出 `matchlen` 个 slot；如果 `matchlen > use`（全命中），
  `[use, matchlen)` 里的缓存 slot 只是临时值，随后会被私有 slot 覆盖。
- 只 lock `[0, use)`，也就是真正复用的部分。
- `kvcache_reserve()` 只驱逐 `in_tree && child_count == 0 &&
  lock_ref == 0` 的叶子。

### 8.2 forward

`graph_execute` 从 `use` 开始执行到序列末尾：

```
执行区间 = [use, total_len)
```

- 如果 `use < 旧历史长度`（chat 续接时尾部 token 重算），执行区间会包含少量旧
  token；它们的 slot 已分配，重新计算即可。
- 如果 `use` 落在新 prompt 中间，说明匹配到了之前缓存过的 prompt，执行剩余
  后缀。
- 最后一行一定在 batch 里，因此 logits 正常产出。

### 8.3 finish：挂树 + 解锁 + 释放尾部

```c
bool kvcache_finish(KvRadixCache *c, const u32 *tokens) {
    if (!c || !c->req.active) return true;

    u32 n = c->req.len;
    u32 from = c->req.shared_len;

    if (!build_key(c, tokens, n)) {
        kvcache_abort(c);
        return false;
    }

    /* 只插入 [from, n-1)；最后一个 token 保持私有 */
    for (u32 i = from; i + 1 < n; i++) {
        u32 slot   = c->req.slots[i];
        u32 parent = (i == 0) ? U32_MAX : c->req.slots[i - 1];

        if (raxInsert(c->tree, c->keybuf, (size_t)(i + 1) * 4,
                      (void *)(uintptr_t)(slot + 1), NULL) != 1) {
            /* OOM / 异常：回滚已插入节点，abort 请求 */
            kvcache_rollback_insert(c, from, i);
            kvcache_abort(c);
            return false;
        }

        c->meta[slot].token       = tokens[i];
        c->meta[slot].parent      = parent;
        c->meta[slot].child_count = 0;
        c->meta[slot].last_used   = ++c->clock;
        c->meta[slot].in_tree     = true;
        if (parent != U32_MAX)
            c->meta[parent].child_count++;
    }

    /* 解锁共享前缀 */
    for (u32 i = 0; i < from; i++)
        c->meta[c->req.slots[i]].lock_ref--;

    /* 释放最后一个私有 slot */
    kvcache_slot_free(c, c->req.slots[n - 1]);
    c->req.active = false;
    return true;
}
```

几个关键点：

- **树不含当前最后 token**。这样下一次全命中时，最后一个 token 总能分到私有
  slot 重算，不会往共享 KV 里写。
- 代价是每个序列少缓存 1 个 token；对长 prompt 可忽略。
- 插入 `[from, n-1)` 的 parent 可能是共享前缀节点，也可能是本请求刚插入的
  节点；`child_count` 正常累加，形成多分支。
- 插入的 key 应该是新的：如果已存在，说明匹配时就应该命中它，不会走到这里。
  实现时可以断言或做一致性检查。
- **插入失败必须回滚**：`kvcache_rollback_insert(c, from, i)` 逆序删除本次已
  插入的 key、`child_count--`、把对应 slot 标记为 `in_tree = false`，然后
  `kvcache_abort()` 统一释放所有私有 slot 并解锁。不能直接 abort，否则会把
  已插入、仍在树里的 slot 重复释放。

### 8.4 abort

```c
void kvcache_abort(KvRadixCache *c) {
    if (!c || !c->req.active) return;

    for (u32 i = c->req.shared_len; i < c->req.len; i++)
        kvcache_slot_free(c, c->req.slots[i]);
    for (u32 i = 0; i < c->req.shared_len; i++)
        c->meta[c->req.slots[i]].lock_ref--;

    c->req.active = false;
}
```

只在 forward 失败、begin 后的分配失败、finish 插入失败等路径使用。abort 不
修改树。

### 8.5 decode

每生成一个 token：

```c
u32 slot = kvcache_decode_alloc(c);
if (slot == U32_MAX) return false;   /* pool 无法腾出空间 */

s->tokens[s->n_tokens] = token;
/* raw forward 在当前 pos 写 KV 到 slot */
graph_execute_raw_one(s, token, slot);

c->req.len++;
```

`kvcache_decode_alloc()`：

```c
u32 kvcache_decode_alloc(KvRadixCache *c) {
    if (!c || !c->enabled || !c->req.active) return U32_MAX;
    if (c->req.len >= c->ctx_size) return U32_MAX;
    if (!kvcache_reserve(c, 1)) return U32_MAX;

    u32 slot = kvcache_slot_alloc(c);
    c->req.slots[c->req.len++] = slot;
    return slot;
}
```

decode 不立即挂树；finish 时统一插入 `[shared_len, len-1)`。这样：

- 最后一个生成 token 始终私有；
- 单 session 下不需要每生成一个 token 就做一次 rax 插入；
- 下一条请求/下一轮对话会在 finish 后命中除最后一个 token 外的全部历史。

### 8.6 chat 续接

chat 模式保留 `s->tokens` 和 `s->n_tokens`：

1. 上一轮生成结束后调用 `kvcache_finish()`，把上一轮除最后 token 外的内容挂树；
2. 下一轮把 continuation token 追加到 `s->tokens`；
3. `kvcache_begin()` 对整个历史 + 新 continuation 做匹配；
4. 由于树里没有上一轮最后一个 token，`use` 通常等于 `old_len - 1`；
5. 从 `use` 开始执行，重算上一轮最后 token + 新 continuation。

这样既复用了几乎全部历史，又保证最后 token 的 logits 可用。

## 9. `op_attn` 改造

### 9.1 写 K/V

当前：

```c
akc->k + kv_h * hs + (pos + i) * khd
```

改成：

```c
u32 slot = c->kv_slots[pos + i];
akc->k + kv_h * hs + slot * khd
```

`hs = akc->cap * khd` 不变，只是 `cap` 现在是 `n_slots`。

当 `c->kv_slots == NULL`（缓存 disabled，或第一阶段 device 层关闭）时，
`op_attn` 退化为原来的行为：

```c
u32 slot = (c->kv_slots != NULL) ? c->kv_slots[pos + i] : (pos + i);
```

这样 disabled 模式不需要额外的请求表，现有路径保持兼容。

### 9.2 读 K/V：CPU gather

当前 attention 假设某段 K 是连续的：

```c
mul_mat_vec(c->scr, kh_base + start * khd, qh, n_vis, khd);
```

slot 化之后，物理 slot 不连续。CPU 第一版采用“先 gather，再复用现有 kernel”：

```c
u32 base = start_for_qi0;          /* 所有 query row 可见区间的左端 */
u32 end  = c->pos + c->n - 1;      /* 本批最后一行能看到的最右 key */

for (u32 t = base; t <= end; t++) {
    u32 slot = c->kv_slots ? c->kv_slots[t] : t;
    memcpy(k_scr + (t - base) * khd,
           akc->k + kv_h * hs + slot * khd,
           khd * sizeof(float));
    memcpy(v_scr + (t - base) * khd,
           akc->v + kv_h * hs + slot * khd,
           khd * sizeof(float));
}

mul_mat_vec(c->scr,
            k_scr + (start - base) * khd,
            qh, n_vis, khd);
```

然后 V 的加权和也在 `v_scr` 上做。

要点：

- 每个 KV head 只 gather 一次；GQA 下同一个 KV head 被 `gqa` 个 Q head 共享，
  不要按 Q head 重复 gather。
- 滑动窗口：每个 query row 的 `start` 可能不同，但可见区间是连续逻辑范围，
  gather 并集 `[base, end]`，每行用偏移 `start - base`。
- scratch 放在 `GraphPlan` 或 executor 里复用：
  `k_scratch[ctx_size * khd]`、`v_scratch[ctx_size * khd]`。
- 如果 attention 以后并行化，每个线程需要独立 scratch，或者改成按 query row
  gather。

### 9.3 其他后端

- CUDA/Metal 最终应改成 indexed attention：kernel 里
  `slot = kv_slots[t]`，直接读 `K[slot]`，不经过 gather scratch。
- 请求表需要上传到设备：`u32 *kv_slots_dev`，每次 batch 前拷贝。
- 第一阶段 device kernel 未落地时，只要 attention 层在 device 上就关闭前缀
  缓存；关闭后逻辑位置 == slot，现有 kernel 仍可工作。

### 9.4 `OpCtx`

增加：

```c
const u32 *kv_slots;   /* 当前请求的逻辑位置 -> 物理 slot；NULL = 位置即 slot */
```

CPU 版是 host 指针；CUDA/Metal 版是设备指针。`graph_compute()`/`graph_op_compute()`
按节点 backend 选择对应副本。`kv_slots == NULL` 时 `op_attn` 退化为
`slot = pos`。

## 10. 驱逐与 LRU

### 10.1 叶子识别

一个 slot 可以驱逐，当且仅当：

```
meta[slot].in_tree == true
meta[slot].child_count == 0     // 没有下游依赖
meta[slot].lock_ref == 0        // 没有被活动请求使用
```

只有叶子能驱逐，因为父节点的 KV 仍被它的子分支当作前缀。

### 10.2 预留 slot

```c
static bool kvcache_reserve(KvRadixCache *c, u32 need) {
    while (c->pool.n_free < need) {
        if (!kvcache_evict_one(c))
            return false;
    }
    return true;
}
```

`kvcache_evict_one()`：

1. 找到 `last_used` 最小的可驱逐叶子；
2. 沿 `meta.token + meta.parent` 重建该节点的 key；
3. `raxRemove(tree, key, key_len, NULL)`；
4. `meta[parent].child_count--`；
5. `kvcache_slot_free(slot)`；
6. 如果 parent 变成叶子，它自然会在下一次扫描中被考虑。

第一版可以 `O(n_slots)` 扫描；后续可以维护叶子堆/链表，把驱逐降到
`O(log n)`。

key 重建：从被驱逐 slot 沿 `meta.token + meta.parent` 走到 root，把 token 收集
到临时数组后反转，再按 4 字节大端编码成 rax key。因为只有叶子会被驱逐，它的
所有祖先仍然有效。

### 10.3 lock_ref

- `begin` 对匹配到的共享前缀 `lock_ref++`；
- `finish` / `abort` 对同一段 `lock_ref--`；
- decode 新分配的私有 slot 不在树里，不会被驱逐；
- finish 插入的节点 `lock_ref == 0`，立即可被 LRU 回收。

### 10.4 池容量

- `n_slots >= ctx_size` 是硬性要求；否则一个满上下文请求都无法完成。
- `n_slots == ctx_size` 时，多分支只能“在总量内共存”，超出部分靠驱逐。
- 想保留更多历史分支，把 `n_slots` 调大（KV 内存线性增长）。
- 建议增加 session 配置 `kv_pool_tokens`，默认等于 `ctx_size`。

### 10.5 失败语义

- `kvcache_reserve()` 找不到可驱逐叶子：`begin` 失败，请求返回错误。
- forward 失败：`kvcache_abort()` 释放私有 slot、解锁前缀，树不变。
- finish 插入 OOM：回滚本次已插入的节点，abort 请求。
- 不做“驱逐正在使用的前缀”来强行腾空间。

## 11. graph executor 集成

### 11.1 请求边界

当前 server/chat 的调用模式是：

```
prefill:  graph_execute(prompt, n>1)
decode:   graph_execute(&token, 1) 反复调用
```

第一版约定：

- 没有 active request，或本次调用 `n > 1`：视为新请求/新一轮；
- active request 且 `n == 1`：视为 decode 一步；
- 生成结束或下一次新请求前，调用 `kvcache_finish()`；
- `reset` 兜底：如果还有 active request，先 finish。

### 11.2 伪代码

```c
bool graph_execute(Session *s, const u32 *tokens, u32 n, float *logits) {
    if (!s || !tokens || n == 0) return false;

    u32 old = s->n_tokens;
    u32 total = old + n;
    if ((u64)total > s->ctx_size) return false;

    /* 完整历史必须进 s->tokens，供 key 构建 / 采样使用 */
    memcpy(s->tokens + old, tokens, (size_t)n * sizeof(u32));

    KvRadixCache *c = graph_radix_cache(s);
    bool decode = (c && c->enabled && c->req.active && n == 1);

    if (c && c->enabled && !decode) {
        /* 新一轮：先结束上一轮 */
        if (c->req.active && !kvcache_finish(c, s->tokens))
            return false;

        u32 skip = 0;
        if (!kvcache_begin(c, s->tokens, total, &skip))
            return false;

        s->n_tokens = skip;
        if (!graph_execute_raw(s, s->tokens + skip, total - skip, logits)) {
            kvcache_abort(c);
            return false;
        }
        s->n_tokens = total;
        return true;
    }

    if (decode) {
        if (kvcache_decode_alloc(c) == U32_MAX) return false;
    }

    s->n_tokens = total - n;   /* raw 的起始位置 */
    if (!graph_execute_raw(s, tokens, n, logits)) {
        if (decode) kvcache_abort(c);
        return false;
    }
    s->n_tokens = total;
    return true;
}
```

说明：

- `graph_execute_raw()` 是现有 `graph_execute()` 的函数体，只负责分块 forward
  和 `s->n_tokens` 推进。
- `OpCtx.kv_slots` 在 raw 执行时指向 `c->req.slots`；cache disabled 时为 `NULL`。
- 新请求执行区间是 `[skip, total)`；decode 执行区间是 `[total-1, total)`。
- `s->tokens` 在 raw 之前已经是完整历史，raw 里的 memcpy 是幂等的。
- 如果 cache disabled，行为与当前完全一致。

### 11.3 finish 的调用点

`graph_finish(Session *s)` 是对外的收尾入口，内部调用
`kvcache_finish(graph_radix_cache(s), s->tokens)`：

- server：生成循环结束后调用 `graph_finish(s)`；如果忘了，下一次 `reset` 会
  finish。
- chat：每轮生成结束后调用 `graph_finish(s)`；下一轮 continuation 会基于完整
  历史重新 begin。
- bench：每次迭代结束调用 `graph_finish(s)` 或 reset。

### 11.4 `AttnKvCache` 与 plan

- `cache_build_attn_kvcache()` 的 `cap` 从 `ctx_size` 改为 `n_slots`；
  每层 bytes 相应变大。
- `AttnKvCache.n` 不再表示逻辑长度，只作为统计字段（可置为 `pool used`）。
- `GraphPlan.kvcache_size`、`session_auto_gpu_layers()` 的估算都要用
  `n_slots`，否则显存会超。
- `n_slots` 来源：`Session.kv_pool_tokens`（新增配置，默认 `ctx_size`）。

## 12. 后端策略

### 12.1 CPU

完整实现：

- 写 K/V 用 slot；
- 读 K/V 先 gather 到连续 scratch；
- 复用 `mul_mat_vec` 和现有 V 累加逻辑。

### 12.2 CUDA / Metal

分两步：

1. **第一阶段**：device indexed kernel 未实现前，只要 attention 层在 device
   就 `enabled = false`。关闭后仍然使用 `n_slots` 缓冲，但逻辑位置 == slot。
2. **第二阶段**：实现 paged/indexed attention：
   - 请求表上传到设备；
   - kernel 用 `slot = kv_slots[t]` 直接读 K/V；
   - 不需要 CPU gather scratch；
   - 和 SGLang 的 paged attention 形态一致。

### 12.3 split plan

- slot id 全局通用，各层缓冲可以在不同 backend。
- 请求表需要 host 一份 + 每个 device backend 一份。
- device 层未实现 indexed kernel 时，split plan 整体关闭前缀缓存，避免同一
  slot 表在不同后端语义不一致。

## 13. 内存与性能

### 13.1 内存

- KV 数据：每层 `n_slots * n_kv_head * kv_head_dim * 2 * 4` 字节。
- 元数据：
  - `meta`：`n_slots * sizeof(KvSlotMeta)`；
  - `free_stack`：`4 * n_slots`；
  - 请求表：`4 * ctx_size`；
  - `keybuf`：`4 * ctx_size`。
- rax 节点：每个缓存 token 一个 key 节点，总边字符约 `O(n)`。
- `n_slots > ctx_size` 才能保留额外分支；内存线性增长。

### 13.2 时间

- begin 匹配：一次 `raxFindPrefix`，`O(n)`；parent 链收集 `O(matchlen)`。
- decode：每次分配一个 slot，`O(1)`；不每步插入树。
- finish 插入：每个非最后 token 一次 `raxInsert`，单次 `O(前缀长度)`，最坏
  `O(n^2)` 字节比较。对 `ctx_size = 4096` 通常可接受；如果偏慢，可以改成
  block checkpoint 或增量 rax 游标。
- 驱逐：第一版 `O(n_slots)` 扫描找最旧叶子；分叉多、池子大时可优化为叶子堆。
- op_attn gather：每个 KV head 多一次 `(pos+n) * khd` 的 K/V 拷贝；长上下文
  和共享前缀长的场景收益远大于拷贝成本。

### 13.3 统计

建议记录并按 INFO 输出：

- `hit_tokens` / `total_tokens`
- `computed_tokens`
- `evictions`
- `slot_used` / `n_slots`

例如：

```
kvcache: prefix hit 1024/1032 tokens (skip 1023, pool 1024/2048, evict 0)
```

注意 skip 会比 hit 少 1（最后一个 token 私有重算）。

## 14. 测试计划

### 14.1 rax 单测

新增 `test/c/test_rax.c`：

- 精确 key。
- 跨分支最长前缀。
- 压缩节点分裂后的前缀匹配。
- key 落在链中间 / 叶子。
- 空树。
- `matchlen` 字节数正确。

### 14.2 kvcache / slot pool 单测

新增 `test/c/test_kvcache.c`，fake cache 不需要真实模型：

- `kvcache_init/free`，disabled 模式。
- slot alloc/free，`n_slots` 耗尽。
- 插入 A→B→C，匹配后通过 parent 链得到 `[slotA, slotB, slotC]`。
- 多分支：ABC 与 ABD 同时存在，分别匹配，A/B 共享 slot。
- 最后 token 私有：finish 后树里只有 A、B；再次请求 ABC 命中 AB，
  最后一个 C 重算。
- 全命中：begin 的 `use = n - 1`，最后一个 token 分配新 slot。
- LRU：访问顺序影响驱逐顺序。
- lock_ref：活动请求的前缀不会被驱逐。
- abort：释放私有 slot、解锁前缀，树不新增节点。
- finish 插入失败回滚（可用注入 OOM / mock）。
- `slot == 0` 的 value 编码正确（`slot + 1`）。
- 池满时 `reserve` 驱逐叶子并成功分配。

### 14.3 op_attn 单测

关键测试：**非连续 slot 表与连续 slot 表结果一致**。

- 构造一个小 attention，逻辑位置 `[0,1,2,3]` 分别映射到 slot
  `[7,3,9,12]`；
- 按 slot 写入 K/V；
- 用请求表执行 attention；
- 与连续映射 `[0,1,2,3]` 的结果比较，必须一致（或极小浮点误差）。
- 覆盖 GQA 和 sliding window（如果启用）。

### 14.4 模型级验证

用纯 attention 模型（Qwen2 / Llama / Falcon / Gemma3 任一）：

1. 同一 prompt 连续跑两次，第二次日志显示命中 `n-1` 个 token。
2. 对比“关闭缓存”和“开启缓存”的 logits：
   - 允许小的浮点差异（分块 matmul 与单 token matvec 的累加顺序不同）；
   - 建议阈值：`max_abs_diff < 1e-3`，且 top-1 token 一致。
3. 多分支：先请求 ABC，再请求 ABD，再请求 ABC；三个请求都正确，A/B 前缀
   共享，C/D 各自独立。
4. 多轮：system prompt 相同的两个请求，第二个命中 system 前缀。
5. 失败注入：begin 后让 forward 失败，再发新请求，结果必须正确。
6. 池驱逐：把 `n_slots` 调到较小，触发 LRU，验证活动请求前缀不被驱逐。

### 14.5 端到端

server 模式：

- 连发两个相同/共享长 system prompt 的请求，第二个 TTFT 明显下降；
- 日志出现 prefix hit；
- 输出与无缓存路径一致；
- 多个分支共存时（pool 足够）不会互相覆盖。

CUDA/Metal（indexed kernel 落地后）：

- 与 CPU 版本 logits 对齐；
- 非连续 slot 表正确；
- 请求表上传路径无竞态。

## 15. 验收标准

- `make`、`make check`、`DEBUG=1 make check` 全部通过，零警告。
- `test_rax`、`test_kvcache`、op_attn parity 测试覆盖第 14 节所有用例。
- 纯 attention 模型：
  - 相同 prompt 第二次请求命中 `n-1` 个 token；
  - logits 与无缓存路径在容差内一致；
  - 多分支可以同时存在，互不覆盖；
  - 驱逐只发生在无锁叶子上。
- Qwen3.5 / Falcon-H1 / DeepSeek：
  - `enabled = false`；
  - 行为与改动前完全一致；
  - reset 仍然正确清零 recurrent state。
- 失败路径不会留下悬挂 slot、锁泄漏或半插入节点。
- 上下文满时由 `graph_execute` 拒绝，不静默丢上下文。
- 所有请求结束后 slot 不泄漏；`kvcache_free` 后无残留分配。

## 16. 风险与待确认项

1. **gather 成本**：CPU 上每个 KV head 多一次 K/V 拷贝；需要 benchmark 确认
   收益大于成本。长上下文、共享前缀长的场景收益最大。
2. **池容量**：`n_slots` 太大浪费内存/显存，太小频繁驱逐；默认 `ctx_size`，
   需要多分支保留时调大。
3. **最后 token 私有**：每个序列少缓存 1 个 token；必须保证不往共享 slot 写。
4. **finish 插入成本**：`O(n^2)` 字节比较；极长上下文下可能成为瓶颈。
5. **驱逐扫描成本**：第一版 `O(n_slots)`；后续可维护叶子堆。
6. **设备请求表**：CUDA/Metal 需要上传 slot 表；split plan 需要多份副本。
7. **device kernel**：indexed attention 未实现前，device 层不能启用前缀缓存。
8. **多 session / 并发**：当前单 session；如果以后共享 pool，需要锁、引用计数
   和跨 session 驱逐策略。
9. **混合架构 / MLA**：第一版关闭；未来需要 recurrent state checkpoint。
10. **page_size > 1**：当前 token 级；page/block 化是后续工作。
11. **key 一致性**：如果未来加入 LoRA / multimodal / cache salt，key 必须包含
    这些 extra key，避免错误共享。
12. **OOM 回滚**：finish 插入失败必须回滚已插入节点，否则树会指向已释放 slot。

## 17. 后续工作

- CUDA/Metal indexed/paged attention kernel。
- page_size > 1 / block table。
- device / host 分层 KV。
- 叶子堆优化 LRU 驱逐。
- 多 session / 并发共享 pool（锁 + 引用计数）。
- 混合架构的 recurrent state checkpoint。
- MLA 前缀缓存。
- 缓存 prompt 结尾 hidden state / logits，做到全命中零 forward。
- 增量插入未完成请求，支持并发请求共享正在生成的 KV。
