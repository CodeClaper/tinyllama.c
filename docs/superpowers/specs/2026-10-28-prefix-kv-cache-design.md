# 基于 rax 的 KV 前缀缓存（graph executor 版）设计

日期：2026-10-28
状态：Draft，待确认

## 1. 背景与目标

server 模式在每个请求开始时会 `reset` 会话，然后重新 prefill 整段 prompt
（`src/server.c`）。如果多个请求共享同一段前缀（常见的是相同的 system
prompt、相同的多轮历史、或完全相同的 prompt），这段前缀的 KV 会被重复
计算。

本设计用 rax 放射树记录 **token 前缀 → KV 缓冲槽位**，在请求开始时做最长
前缀匹配，命中部分直接复用已算好的 KV，只对后缀执行 forward。

目标：

- 相同/部分相同 prompt 的 prefill 跳过重复计算。
- 不复制、不额外分配 KV 数据；rax 只保存元数据。
- 与现有 graph executor（`src/graph.c`）、`GraphPlan.cache`、CPU/CUDA/Metal
  三种后端兼容。

非目标（第一版不做）：

- paged KV / block table。
- 多个不同分支的 KV 同时常驻。
- LRU 驱逐。
- 混合架构（Gated DeltaNet / Mamba2）的 recurrent state 快照。
- MLA 前缀缓存。
- 多 session / 并发共享缓存。

## 2. 现状与约束

以下事实决定了本方案的形状：

1. **KV 缓存在 `GraphPlan` 里，不在 `Session` 里**
   - `GraphPlan.cache` 是 `KvCache *`（`src/def.h`）。
   - 标准注意力由 `cache_build_attn_kvcache()` 分配（`src/graph.c`），每层一份
     `AttnKvCache.k/v`，容量为 `ctx_size`，布局为
     `[head][position][kv_head_dim]`。
   - `cache_build()` 在 `graph_plan()` 内被调用，plan 是懒加载的。

2. **`op_attn` 用逻辑位置直接索引 KV 槽位**
   - `op_attn` 使用 `c->pos`、`c->n` 和 `akc->cap`（CPU/CUDA/Metal 三份实现
     一致）。
   - 因此第一版必须保持不变量：
     **活动序列第 i 个 token 的 KV 就位于槽位 i**。
     只有这样，`c->pos` 既能当 RoPE 绝对位置，又能当 KV 槽位。

3. **`graph_execute` 以 `s->n_tokens` 作为起始位置**
   - 内部按 `n_ubatch` 分块，每块调用 `graph_compute()`。
   - 结束时 `s->n_tokens = pos + n_tokens`。
   - 前缀匹配必须发生在进入分块循环之前。

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
| 分支能力 | 单活动分支 | 单份连续 KV 缓冲，分叉时覆盖旧后缀 |
| 全命中 | 至少重算最后一个 token | KV 里没有 hidden state / logits，采样需要 logits |
| 缓存内容 | prompt + 生成结果 | decode 时逐 token 挂树，多轮更容易命中 |
| 并发 | 单 session | server 当前是单线程单 session |
| 位置与槽位 | 一一对应 | 不引入 block table，不改 `op_attn` 寻址 |
| 驱逐 | 写前驱逐 | 避免 forward 失败后留下悬空引用 |
| 满上下文 | 由 `graph_execute` 拒绝 | 不做“静默 flush 后继续生成” |

## 4. 总体方案

```
token 前缀（4 字节大端/token）        KV 缓冲（每层连续）
        t0          ------------->    slot 0  K/V
        t0 t1       ------------->    slot 1  K/V
        t0 t1 t2    ------------->    slot 2  K/V
        ...                           ...

rax: 每个前缀都是一个 key 节点
     node value = 该 token 的 KV 槽位（存 slot+1，避免 0 与 NULL 混淆）
```

请求流程：

```
reset（清活动路径，保留树）
    |
    +--> 最长前缀匹配
    |
    +--> 驱逐 [write_start, max_pos)
    |
    +--> 只对后缀跑 graph_execute
    |
    +--> 后缀逐 token 挂树
    |
    +--> decode: 每生成一个 token 追加一个节点
```

### 4.1 为什么节点值能直接给出槽位

不变量保证：一条活路径上第 i 个 token 的槽位就是 i。因此：

- 匹配到深度为 `matchlen` 的 key 节点；
- 该节点的槽位是 `matchlen - 1`；
- 后缀从槽位 `matchlen` 开始写。

如果未来要让槽位和逻辑位置解耦（paged KV、多分支），必须改 `op_attn` 的
寻址方式；第一版不做。

### 4.2 单缓冲带来的限制

分叉时旧后缀会被覆盖并驱逐，因此 **同一时刻只保留一条活动分支**：

```
请求1: A B C -> 生成 X Y
请求2: A B D
命中 A B，D 写到 slot 2，C/X/Y 的 KV 和树节点全部失效
```

这不是 SGLang 那种多分支常驻的 RadixAttention，而是“RadixAttention-lite”。
它适合：

- 相同 system prompt + 不同 user message；
- 新 prompt 以旧 prompt 为前缀的多轮场景；
- 完全相同的 prompt。

它不适合：

- 多个不同历史分支需要同时常驻；
- 多个并发会话共享 KV。

在“单活动分支”前提下，radix tree 实际上会退化成一条路径；rax 的主要价值
是为未来多分支/多 session 留结构。若确认永远不需要多分支，简单的
`cached_tokens[] + LCP` 也足够。

## 5. 数据结构与接口

### 5.1 `src/kvcache.h`

```c
typedef struct KvPrefixCache {
    rax     *tree;       /* token 前缀 -> KV slot；disabled 时为 NULL */
    u32      pos;        /* 当前活动序列长度 = 下一个写入槽 */
    u32      max_pos;    /* 活跃缓存区域的最高边界（下一个空闲槽） */
    bool     enabled;
    KvCache *kv;         /* 指向 GraphPlan.cache */
    u32      n_layer;
    u32      cap;        /* = ctx_size */
    unsigned char *curkey;   /* 当前完整 token 路径，增量扩展 */
    size_t       curkey_len;
    size_t       curkey_cap;
} KvPrefixCache;
```

接口：

```c
bool kvcache_init(KvPrefixCache *pc, KvCache *kv, u32 n_layer, u32 cap,
                  bool enabled);
void kvcache_free(KvPrefixCache *pc);
void kvcache_flush(KvPrefixCache *pc);   /* 清树 + pos/max_pos，保留 KV 缓冲 */
void kvcache_reset(KvPrefixCache *pc);   /* 清活动路径，保留树和 max_pos */

/* 匹配 + 截断 + 写前驱逐。返回实际可跳过的 token 数（skip）。 */
bool kvcache_prepare(KvPrefixCache *pc, const u32 *tokens, u32 n, u32 *skip);

/* forward 成功后，把 [skip, n) 的节点挂进树。 */
bool kvcache_commit(KvPrefixCache *pc, const u32 *tokens, u32 n, u32 skip);

/* decode：在 pc->pos 处已经写完 KV，追加一个节点。 */
bool kvcache_append(KvPrefixCache *pc, u32 token);

/* chat 续接：一次追加多个 token（活动路径已经存在）。 */
bool kvcache_append_batch(KvPrefixCache *pc, const u32 *tokens, u32 n);
```

`kvcache_flush()` 只在元数据不一致、OOM、显式重置等异常路径使用；正常
上下文满由 `graph_execute` 的 `ctx_size` 检查处理，不静默 flush。

### 5.2 key 编码

每个 token 编码成 4 字节大端：

```
t0          -> [00 00 00 01]
t0 t1       -> [00 00 00 01][00 00 00 02]
t0 t1 t2    -> [00 00 00 01][00 00 00 02][00 00 00 03]
```

key 长度 = `4 * token_count`。大端只是为了调试可读；关键是固定宽度、无分隔
符、前缀性质成立。

每个前缀都插入一个 key 节点：

```
key = tokens[0..i]   (i = 0..n-1)
value = slot_i + 1
```

### 5.3 value 编码

存 `slot + 1`：

```c
value = (void *)(uintptr_t)(slot + 1);
slot  = (u32)(uintptr_t)value - 1;
```

原因：`slot == 0` 时 `NULL` 会被 rax 当成 `isnull`，虽然 `NULL != raxNotFound`
仍能区分，但容易在调用侧误判。`slot + 1` 让所有值非 NULL，语义更干净。

### 5.4 所有权

`KvPrefixCache` 的生命周期绑定到 `GraphPlan`，因为它的 `kv` 指针指向
`GraphPlan.cache`：

```c
/* def.h */
typedef struct KvPrefixCache KvPrefixCache;

typedef struct GraphPlan {
    ...
    KvPrefixCache *prefix;   /* 与 cache 同生命周期 */
} GraphPlan;
```

- 在 `graph_plan()` 完成 `cache_build()` 和 `graph_warmup()` 之后创建。
- 在 `cache_destroy()` 中、释放 KV arena 之前销毁。
- 不放进 `Session`，避免 graph 重建/释放后出现悬空指针。

对外访问辅助：

```c
KvPrefixCache *graph_prefix_cache(Session *s);
```

返回 `s->graph && s->graph->plan ? s->graph->plan->prefix : NULL`。

## 6. rax 扩展：`raxFindPrefix`

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

返回值可能为 `NULL`（如果使用 slot 编码且 slot 0）——本设计用 `slot + 1`，
不会出现这种数据值。

## 7. 生命周期

### 7.1 init

由 `graph_plan()` 在 plan 成功后调用：

```c
bool enabled = kvcache_arch_supported(s->en->model->arch);
g->plan->prefix = smalloc(sizeof(*g->plan->prefix));
if (!g->plan->prefix ||
    !kvcache_init(g->plan->prefix, g->plan->cache, s->cfg.n_layer,
                  s->ctx_size, enabled)) {
    /* 失败：释放 prefix，plan_reset，返回 false */
}
```

`kvcache_arch_supported()` 默认返回：

```
Llama / Qwen2 / Falcon / Gemma3 -> true
Qwen3.5 / Falcon-H1 / DeepSeek -> false
```

### 7.2 free

`cache_destroy()` 中：

```c
if (p->prefix) {
    kvcache_free(p->prefix);
    sfree(p->prefix);
    p->prefix = NULL;
}
/* 然后才释放 p->cache 和 cache_arena */
```

`kvcache_free()` 只释放 rax 树和 `curkey`，不释放 KV。

### 7.3 reset

`kvcache_reset()` 只清活动路径：

```c
pc->curkey_len = 0;
pc->pos = 0;
for (u32 i = 0; i < pc->n_layer; i++)
    pc->kv->std[i].n = 0;
```

**不清树，不清 `max_pos`**：树里的节点仍然指向有效的 KV 槽位，`max_pos`
仍然描述活跃缓存区域的上界。

各架构 reset：

- Llama / Qwen2 / Falcon / Gemma3：
  `kvcache_reset(graph_prefix_cache(s)); s->n_tokens = 0;`
- Qwen3.5 / Falcon-H1 / DeepSeek：
  `kvcache_reset(...)`（disabled 时是空操作），保留原有 recurrent state
  清零逻辑，`s->n_tokens = 0`。

## 8. Prefill 流程

前提：server 已经 reset，`s->n_tokens == 0`，树和 KV 保留。

```c
u32 skip = 0;
if (!kvcache_prepare(pc, prompt, n, &skip))
    return false;

/* 完整 prompt 必须进 s->tokens，采样 / repeat penalty 会读它 */
memcpy(s->tokens, prompt, (size_t)n * sizeof(u32));

/* 只跑后缀 */
s->n_tokens = skip;
if (!graph_execute_raw(s, prompt + skip, n - skip, logits))
    return false;

if (!kvcache_commit(pc, prompt, n, skip))
    return false;
```

### 8.1 `kvcache_prepare`

```c
bool kvcache_prepare(KvPrefixCache *pc, const u32 *tokens, u32 n, u32 *skip) {
    *skip = 0;
    if (!pc || !pc->enabled || n == 0) return true;

    /* 1. 编码全量 query key */
    build_key(pc, tokens, n);

    /* 2. 最长前缀匹配 */
    size_t klen = 0;
    void *data = raxFindPrefix(pc->tree, pc->curkey, pc->curkey_len, &klen);

    u32 matchlen = (u32)(klen / 4);
    u32 match_slot = (data == raxNotFound)
                   ? 0
                   : (u32)(uintptr_t)data - 1;

    /* 3. 不变量检查：活路径必须槽位 == 逻辑位置 */
    if (data != raxNotFound && match_slot + 1 != matchlen) {
        slog(WARN, "kvcache: slot/position mismatch, flushing");
        kvcache_flush(pc);
        matchlen = 0;
    }

    /* 4. 至少重算最后一个 token，否则拿不到 logits */
    u32 use = (matchlen >= n) ? n - 1 : matchlen;

    /* 5. 写前驱逐：
     *    - matchlen < n : 后缀会写 [matchlen, n)，驱逐 slot >= matchlen
     *    - matchlen == n: 只重算 slot n-1，保留它；驱逐 slot >= n
     */
    u32 evict_from = (matchlen >= n) ? n : use;

    if (evict_from < pc->max_pos) {
        kvcache_evict_from(pc, evict_from);
        pc->max_pos = evict_from;
    }

    pc->pos = use;
    *skip = use;
    return true;
}
```

注意：

- 驱逐必须在 forward **之前**完成。
- 如果 forward 失败，树里只剩 `slot < evict_from` 的有效节点，不会留下指向
  被覆盖 KV 的悬空引用。
- `kvcache_commit()` 只负责添加新节点，不再做驱逐。

### 8.2 `kvcache_commit`

```c
bool kvcache_commit(KvPrefixCache *pc, const u32 *tokens, u32 n, u32 skip) {
    if (!pc || !pc->enabled) return true;

    build_key(pc, tokens, n);

    for (u32 i = skip; i < n; i++) {
        /* 槽位 == 逻辑位置 i */
        void *v = (void *)(uintptr_t)(i + 1);
        if (raxInsert(pc->tree, pc->curkey, (size_t)(i + 1) * 4, v, NULL) != 1
            && errno == ENOMEM)
            return false;
    }

    pc->pos = n;
    if (n > pc->max_pos) pc->max_pos = n;
    for (u32 i = 0; i < pc->n_layer; i++)
        pc->kv->std[i].n = n;
    return true;
}
```

全命中时 `skip = n - 1`，会重新插入最后一个 token 的 key（value 不变）；
这是幂等的，不会破坏树。

### 8.3 为什么全命中还要重算最后一个 token

KV 只保存 K/V，不保存最后一个 token 的 hidden state。要采样下一个 token，
必须得到 logits；logits 来自最后一个 token 的 forward。因此：

```c
use = (matchlen >= n) ? n - 1 : matchlen;
```

收益仍然是 O(n) prefill 变成 O(1) decode。

如果以后要求“全命中零计算”，需要在缓存里额外保存每个请求结尾的 hidden
state 或 logits；本版不做。

## 9. Decode 流程

每次 decode 只处理 1 个 token：

```c
u32 pos_before = s->n_tokens;
if (!graph_execute_raw(s, &token, 1, logits))
    return false;
if (pc && pc->enabled && s->n_tokens == pos_before + 1)
    kvcache_append(pc, token);
```

`kvcache_append`：

```c
bool kvcache_append(KvPrefixCache *pc, u32 token) {
    if (!pc || !pc->enabled) return true;
    if (pc->pos + 1 > pc->cap) return false;   /* 上下文满，由调用方停止生成 */

    if (!key_grow(pc, pc->curkey_len + 4)) return false;
    key_encode_append(token, pc->curkey, pc->curkey_len);
    pc->curkey_len += 4;

    u32 slot = pc->pos;
    void *v = (void *)(uintptr_t)(slot + 1);
    if (raxInsert(pc->tree, pc->curkey, pc->curkey_len, v, NULL) != 1
        && errno == ENOMEM)
        return false;

    pc->pos = slot + 1;
    if (pc->pos > pc->max_pos) pc->max_pos = pc->pos;
    for (u32 i = 0; i < pc->n_layer; i++)
        pc->kv->std[i].n = pc->pos;
    return true;
}
```

只有 `graph_execute_raw` 成功后才 append。失败时 KV 槽位可能已经写了，但
树里没有对应节点，不会被后续请求读取。

## 10. Chat 续接

chat 模式正常续接时不清空会话，活动路径已经存在，新 prompt 是追加的：

```c
u32 pos_before = s->n_tokens;
if (!graph_execute_raw(s, tokens, n, logits))
    return false;
if (pc && pc->enabled)
    kvcache_append_batch(pc, tokens, n);
```

`kvcache_append_batch` 对每个 token 调用一次 `kvcache_append` 的逻辑，保持
`curkey` 连续。它不会重新从位置 0 构建 key。

`/clear` 走 reset：清活动路径，保留树；下一条消息如果是旧分支的前缀，可以
直接命中。

## 11. 驱逐规则

### 11.1 不变量

- 活动路径上的 KV 槽位严格为 `0, 1, 2, ..., pos-1`。
- 树节点的 `slot` 等于其 token 在路径中的逻辑位置。
- 任何写入都从 `skip` 开始向后连续延伸。

### 11.2 规则

写后缀之前：

```
if evict_from < max_pos:
    删除所有 slot >= evict_from 的树节点
    max_pos = evict_from
else:
    O(1)，不扫树
```

其中：

- 普通分叉：`evict_from = matchlen = skip`。
- 全命中（`matchlen >= n`）：`evict_from = n`，保留最后一个 token 的节点，
  只清掉它之后的旧生成后缀。

### 11.3 例子

```
请求1: A B C，生成 X Y
槽位:  A0 B1 C2 X3 Y4
树:    A(0) AB(1) ABC(2) ABCX(3) ABCXY(4)

请求2: A B D
matchlen = 2, skip = 2, evict_from = 2
删除 slot >= 2: C(2) X(3) Y(4)
写入 D 到 slot 2
commit: A0 B1 ABD(2)
```

### 11.4 失败原子性

```
prepare:
    匹配 + 驱逐（树只保留 < evict_from 的节点）
forward:
    可能失败；失败时树仍然一致
commit:
    只在成功后添加节点
```

不允许“先写 KV，再在 commit 里驱逐”的顺序，因为 forward 失败会让树指向被
覆盖的槽位。

### 11.5 满上下文

`graph_execute_raw` 已有检查：

```c
if ((u64)pos + n > s->ctx_size) return false;
```

因此：

- prefill 超过 `ctx_size`：直接失败。
- decode 到达 `ctx_size`：server/chat 的循环会停止生成，不继续调用。
- 不做“flush 后从 0 继续生成”——那会静默丢掉上下文，语义错误。

## 12. graph executor 集成

### 12.1 函数拆分

把当前 `graph_execute()` 的函数体改名为：

```c
static bool graph_execute_raw(Session *s, const u32 *tokens, u32 n,
                              float *logits);
```

新增对外的：

```c
bool graph_execute(Session *s, const u32 *tokens, u32 n, float *logits);
```

`graph_execute()` 负责缓存逻辑，`graph_execute_raw()` 只做原来的分块 forward。

### 12.2 包装逻辑

```c
bool graph_execute(Session *s, const u32 *tokens, u32 n, float *logits) {
    if (!s || !tokens || n == 0) return false;

    /* 沿用原有 pos + n <= ctx_size 检查 */
    if ((u64)s->n_tokens + n > s->ctx_size) return false;

    KvPrefixCache *pc = graph_prefix_cache(s);
    bool fresh = (s->n_tokens == 0);
    u32 skip = 0;

    if (fresh && pc && pc->enabled) {
        if (!kvcache_prepare(pc, tokens, n, &skip))
            return false;

        /* 完整 prompt 进 ring buffer，采样会用到 */
        memcpy(s->tokens, tokens, (size_t)n * sizeof(u32));
        s->n_tokens = skip;
    }

    if (!graph_execute_raw(s, tokens + skip, n - skip, logits))
        return false;

    if (pc && pc->enabled) {
        if (fresh)
            return kvcache_commit(pc, tokens, n, skip);
        if (n == 1)
            return kvcache_append(pc, tokens[0]);
        return kvcache_append_batch(pc, tokens, n);
    }
    return true;
}
```

说明：

- `fresh` 由 `s->n_tokens == 0` 判断；server 每次请求前 reset 正好满足。
- chat 正常续接时 `s->n_tokens > 0`，走 append/append_batch，不做前缀匹配。
- `/clear` 后 `s->n_tokens == 0`，重新走前缀匹配。
- 如果 `pc->enabled == false`，行为与当前完全一致。

### 12.3 位置与 token 缓冲

- fresh prefill 时，`s->tokens[0..n)` 必须复制完整 prompt，因为
  `graph_execute_raw` 只写后缀。
- `s->n_tokens = skip` 让 RoPE 和 `op_attn` 使用正确的绝对位置。
- `graph_execute_raw` 结束时把 `s->n_tokens` 推到 `n`。

## 13. 后端兼容性

前缀缓存只操作元数据（rax、pos、max_pos、curkey），不碰 KV 数据：

- CPU：`GraphPlan.cache` 是 host 内存。
- CUDA：每层 cache 可能位于 VRAM；`op_attn` 用 `c->pos` 和 `akc->k/v` 指针。
- Metal：同 CUDA。
- split plan：不同层可以位于不同后端，缓存逻辑不感知。

只要保持“槽位 == 逻辑位置”，三种后端都不需要修改 `op_attn`。

## 14. 内存与性能

### 14.1 内存

- KV 数据：无额外副本，仍是一份 `ctx_size` 缓冲。
- rax 元数据：每个缓存 token 一个 key 节点；总边字符数约 `O(n)`，节点开销
  与 token 数成正比。
- `curkey`：最多 `4 * ctx_size` 字节，可增长。
- `max_pos` 限制在 `ctx_size` 内。

### 14.2 时间

- 匹配：一次 `raxFindPrefix`，`O(n)`。
- 驱逐：仅在分叉时扫描树，`O(树节点数)`，最多 `O(n)`；连续扩展是 `O(1)`。
- commit：每个后缀 token 一次 `raxInsert`，单次 `O(前缀长度)`，最坏
  `O(n^2)` 字节比较。对 `ctx_size = 4096` 通常可接受；如果实测偏慢，可以
  只在 block 边界（例如每 16 个 token）挂 checkpoint，用匹配精度换插入速度。
- 收益：命中 `m` 个 token 时，跳过 `m` 个 token 的 transformer forward。

### 14.3 统计

建议记录并按 INFO 输出：

- `hit_tokens` / `total_tokens`
- `computed_tokens`
- `evictions`
- `flushes`

例如：

```
kvcache: prefix hit 1024/1032 tokens (skip 1023, evict 0)
```

注意日志里的 skip 会比 hit 少 1（最后一个 token 重算）。

## 15. 测试计划

### 15.1 rax 单测

新增 `test/c/test_rax.c`：

- 精确 key。
- 跨分支最长前缀。
- 压缩节点分裂后的前缀匹配。
- key 落在链中间 / 叶子。
- 空树。
- `matchlen` 字节数正确。

### 15.2 kvcache 单测

新增 `test/c/test_kvcache.c`，使用 2 层、1 个 kv head、head_dim 4、cap 16
的 fake cache：

- init / free / disabled。
- 精确命中、部分命中、无命中。
- 连续扩展：`skip == max_pos`，零驱逐。
- 分叉驱逐：ABC + XY 后插入 ABD，C/X/Y 节点消失。
- 全命中：`skip = n - 1`，最后一个 token 节点保留，commit 后 `pos = n`。
- `slot == 0`：value 为 1，不会被当成 NULL。
- reset 保留树和 `max_pos`。
- decode 逐 token 挂树。
- chat `append_batch`。
- prepare 驱逐后 forward 失败、不 commit：下一次匹配只能命中前缀，不能命中
  被覆盖的后缀。
- 上下文溢出：`graph_execute` 拒绝，不静默 flush。
- disabled 模式：不建树，行为与旧路径一致。

### 15.3 模型级验证

用纯 attention 模型（Qwen2 / Llama / Falcon / Gemma3 任一）：

1. 同一 prompt 连续跑两次，第二次日志显示命中 `n-1` 个 token。
2. 对比“关闭缓存”和“开启缓存”的 logits：
   - 允许小的浮点差异（分块 matmul 与单 token matvec 的累加顺序不同）；
   - 建议阈值：`max_abs_diff < 1e-3`，且 top-1 token 一致。
3. 分叉：先请求 ABC，再请求 ABD，再请求 ABC；第三次只能命中 AB，不能读到
   已被 D 覆盖的 C。
4. 多轮：system prompt 相同的两个请求，第二个命中 system 前缀。
5. 失败注入：prepare 后让 forward 失败，再发新请求，结果必须正确。

### 15.4 端到端

server 模式连发两个相同/共享长 system prompt 的请求：

- 第二个请求 TTFT 明显下降；
- 日志出现 prefix hit；
- 输出与无缓存路径一致。

## 16. 验收标准

- `make`、`make check`、`DEBUG=1 make check` 全部通过，零警告。
- `test_rax`、`test_kvcache` 覆盖第 15.1 / 15.2 节所有用例。
- 纯 attention 模型：
  - 相同 prompt 第二次请求跳过 `n-1` 个 token；
  - logits 与无缓存路径在容差内一致；
  - 分叉后不会读到旧后缀。
- Qwen3.5 / Falcon-H1 / DeepSeek：
  - `enabled = false`；
  - 行为与改动前完全一致；
  - reset 仍然正确清零 recurrent state。
- 失败路径不会在树里留下指向被覆盖 KV 的节点。
- 上下文满时由 `graph_execute` 拒绝，不静默丢上下文。

## 17. 风险与待确认项

1. **单分支限制**：本方案不能同时保留多个不同历史分支。若产品需要多个会话
   并行共享缓存，应改为 paged KV + block table。
2. **全命中仍需 1 token forward**：如果要求 TTFT 严格为 0，需要额外缓存
   hidden state 或 logits。
3. **混合架构**：Qwen3.5 / Falcon-H1 的 recurrent state 不能只靠 KV 复用；
   第一版关闭。未来要支持，需要在每个 checkpoint 保存 SSM/conv state 快照。
4. **MLA**：第一版关闭。
5. **并发**：当前单 session 单线程；如果未来多 session 共享同一棵树，需要
   锁和引用计数，或改为每 session 独立缓存。
6. **commit 的 `O(n^2)` 插入成本**：极长上下文下如果成为瓶颈，可改为 block
   checkpoint 或增量 rax 游标。
7. **槽位/位置不变量**：任何未来改动（paged KV、多分支、滑动窗口特殊布局）
   都必须重新审视 `op_attn` 的索引方式。
8. **rax 值编码**：坚持 `slot + 1`，避免 slot 0 与 NULL 混淆。

## 18. 后续工作

- paged KV / block table，实现真正的多分支 RadixAttention。
- LRU / 引用计数驱逐。
- 缓存 prompt 结尾的 hidden state，做到全命中零 forward。
- 混合架构的 recurrent state checkpoint。
- MLA 前缀缓存。
- 多 session / 并发缓存共享。
