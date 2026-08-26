# 基于 rax 的 KV 前缀缓存（RadixAttention-lite）设计

日期：2026-08-26
状态：已获批准（4 节逐节确认）

## 背景与目标

server 模式每次请求都 `reset` 后全量 prefill 整段对话（`server.c`），多轮对话时每轮都重复计算全部历史 KV。chat 模式是续接式，收益较小，但用户编辑/分支对话时同样浪费。

目标：用 rax 放射树实现 token 前缀 → KV 缓存段的映射（SGLang RadixAttention 的轻量版），让带相同前缀的 prefill 跳过重复计算。两场景（server 多请求复用 + chat 多轮分支）共用同一套基础设施。

非目标（YAGNI）：紧凑化/搬移、LRU 驱逐、paged block、MLA 前缀缓存——全部留给后续。

## 现状

- `kvcache.c/h` 为空占位；`AttnKvCache`（逐层连续 k/v 缓冲 + n/cap）与 `KvCache` union 已定义于 `def.h`
- 所有后端 `prefill/generate` 是 stub，推理未实现——KV 缓冲布局现在定，无迁移成本
- rax 为完整 vendored Redis 实现，含 `raxInsert/raxRemove/raxFind/raxTouch/raxFreeWithCallback`，API 见 `rax.h`

## 方案

- **token 级边**：每个 token 一条树边，key = token id 的 4 字节大端编码拼成的字节串；**每个前缀都是 key 节点**，值 = `u32 pos`（该节点自身那个 token 的 KV 在缓冲中的位置）。所有层形状相同，一个 pos 全层通用
- **连续缓冲 + 满则清空**：不紧凑化，`max_pos == cap` 时整体 flush
- **rax 小扩展**：新增 `raxFindPrefix(rax, key, len, &matchlen)` → 查询路径上最深的 key 节点（复用内部 `raxLowWalk`），并返回其 key 长度。这是对 vendored rax 的唯一改动

## 数据结构

```c
/* kvcache.h */
typedef struct {
    rax     *tree;      /* token 序列 → 缓存段；enabled=false 时为 NULL */
    u32      pos;       /* 当前活动序列写位置（= 活动长度） */
    u32      max_pos;   /* 缓冲历史最高写位置 */
    bool     enabled;   /* MLA 架构为 false */
    KvCache *kv;        /* 指向 Session.cache（缓冲归 arch 管理） */
    unsigned char *curkey; /* 当前路径 key（growable，decode 增量扩展） */
    size_t       curkey_len;
    size_t       curkey_cap;
} KvPrefixCache;
```

- 节点值 = 单个 `u32 pos`（无需 len：token 级边下每节点贡献恰好 1 个 token 的 KV），直接 `(void *)(uintptr_t)pos` 存指针，**零分配**
- `def.h` 只加 `KvPrefixCache` 前向声明 + Session 一个字段；缓冲结构体留在 def.h 不动
- key 编码：`token → 4 字节大端`（树只在内存中，用 BE 便于调试）

**关键不变量（路径连续）**：驱逐规则保证活段永不被覆写（分叉时 `[match_end, max_pos)` 全部驱逐，写游标只会落在死区），且写入总是从 match_end 连续延伸——因此命中段 `[match_end - matchlen, match_end)` 必然连续，后端可直接按连续区间计算 attention。

## 数据流

**prefill(s, tokens, n) 三步：**
1. `kvcache_prefix_match()` → `{matchlen, match_end}`（match_end = 命中最深节点的 pos+1；无命中时为 0）。**此处做 flush 检查**：`match_end + (n - matchlen) > cap` → 整体清空、返回 0/0
2. 后端只对后缀 `[matchlen, n)` 计算 KV，从 `match_end` 起写入
3. `kvcache_prefix_insert()`：若 `match_end < max_pos` 先做驱逐，再为后缀逐 token 挂边、写 `pos` 值，更新 `pos/max_pos/curkey`；rax 遇分叉自动节点分裂，已缓存分支不动

**generate(s, token) 一步：** 后端在 `pos` 处算一个 token 的 KV，然后 `kvcache_append()`：`pos + 1 > cap` → flush；否则 `curkey` 追加 4 字节、raxInsert 挂边、`pos++`

**效果**：server 请求 2 的 prompt 命中请求 1 的整条历史路径 → `matched_len == n`，全部 KV 计算跳过。chat 同理，`/clear` 后旧分支留树，重开即命中。

## 正确性：重叠写驱逐规则

连续缓冲 + 分叉复用，风险是**缓冲位置被覆盖而树里仍留旧引用**：

```
请求1: prompt ABC → 缓冲 [0,3)；生成 X,Y → [3,5)
树: A→(0), B→(1), C→(2), X→(3), Y→(4)   (key 分别是 A, AB, ABC, ABCX, ABCXY)

请求2: prompt ABD
最长命中: B(pos=1, matchlen=2)，match_end=2，D 写入位置 2 → 覆盖 C 的 KV
树里 C→(2) 仍指向位置 2 → 损坏
```

**驱逐规则（修正版，规划阶段发现原精确区间规则有漏洞）**：写游标从 `match_end` 起延伸——本请求的后缀会写 `[match_end, match_end+suffix)`，且**后续 generate 还会从 `pos` 继续写**。仅按后缀区间驱逐会被"空后缀 prefill 后紧接 generate"击穿（prefill 命中 AB 无后缀，随后 generate 写位置 2 覆盖 C）。因此：

```
若 match_end < max_pos：驱逐所有 pos >= match_end 的节点
若 match_end == max_pos：零驱逐（最常见的向后延伸路径，O(1)）
```

上例：D 写位置 2 → 驱逐 C(2)、X(3)、Y(4)（X、Y 所在路径因 C 被驱逐而失效，一并清理）；A(0)、B(1) 保留。树剩 A、B，D 写入后插 ABD 分支。正确。

**代价**：分叉时可能误伤 `[match_end, max_pos)` 内与本次写入不重叠的节点——它们所在路径随分叉断裂，驱逐是正确的；分叉是少数派，代价可接受。驱逐后位置变死区，可安全覆写。

**实现**：rax 迭代器全树扫描 + 检查值——仅在 `match_end < max_pos` 时触发，O(已缓存节点数)。`raxRemove` 支持删内部 key 节点（有子节点时只摘 key/value）。

**flush**：`match_end + (n - matchlen) > cap` 时整体清空（`raxFree` + 状态归零），打 WARN；append 侧 `pos + 1 > cap` 同样 flush。清空后本次请求全量重算（后端从位置 0 写起）。

## reset / 驱逐语义

- `reset(s)` 只清：tokens 环、`n_tokens`、`cur = NULL`。**不动树和缓冲**——server 每次请求前调 reset 正是缓存生效前提；chat `/clear` 同理，旧分支被新分叉自然驱逐
- 现有后端 stub 的 reset 保持清 `n_tokens` 即可

## MLA 处理

`enabled = false`：不走树，prefill 全量计算。MLA 压缩态缓存（attn_comp_kv 等）不是干净的逐 token 切片。标准注意力架构（Llama/Qwen2/Falcon）先拿收益。

## 后端接口变化

- `ArchOps.prefill/generate` 签名不变，后端内部调 kvcache API（操作 `KvPrefixCache *pc`，不依赖 Session）：
  - `kvcache_init(pc, kv, n_layer, cap, enabled)` / `kvcache_free(pc)` — session_create/free 调用；`kv` 指向已分配好的 `Session.cache`
  - `kvcache_reset(pc)` — 清 curkey、pos=0、各层 `n=0`；**不动树**
  - `kvcache_prefix_match(pc, tokens, n, &matchlen, &match_end)` — 含 flush 检查
  - `kvcache_prefix_insert(pc, tokens, n, matchlen, match_end)` — 含驱逐检查
  - `kvcache_append(pc, token)` — decode 挂边，含 flush 检查
- 打 INFO/WARN 统计 hit/miss，便于肉眼验证收益

## 测试

- 单测（纯 C，无需模型）：rax + 小维度缓冲模拟——
  - 最长前缀命中（含 `matched_len == n` 全命中）
  - 分叉驱逐正确性（第 3 节 ABD 场景）
  - 精确边界不误伤相邻分支
  - flush-on-full
  - decode 逐 token 挂边
  - reset 后复用旧路径
- 集成：server 同一对话连发两请求，第二请求 prefill 时间 ≈ 0

## 非目标（后续候选）

- 紧凑化（SGLang move-to-hole）
- raxTouch LRU 驱逐
- vLLM 式 block paged
- MLA 前缀缓存
