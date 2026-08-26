# rax KV 前缀缓存实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 rax 放射树把 token 前缀映射到 KV 缓存段，让带相同前缀的 prefill 跳过重复计算（server 多请求复用 + chat 多轮分支共用一套基础设施）。

**Architecture:** `kvcache.c/h` 实现 `KvPrefixCache` 管理器：rax 树（token 序列 → KV 缓冲位置）+ 连续逐层缓冲（归 arch 管理，kvcache 只持指针）。prefill 三步：`kvcache_prefix_match`（最长前缀命中，含 flush 检查）→ 后端只算后缀 → `kvcache_prefix_insert`（分叉时驱逐 + 挂边）。decode 每 token 调 `kvcache_append`。rax 树为 token 级边，每个前缀都是 key 节点，节点值 = `(void *)(uintptr_t)pos`（零分配）。

**Tech Stack:** C11, gcc, vendored rax（Redis radix tree），minunit 测试框架。

**Spec:** `docs/superpowers/specs/2026-08-26-prefix-kv-cache-design.md`

## Global Constraints

- key 编码：token → 4 字节大端；节点值 = 单个 `u32 pos` 存为 `(void *)(uintptr_t)pos`，**不堆分配**
- 驱逐规则：`insert` 时若 `match_end < max_pos`，驱逐所有 `pos >= match_end` 的节点；`match_end == max_pos` 时零驱逐
- flush：`match` 中 `match_end + (n - matchlen) > cap` 时整体清空并返回 0/0；`append` 中 `pos + 1 > cap` 时清空
- MLA（DeepSeek）：`enabled = false`，树为 NULL，各操作只推进 pos，不碰 `kv->mla`
- `reset` 只清 curkey_len、pos、各层 `n`；**不动树和缓冲**
- 测试风格：minunit，`test_utils.c` 的 main 模式（`MU_RUN_TEST` 列表 + `MU_REPORT()` + `return MU_EXIT_CODE`）
- 编译：`make`（release）、`DEBUG=1 make`（-Wall 断言）、`make check`；所有测试必须通过

---

### Task 1: raxFindPrefix（rax 最长前缀匹配）

**Files:**
- Modify: `src/rax.h`（声明，`raxFind` 之后）
- Modify: `src/rax.c`（实现，`raxLowWalk` 之后）
- Create: `test/c/test_rax.c`
- Modify: `test/Makefile`（`rax.o` 规则 + `test_rax` 目标 + `check` 接线）

**Interfaces:**
- Consumes: rax 内部宏 `raxGetData(node)`、`raxNodeFirstChildPtr(node)`（`rax.c` 内已有，无需导出）、`raxNotFound`
- Produces: `void *raxFindPrefix(rax *rt, unsigned char *s, size_t len, size_t *matchlen)` — 返回查询路径上最深 key 节点的值（`raxNotFound` 表示无命中），`*matchlen` = 命中的 key 长度（字节数）。Task 3 的 `kvcache_prefix_match` 依赖此函数。

**rax 节点 key 语义（实现依据，已用实验验证）：**
- 非压缩节点：key = 到达该节点的路径（自身的 char 是子边，下跳时消费）
- 压缩节点：存储的 key = 到达其起点的路径，**不含自身字符串**（字符串属于其下虚拟节点）
- 因此遍历时**每到达一个节点先检查 `iskey`**，再消费/下跳

- [ ] **Step 1: 写失败测试**

Create `test/c/test_rax.c`:

```c
#include <stdio.h>
#include "minunit.h"
#include "../../src/rax.h"

/* Insert key with an integer value; must be a fresh key. */
static void ins(rax *rt, const char *s, size_t len, uintptr_t v) {
    void *old = NULL;
    int rc = raxInsert(rt, (unsigned char *)s, len, (void *)v, &old);
    mu_check(rc == 1);
    mu_check(old == NULL);
}

static void expect_prefix(rax *rt, const char *q, size_t qlen,
                          uintptr_t v, size_t klen) {
    size_t m = 0;
    void *d = raxFindPrefix(rt, (unsigned char *)q, qlen, &m);
    mu_check(d != raxNotFound);
    mu_assert((uintptr_t)d == v, "value mismatch");
    mu_assert(m == klen, "matchlen mismatch");
}

static void expect_no_prefix(rax *rt, const char *q, size_t qlen) {
    size_t m = 99;
    void *d = raxFindPrefix(rt, (unsigned char *)q, qlen, &m);
    mu_check(d == raxNotFound);
    mu_check(m == 0);
}

MU_TEST(test_exact_key) {
    rax *rt = raxNew();
    ins(rt, "ABC", 3, 10);
    expect_prefix(rt, "ABC", 3, 10, 3);
    expect_no_prefix(rt, "AB", 2);      /* AB is not a key */
    expect_no_prefix(rt, "X", 1);
    raxFree(rt);
}

MU_TEST(test_longest_prefix_across_branches) {
    rax *rt = raxNew();
    ins(rt, "A", 1, 1);
    ins(rt, "AB", 2, 2);
    ins(rt, "ABC", 3, 3);
    ins(rt, "ABD", 3, 4);
    expect_prefix(rt, "ABDE", 4, 4, 3);   /* longest: ABD */
    expect_prefix(rt, "ABCX", 4, 3, 3);   /* longest: ABC */
    expect_prefix(rt, "AB", 2, 2, 2);
    expect_prefix(rt, "ABC", 3, 3, 3);
    expect_no_prefix(rt, "X", 1);
    raxFree(rt);
}

MU_TEST(test_prefix_split_compressed) {
    /* Inserting "ANNI" into "ANNIBALESCO" splits the compressed node:
     * the new key lands on the postfix node whose stored key is the
     * path to its start, while the leaf keeps the original full key. */
    rax *rt = raxNew();
    ins(rt, "ANNIBALESCO", 11, 1);
    ins(rt, "ANNI", 4, 2);
    expect_prefix(rt, "ANNI", 4, 2, 4);
    expect_prefix(rt, "ANNIBALESCO", 11, 1, 11);
    expect_prefix(rt, "ANNIBALE", 8, 2, 4);        /* longest: ANNI */
    expect_prefix(rt, "ANNIBALESCOX", 12, 1, 11);  /* longest: ANNIBALESCO */
    expect_no_prefix(rt, "ANN", 3);
    raxFree(rt);
}

MU_TEST(test_leaf_and_chain_keys) {
    /* X -> XY -> XYZW: keys land on a mid-chain node, a compressed
     * node's start, and a leaf. */
    rax *rt = raxNew();
    ins(rt, "X", 1, 7);
    ins(rt, "XY", 2, 8);
    ins(rt, "XYZW", 4, 9);
    expect_prefix(rt, "X", 1, 7, 1);
    expect_prefix(rt, "XY", 2, 8, 2);
    expect_prefix(rt, "XYZW", 4, 9, 4);
    expect_prefix(rt, "XYZWQ", 5, 9, 4);           /* longest: XYZW */
    expect_prefix(rt, "XZ", 2, 7, 1);              /* longest: X */
    expect_no_prefix(rt, "", 0);
    raxFree(rt);
}

MU_TEST(test_empty_tree) {
    rax *rt = raxNew();
    expect_no_prefix(rt, "ANY", 3);
    expect_no_prefix(rt, "", 0);
    raxFree(rt);
}

int main(void) {
    MU_RUN_TEST(test_exact_key);
    MU_RUN_TEST(test_longest_prefix_across_branches);
    MU_RUN_TEST(test_prefix_split_compressed);
    MU_RUN_TEST(test_leaf_and_chain_keys);
    MU_RUN_TEST(test_empty_tree);
    MU_REPORT();
    return MU_EXIT_CODE;
}
```

- [ ] **Step 2: 运行确认失败**

在 `test/` 下运行：`make test_rax`
Expected: 链接失败 `undefined reference to 'raxFindPrefix'`

- [ ] **Step 3: 实现 raxFindPrefix**

Add to `src/rax.h` after `raxFind`:

```c
void *raxFindPrefix(rax *rax, unsigned char *s, size_t len, size_t *matchlen);
```

Add to `src/rax.c` after `raxLowWalk`:

```c
/* Find the longest key that is a prefix of the string 's', returning its
 * associated data and setting *matchlen to the matched key length.
 * Returns raxNotFound when no key matches.
 *
 * Node key semantics (see the raxNode layout comment at the top of this
 * file): a non-compressed node's key is the path to the node (its own
 * char is a child edge, consumed when descending past it); a compressed
 * node's stored key is the path to the node's start, not including the
 * node's own string. The walk therefore checks iskey on every node at
 * arrival, before consuming anything. */
void *raxFindPrefix(rax *rt, unsigned char *s, size_t len, size_t *matchlen) {
    raxNode *h = rt->head;
    size_t i = 0;    /* query chars consumed (path length to h) */
    size_t klen = 0; /* key length of the best match */
    void *best = raxNotFound;

    while (1) {
        if (h->iskey) {
            best = raxGetData(h);
            klen = i;
        }
        if (i >= len || h->size == 0) break;
        if (h->iscompr) {
            unsigned char *v = h->data;
            size_t j;
            for (j = 0; j < h->size && i + j < len; j++) {
                if (v[j] != s[i + j]) break;
            }
            if (j == 0) break;          /* no common prefix */
            i += j;
            if (j != h->size) break;    /* partial match: stop here */
            memcpy(&h, raxNodeFirstChildPtr(h), sizeof(h));
        } else {
            unsigned char *v = h->data;
            size_t j;
            for (j = 0; j < h->size; j++) {
                if (v[j] == s[i]) break;
            }
            if (j == h->size) break;    /* no matching child */
            i++;
            memcpy(&h, raxNodeFirstChildPtr(h) + j, sizeof(h));
        }
    }
    if (matchlen) *matchlen = klen;
    return best;
}
```

- [ ] **Step 4: 运行确认通过**

Add to `test/Makefile`（`test_utils` 目标附近）:

```make
rax.o: $(SRC_DIR)/rax.c $(SRC_DIR)/rax.h $(SRC_DIR)/rax_malloc.h $(SRC_DIR)/mm.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/rax.c -o $@

test_rax: c/test_rax.c rax.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) -lm
```

在 `test/` 下运行：`make test_rax && ./test_rax`
Expected: 5 个测试全过，0 failures

- [ ] **Step 5: 接线 `check` 并提交**

Modify `test/Makefile`:

```make
check: test_pthreads test_utils test_qwen25 test_rax $(GPU_TESTS) $(METAL_TESTS)
	./test_pthreads
	./test_utils
	./test_qwen25
	./test_rax
```

运行：`make check`（或先 `make -C test check`），全部通过后：

```bash
git add src/rax.h src/rax.c test/c/test_rax.c test/Makefile
git commit -m "feat: add raxFindPrefix longest-prefix lookup with tests"
```

---

### Task 2: kvcache 骨架（结构体 / init / free / reset + Session 接线）

**Files:**
- Create: `src/kvcache.h`（覆盖空文件）
- Create: `src/kvcache.c`（覆盖空文件，本任务只实现 init/free/reset + 编码辅助）
- Modify: `src/def.h`（前向声明 + Session 字段）
- Modify: `src/core.c`（`session_create`/`session_free` 接线）
- Modify: `src/Makefile`（`kvcache.o` 规则）
- Modify: `test/Makefile`（`kvcache.o` 规则 + `test_kvcache` 目标 + `check` 接线）
- Create: `test/c/test_kvcache.c`

**Interfaces:**
- Consumes: `def.h` 的 `KvCache`/`AttnKvCache`；`mm.h`（smalloc/sfree/srealloc）；`slog.h`（WARN）
- Produces: `KvPrefixCache` 结构体与 `kvcache_init/free/reset`。Task 3 在同一个 `.c` 文件里补 `kvcache_prefix_match/insert/append`；Task 4 的 arch reset 调 `kvcache_reset(s->pcache)`。

- [ ] **Step 1: 写失败测试**

Create `test/c/test_kvcache.c`:

```c
#include <stdio.h>
#include "minunit.h"
#include "../../src/kvcache.h"
#include "../../src/mm.h"

/* Minimal fake per-layer cache: 2 layers, 1 kv head, head_dim 4, cap 16. */
#define T_NLAYER 2u
#define T_CAP    16u

static void make_cache(KvCache *kc) {
    kc->n_layer = T_NLAYER;
    kc->head_dim = 4;
    kc->n_kv_head = 1;
    kc->std = scalloc(T_NLAYER, sizeof(AttnKvCache));
    for (u32 i = 0; i < T_NLAYER; i++) {
        kc->std[i].k = scalloc((u64)T_CAP * 4, sizeof(float));
        kc->std[i].v = scalloc((u64)T_CAP * 4, sizeof(float));
        kc->std[i].cap = T_CAP;
    }
}

static void free_cache(KvCache *kc) {
    for (u32 i = 0; i < T_NLAYER; i++) {
        sfree(kc->std[i].k);
        sfree(kc->std[i].v);
    }
    sfree(kc->std);
}

MU_TEST(test_init_free) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    mu_check(kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true));
    mu_check(pc.tree != NULL);
    mu_check(pc.pos == 0);
    mu_check(pc.max_pos == 0);
    mu_check(raxSize(pc.tree) == 0);
    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_init_disabled) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    mu_check(kvcache_init(&pc, &kc, T_NLAYER, T_CAP, false));
    mu_check(pc.tree == NULL);
    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_reset_clears_path_keeps_tree) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    /* A path already in the tree (simulating a cached conversation). */
    unsigned char key[8] = {0, 0, 0, 1, 0, 0, 0, 2};
    mu_check(raxInsert(pc.tree, key, 8, (void *)(uintptr_t)1, NULL) == 1);
    mu_check(raxSize(pc.tree) == 1);

    kc.std[0].n = 7;              /* stale active length */
    kvcache_reset(&pc);
    mu_check(raxSize(pc.tree) == 1);   /* tree survives */
    mu_check(pc.pos == 0);
    mu_check(pc.curkey_len == 0);
    mu_check(kc.std[0].n == 0);        /* active length cleared */
    mu_check(kc.std[1].n == 0);

    kvcache_free(&pc);
    free_cache(&kc);
}

int main(void) {
    MU_RUN_TEST(test_init_free);
    MU_RUN_TEST(test_init_disabled);
    MU_RUN_TEST(test_reset_clears_path_keeps_tree);
    MU_REPORT();
    return MU_EXIT_CODE;
}
```

- [ ] **Step 2: 运行确认失败**

在 `test/` 下运行：`make test_kvcache`
Expected: 链接失败 `undefined reference to 'kvcache_init'`

- [ ] **Step 3: 实现骨架**

Create `src/kvcache.h`:

```c
#ifndef KVCACHE_H
#define KVCACHE_H

#include "def.h"
#include "rax.h"

/* KV prefix cache manager: a rax radix tree maps token-sequence prefixes
 * to cached KV segments in the contiguous per-layer buffers.
 *
 * Each tree node stores one token's worth of KV: its key is the token
 * sequence up to and including that token (each token encoded as 4 bytes
 * big-endian), and its value is the buffer position of that token's KV,
 * stored directly as a pointer cast ((void *)(uintptr_t)pos — zero
 * allocation, never heap-freed). */

typedef struct KvPrefixCache {
    rax     *tree;          /* token seq -> KV pos; NULL when disabled */
    u32      pos;           /* current active write position (= length) */
    u32      max_pos;       /* highest position ever written */
    bool     enabled;       /* false for MLA architectures */
    KvCache *kv;            /* per-layer buffers, owned by the arch */
    unsigned char *curkey;  /* current path key, grows on decode */
    size_t       curkey_len;
    size_t       curkey_cap;
} KvPrefixCache;

bool kvcache_init(KvPrefixCache *pc, KvCache *kv, u32 n_layer, u32 cap,
                  bool enabled);
void kvcache_free(KvPrefixCache *pc);
void kvcache_reset(KvPrefixCache *pc);

/* Prefill: match the longest cached prefix of tokens[0..n). Returns
 * matchlen (token count) and match_end (buffer position where the
 * unmatched suffix must be written; 0 when no prefix matched). Flushes
 * the cache when the suffix would overflow the buffers. The caller must
 * then compute KV for [matchlen, n) into the buffers at match_end and
 * call kvcache_prefix_insert with the same arguments. */
bool kvcache_prefix_match(KvPrefixCache *pc, const u32 *tokens, u32 n,
                          u32 *matchlen, u32 *match_end);
bool kvcache_prefix_insert(KvPrefixCache *pc, const u32 *tokens, u32 n,
                           u32 matchlen, u32 match_end);

/* Decode: extend the current path by one token whose KV was written at
 * pc->pos. Call after computing the token's KV into the buffers. */
bool kvcache_append(KvPrefixCache *pc, u32 token);

#endif
```

Create `src/kvcache.c`:

```c
#include "kvcache.h"
#include "mm.h"
#include "slog.h"

/* Append token as 4 big-endian bytes at buf[off..off+4). */
static void key_encode_append(u32 token, unsigned char *buf, size_t off) {
    buf[off + 0] = (unsigned char)(token >> 24);
    buf[off + 1] = (unsigned char)(token >> 16);
    buf[off + 2] = (unsigned char)(token >> 8);
    buf[off + 3] = (unsigned char)token;
}

/* Grow the curkey buffer so it can hold 'need' bytes. */
static bool key_grow(KvPrefixCache *pc, size_t need) {
    if (need <= pc->curkey_cap) return true;
    size_t cap = pc->curkey_cap ? pc->curkey_cap : 256;
    while (cap < need) cap *= 2;
    unsigned char *p = srealloc(pc->curkey, cap);
    if (!p) return false;
    pc->curkey = p;
    pc->curkey_cap = cap;
    return true;
}

bool kvcache_init(KvPrefixCache *pc, KvCache *kv, u32 n_layer, u32 cap,
                  bool enabled) {
    (void)n_layer; /* used by insert/append */
    (void)cap;     /* used by match/append flush checks */
    pc->tree       = enabled ? raxNew() : NULL;
    pc->pos        = 0;
    pc->max_pos    = 0;
    pc->enabled    = enabled;
    pc->kv         = kv;
    pc->curkey     = NULL;
    pc->curkey_len = 0;
    pc->curkey_cap = 0;
    return true;
}

void kvcache_free(KvPrefixCache *pc) {
    if (!pc) return;
    if (pc->tree) raxFree(pc->tree);
    pc->tree = NULL;
    sfree(pc->curkey);
    pc->curkey = NULL;
    pc->curkey_len = 0;
    pc->curkey_cap = 0;
}

void kvcache_reset(KvPrefixCache *pc) {
    if (!pc) return;
    /* Clear the active path only; the tree and buffers stay cached. */
    pc->curkey_len = 0;
    pc->pos = 0;
    if (pc->enabled && pc->kv && pc->kv->std) {
        for (u32 i = 0; i < pc->kv->n_layer; i++)
            pc->kv->std[i].n = 0;
    }
}
```

Modify `src/def.h` — add before `struct Session` (near line 293):

```c
typedef struct KvPrefixCache KvPrefixCache;   /* kvcache.h */
```

Modify `src/def.h` Session struct — replace:

```c
    /* ---- KV cache ---- */
    KvCache    cache;
```

with:

```c
    /* ---- KV cache ---- */
    KvCache    cache;
    KvPrefixCache *pcache;    /* rax prefix cache manager (kvcache.h) */
```

Modify `src/core.c`:
- 顶部 `#include "kvcache.h"`
- `session_create` 中，`if (!s->ops.init(s)) {...}` 之后、`slog(INFO, "Session created...")` 之前插入：

```c
    /* KV prefix cache manager over the arch-allocated buffers. */
    s->pcache = smalloc(sizeof(*s->pcache));
    if (!kvcache_init(s->pcache, &s->cache, s->cfg.n_layer, s->ctx_size,
                      s->en->model->arch != ARCH_DEEPSEEK)) {
        slog(WARN, "Session init failed — cleaning up.");
        sfree(s->pcache);
        sfree(s);
        return NULL;
    }
```

- `session_free` 改为：

```c
void session_free(Session *s) {
    if (!s) return;
    pthreads_destroy(s->pthreads);
    if (s->ops.free) s->ops.free(s);
    if (s->pcache) {
        kvcache_free(s->pcache);
        sfree(s->pcache);
        s->pcache = NULL;
    }
    sfree(s);
}
```

Modify `src/Makefile` — 在 `asctx.o` 规则后加：

```make
kvcache.o: kvcache.c kvcache.h rax.h def.h
	$(CC) $(CFLAGS) -c kvcache.c -o $@
```

Modify `test/Makefile` — 加：

```make
kvcache.o: $(SRC_DIR)/kvcache.c $(SRC_DIR)/kvcache.h $(SRC_DIR)/rax.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/kvcache.c -o $@

test_kvcache: c/test_kvcache.c kvcache.o rax.o mm.o asctx.o slog.o utils.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) -lm
```

- [ ] **Step 4: 运行确认通过**

在 `test/` 下运行：`make test_kvcache && ./test_kvcache`
Expected: 3 个测试全过，0 failures

在 `src/` 下运行：`make server`（或顶层 `make`）
Expected: 编译通过（kvcache.c 编译无警告）

- [ ] **Step 5: 接线 `check` 并提交**

Modify `test/Makefile`:

```make
check: test_pthreads test_utils test_qwen25 test_rax test_kvcache $(GPU_TESTS) $(METAL_TESTS)
	./test_pthreads
	./test_utils
	./test_qwen25
	./test_rax
	./test_kvcache
```

运行：`make check`，全部通过后：

```bash
git add src/kvcache.h src/kvcache.c src/def.h src/core.c src/Makefile test/Makefile test/c/test_kvcache.c
git commit -m "feat: add kvcache prefix-cache skeleton with session wiring"
```

---

### Task 3: kvcache 核心逻辑（match / insert / append / 驱逐 / flush）

**Files:**
- Modify: `src/kvcache.c`（补 `kvcache_prefix_match/insert/append` + 驱逐与 flush 内部函数）
- Modify: `test/c/test_kvcache.c`（追加正确性测试）

**Interfaces:**
- Consumes: Task 1 的 `raxFindPrefix`；Task 2 的 `KvPrefixCache`/`key_grow`/`key_encode_append`
- Produces: `kvcache_prefix_match/insert/append` 完整语义（见 Task 2 头注释 + 本节实现）。Task 4 只消费 `kvcache_reset`。

- [ ] **Step 1: 写失败测试（追加到 test_kvcache.c）**

在 `test_kvcache.c` 的 `main` 之前追加辅助与测试。**模拟后端**：把 token id 作为 marker 写进 layer-0 的 k 缓冲对应位置，测试同时断言 marker 位置与树结构：

```c
/* ---- Simulated backend ------------------------------------------- */
/* "Compute" KV for tokens [matchlen, n) at buffer position match_end. */
static void fake_compute(KvPrefixCache *pc, const u32 *tokens, u32 n,
                         u32 matchlen, u32 match_end) {
    u32 pos = match_end;
    for (u32 i = matchlen; i < n; i++, pos++)
        pc->kv->std[0].k[pos] = (float)tokens[i];
}

/* Simulated prefill: match -> compute suffix -> insert. */
static void sim_prefill(KvPrefixCache *pc, const u32 *tokens, u32 n) {
    u32 matchlen = 0, match_end = 0;
    mu_check(kvcache_prefix_match(pc, tokens, n, &matchlen, &match_end));
    fake_compute(pc, tokens, n, matchlen, match_end);
    mu_check(kvcache_prefix_insert(pc, tokens, n, matchlen, match_end));
}

/* Simulated decode step: compute at pos, then append. */
static void sim_generate(KvPrefixCache *pc, u32 token) {
    pc->kv->std[0].k[pc->pos] = (float)token;
    mu_check(kvcache_append(pc, token));
}

static void assert_key(KvPrefixCache *pc, const u32 *tokens, u32 n,
                       uintptr_t expect_pos) {
    unsigned char key[64];
    for (u32 i = 0; i < n; i++) {
        key[4 * i + 0] = (unsigned char)(tokens[i] >> 24);
        key[4 * i + 1] = (unsigned char)(tokens[i] >> 16);
        key[4 * i + 2] = (unsigned char)(tokens[i] >> 8);
        key[4 * i + 3] = (unsigned char)tokens[i];
    }
    void *d = raxFind(pc->tree, key, (size_t)n * 4);
    mu_check(d != raxNotFound);
    mu_assert((uintptr_t)d == expect_pos, "cached pos mismatch");
}

static void assert_no_key(KvPrefixCache *pc, const u32 *tokens, u32 n) {
    unsigned char key[64];
    for (u32 i = 0; i < n; i++) {
        key[4 * i + 0] = (unsigned char)(tokens[i] >> 24);
        key[4 * i + 1] = (unsigned char)(tokens[i] >> 16);
        key[4 * i + 2] = (unsigned char)(tokens[i] >> 8);
        key[4 * i + 3] = (unsigned char)tokens[i];
    }
    mu_check(raxFind(pc->tree, key, (size_t)n * 4) == raxNotFound);
}

/* ---- Tests -------------------------------------------------------- */

MU_TEST(test_prefill_extends_without_eviction) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    u32 abc[3] = {1, 2, 3};
    sim_prefill(&pc, abc, 3);
    assert_key(&pc, abc, 1, 0);
    assert_key(&pc, abc, 2, 1);
    assert_key(&pc, abc, 3, 2);
    mu_check(pc.pos == 3);
    mu_check(pc.max_pos == 3);
    mu_check(kc.std[0].n == 3);

    /* Extension: match_end == max_pos -> no eviction. */
    u32 abcd[4] = {1, 2, 3, 4};
    sim_prefill(&pc, abcd, 4);
    mu_check(raxSize(pc.tree) == 4);
    assert_key(&pc, abcd, 4, 3);
    mu_check(pc.pos == 4);

    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_divergence_evicts_suffix) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    u32 abc[3] = {1, 2, 3};
    sim_prefill(&pc, abc, 3);
    sim_generate(&pc, 9);   /* path 1,2,3,9 -> C at pos 2, X at pos 3 */

    /* Divergence: ABD. D is written at pos 2, overwriting C's KV; the
     * cached C(2) and X(3) must be evicted. */
    u32 abd[3] = {1, 2, 4};
    sim_prefill(&pc, abd, 3);
    assert_no_key(&pc, abc, 3);          /* C gone */
    assert_no_key(&pc, abc, 4);          /* X gone */
    assert_key(&pc, abd, 3, 2);          /* ABD at 2 */
    mu_assert(kc.std[0].k[2] == 4.0f, "D marker at pos 2");
    mu_check(pc.pos == 3);
    mu_check(raxSize(pc.tree) == 3);     /* A, AB, ABD */

    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_empty_suffix_then_generate) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    u32 abc[3] = {1, 2, 3};
    sim_prefill(&pc, abc, 3);
    mu_check(raxSize(pc.tree) == 3);

    /* Full-hit prefill with an empty suffix: AB matches, nothing to
     * compute, but pos stays at 2 < max_pos — the next generate writes
     * at 2 and MUST evict C first. */
    u32 ab[2] = {1, 2};
    sim_prefill(&pc, ab, 2);
    assert_no_key(&pc, abc, 3);          /* C evicted by insert */
    sim_generate(&pc, 7);                /* writes at pos 2 */
    assert_key(&pc, ab, 2, 1);
    mu_assert(kc.std[0].k[2] == 7.0f, "generate marker at pos 2");
    mu_check(pc.pos == 3);

    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_reuse_after_reset) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    u32 abc[3] = {1, 2, 3};
    sim_prefill(&pc, abc, 3);
    sim_generate(&pc, 9);
    sim_generate(&pc, 10);
    mu_check(pc.pos == 5);

    kvcache_reset(&pc);
    mu_check(raxSize(pc.tree) == 5);     /* tree survives */

    /* Same conversation again: full hit, zero recompute. */
    u32 full[5] = {1, 2, 3, 9, 10};
    sim_prefill(&pc, full, 5);
    mu_check(pc.pos == 5);
    mu_check(kc.std[0].k[4] == 10.0f);   /* untouched */
    sim_generate(&pc, 11);               /* continues at pos 5 */
    assert_key(&pc, full, 5, 4);
    mu_check(pc.pos == 6);

    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_flush_on_prefill_overflow) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    u32 a[16];
    for (u32 i = 0; i < 16; i++) a[i] = i + 1;
    sim_prefill(&pc, a, 16);             /* fills exactly to cap */
    mu_check(pc.pos == 16);
    mu_check(raxSize(pc.tree) == 16);

    /* The extension (16 cached + 1 new) would overflow cap 16, and the
     * flush condition is match_end + suffix > cap — with the contiguity
     * invariant match_end == matchlen, so this is n > cap. Flush clears
     * the tree, and the whole request is recomputed from position 0. */
    u32 b[17];
    for (u32 i = 0; i < 16; i++) b[i] = a[i];
    b[16] = 200;
    sim_prefill(&pc, b, 17);
    mu_check(raxSize(pc.tree) == 17);    /* old tree gone, new path in */
    mu_check(pc.max_pos == 17);
    mu_assert(kc.std[0].k[16] == 200.0f, "recomputed suffix at 16");
    mu_check(pc.pos == 17);

    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_flush_on_decode_overflow) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, true);

    u32 a[16];
    for (u32 i = 0; i < 16; i++) a[i] = i + 1;
    sim_prefill(&pc, a, 16);             /* exactly cap */
    mu_check(pc.pos == 16);
    mu_check(pc.max_pos == 16);

    sim_generate(&pc, 99);               /* pos 17 > cap -> flush */
    mu_check(pc.pos == 1);
    mu_check(pc.max_pos == 1);
    mu_check(raxSize(pc.tree) == 1);
    mu_assert(kc.std[0].k[0] == 99.0f, "flushed generate at 0");

    kvcache_free(&pc);
    free_cache(&kc);
}

MU_TEST(test_disabled_mode_pos_tracking) {
    KvCache kc;
    make_cache(&kc);
    KvPrefixCache pc;
    kvcache_init(&pc, &kc, T_NLAYER, T_CAP, false);

    u32 toks[3] = {1, 2, 3};
    u32 matchlen = 99, match_end = 99;
    mu_check(kvcache_prefix_match(&pc, toks, 3, &matchlen, &match_end));
    mu_check(matchlen == 0);
    mu_check(match_end == 0);
    mu_check(kvcache_prefix_insert(&pc, toks, 3, 0, 0));
    mu_check(pc.pos == 3);
    mu_check(kvcache_append(&pc, 4));
    mu_check(pc.pos == 4);
    mu_check(pc.tree == NULL);

    kvcache_free(&pc);
    free_cache(&kc);
}
```

更新 `main`：

```c
int main(void) {
    MU_RUN_TEST(test_init_free);
    MU_RUN_TEST(test_init_disabled);
    MU_RUN_TEST(test_reset_clears_path_keeps_tree);
    MU_RUN_TEST(test_prefill_extends_without_eviction);
    MU_RUN_TEST(test_divergence_evicts_suffix);
    MU_RUN_TEST(test_empty_suffix_then_generate);
    MU_RUN_TEST(test_reuse_after_reset);
    MU_RUN_TEST(test_flush_on_prefill_overflow);
    MU_RUN_TEST(test_flush_on_decode_overflow);
    MU_RUN_TEST(test_disabled_mode_pos_tracking);
    MU_REPORT();
    return MU_EXIT_CODE;
}
```

- [ ] **Step 2: 运行确认失败**

在 `test/` 下运行：`make test_kvcache && ./test_kvcache`
Expected: 链接失败 `undefined reference to 'kvcache_prefix_match'`（Task 2 未实现）

- [ ] **Step 3: 实现核心逻辑**

Append to `src/kvcache.c`（在 `kvcache_reset` 之后）：

```c
/* Drop every cached node and rewind the write cursor. The contiguous
 * buffers are left as dead space; the next prefill overwrites from 0. */
static void kvcache_flush(KvPrefixCache *pc) {
    if (pc->tree) {
        raxFree(pc->tree);
        pc->tree = raxNew();
    }
    pc->curkey_len = 0;
    pc->pos = 0;
    pc->max_pos = 0;
}

/* Remove every key node whose cached position is >= start. */
static void kvcache_evict_from(KvPrefixCache *pc, u32 start) {
    typedef struct { unsigned char *key; size_t len; } EvictKey;
    EvictKey *list = NULL;
    size_t count = 0, cap = 0;

    raxIterator it;
    raxStart(&it, pc->tree);
    if (raxSeek(&it, "^", NULL, 0)) {
        do {
            u32 pos = (u32)(uintptr_t)it.data;
            if (pos >= start) {
                if (count == cap) {
                    cap = cap ? cap * 2 : 16;
                    list = srealloc(list, cap * sizeof(EvictKey));
                }
                list[count].key = smalloc(it.key_len);
                memcpy(list[count].key, it.key, it.key_len);
                list[count].len = it.key_len;
                count++;
            }
        } while (raxNext(&it));
    }
    raxStop(&it);

    for (size_t i = 0; i < count; i++) {
        void *old = NULL;
        raxRemove(pc->tree, list[i].key, list[i].len, &old);
        sfree(list[i].key);
    }
    sfree(list);
}

bool kvcache_prefix_match(KvPrefixCache *pc, const u32 *tokens, u32 n,
                          u32 *matchlen, u32 *match_end) {
    *matchlen = 0;
    *match_end = 0;
    if (!pc->enabled || n == 0) return true;

    /* Encode the full query; it doubles as the new current path. */
    if (!key_grow(pc, (size_t)n * 4)) return false;
    for (u32 i = 0; i < n; i++)
        key_encode_append(tokens[i], pc->curkey, (size_t)i * 4);
    pc->curkey_len = (size_t)n * 4;

    size_t klen = 0;
    void *data = raxFindPrefix(pc->tree, pc->curkey, pc->curkey_len, &klen);
    if (data == raxNotFound) return true;

    u32 pos = (u32)(uintptr_t)data;
    *matchlen = (u32)(klen / 4);
    *match_end = pos + 1;
    if (*matchlen > 0)
        slog(INFO, "kvcache: prefix hit %u/%u tokens", *matchlen, n);

    /* Flush when the suffix would overflow the contiguous buffers. */
    if (*match_end + (n - *matchlen) > pc->kv->std[0].cap) {
        slog(WARN, "kvcache: ctx full (pos %u + %u > cap %u), flushing",
             *match_end, n - *matchlen, pc->kv->std[0].cap);
        kvcache_flush(pc);
        *matchlen = 0;
        *match_end = 0;
    }
    return true;
}

bool kvcache_prefix_insert(KvPrefixCache *pc, const u32 *tokens, u32 n,
                           u32 matchlen, u32 match_end) {
    if (!pc->enabled) {
        pc->pos = n;
        return true;
    }
    /* Rebuild the full query key (cheap; keeps insert self-contained). */
    if (!key_grow(pc, (size_t)n * 4)) return false;
    for (u32 i = 0; i < n; i++)
        key_encode_append(tokens[i], pc->curkey, (size_t)i * 4);
    pc->curkey_len = (size_t)n * 4;

    /* Eviction: the write cursor advances from match_end through the new
     * suffix AND the subsequent decode tokens, so any cached node at
     * pos >= match_end must go before we write over it. match_end ==
     * max_pos is the common extension path: zero eviction, O(1). */
    if (match_end < pc->max_pos)
        kvcache_evict_from(pc, match_end);

    u32 pos = match_end;
    for (u32 i = matchlen; i < n; i++, pos++) {
        if (raxInsert(pc->tree, pc->curkey, (size_t)(i + 1) * 4,
                      (void *)(uintptr_t)pos, NULL) != 1 && errno == ENOMEM)
            return false;
    }
    pc->pos = pos;
    if (pos > pc->max_pos) pc->max_pos = pos;
    if (pc->enabled && pc->kv && pc->kv->std) {
        for (u32 i = 0; i < pc->kv->n_layer; i++)
            pc->kv->std[i].n = pos;
    }
    return true;
}

bool kvcache_append(KvPrefixCache *pc, u32 token) {
    if (!pc->enabled) {
        pc->pos++;
        return true;
    }
    if (pc->pos + 1 > pc->kv->std[0].cap) {
        slog(WARN, "kvcache: ctx full at decode, flushing");
        kvcache_flush(pc);
    }
    if (!key_grow(pc, pc->curkey_len + 4)) return false;
    key_encode_append(token, pc->curkey, pc->curkey_len);
    pc->curkey_len += 4;

    u32 pos = pc->pos;
    if (raxInsert(pc->tree, pc->curkey, pc->curkey_len,
                  (void *)(uintptr_t)pos, NULL) != 1 && errno == ENOMEM)
        return false;
    pc->pos = pos + 1;
    if (pc->pos > pc->max_pos) pc->max_pos = pc->pos;
    if (pc->enabled && pc->kv && pc->kv->std) {
        for (u32 i = 0; i < pc->kv->n_layer; i++)
            pc->kv->std[i].n = pc->pos;
    }
    return true;
}
```

注意 `kvcache.c` 顶部已 `#include <errno.h>`？—— mm.h 或 def.h 会引入，若编译报 `errno` 未声明，在文件顶部加 `#include <errno.h>`。

- [ ] **Step 4: 运行确认通过**

在 `test/` 下运行：`make test_kvcache && ./test_kvcache`
Expected: 10 个测试全过，0 failures

在 `src/` 下运行：`make` 且 `DEBUG=1 make`（-Wall 下无警告）

- [ ] **Step 5: 提交**

```bash
git add src/kvcache.c test/c/test_kvcache.c
git commit -m "feat: implement kvcache match/insert/append with eviction and flush"
```

---

### Task 4: 后端 reset 接线与全量验证

**Files:**
- Modify: `src/model/llama.c`、`src/model/qwen25.c`、`src/model/qwen35.c`、`src/model/falcon.c`（reset 走 `kvcache_reset`；`src/model/deepseek.c` 不动）
- Modify: `test/Makefile`（无——上一任务已完成；本任务只验证）

**Interfaces:**
- Consumes: Task 2 的 `kvcache_reset(KvPrefixCache *)`（通过 `s->pcache`）
- Produces: 无新接口——后端 reset 语义统一为"清活动路径、保留缓存"

- [ ] **Step 1: 改 llama.c**

`src/model/llama.c`：顶部 include 区加 `#include "../kvcache.h"`（`#include "../mm.h"` 之后）；`llama_reset` 替换为：

```c
static void llama_reset(Session *s) {
    kvcache_reset(s->pcache);
    s->n_tokens = 0;
}
```

- [ ] **Step 2: 改 qwen25.c 与 falcon.c**

`src/model/qwen25.c` 与 `src/model/falcon.c`：同样的 include 与 reset 替换：

```c
static void qwen25_reset(Session *s) {
    kvcache_reset(s->pcache);
    s->n_tokens = 0;
}
```

（falcon 的 reset 函数名换成 `falcon_reset`。）

- [ ] **Step 3: 改 qwen35.c**

`src/model/qwen35.c`：include `../kvcache.h`；`qwen35_reset` 中把 `for (u32 i = 0; i < kc->n_layer; i++) kc->std[i].n = 0;` 两行替换为 `kvcache_reset(s->pcache);`，**保留** SSM/conv 状态清零与 `s->n_tokens = 0`：

```c
static void qwen35_reset(Session *s) {
    kvcache_reset(s->pcache);
    s->n_tokens = 0;

    /* Zero SSM recurrent state and conv state. */
    Qwen35Workspace *ws = (Qwen35Workspace *)s->arch_data;
    if (ws && ws->ssm_state) {
        ... 原样保留 ...
    }
}
```

- [ ] **Step 4: 全量构建与测试**

在仓库根目录运行：

```bash
make            # release 构建
make check      # 全部单元测试
DEBUG=1 make    # -Wall 警告检查（有警告则修复）
DEBUG=1 make check
```

Expected: 全部通过、零警告。`make clean` 后重复一次 release 构建确认干净。

> **集成验证推迟**：spec 中"server 同一对话连发两请求、第二请求 prefill 时间 ≈ 0"的验证依赖推理实现（各后端 `prefill/generate` 仍为 stub）。本计划交付后，验收方式改为：`test_kvcache` 全部通过 + `make check` 全绿。端到端收益验证随推理实现计划一并做。

- [ ] **Step 5: 提交**

```bash
git add src/model/llama.c src/model/qwen25.c src/model/qwen35.c src/model/falcon.c
git commit -m "refactor: route arch resets through kvcache_reset to keep prefix cache"
```
