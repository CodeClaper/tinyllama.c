#ifndef __KVCACHE_H__
#define __KVCACHE_H__

#include "def.h"
#include "rax.h"

/* Per-slot metadata for the KV cache radix tree. */
typedef struct {
    u32 token;        /* token id stored in this slot */
    u32 parent;       /* slot of the previous token; U32_MAX for root */
    u32 child_count;  /* number of direct children, used to detect leaves */
    u32 ref_count;    /* number of requests currently using this slot */
    u64 last_used;    /* LRU timestamp */
    u32 snap;         /* state snapshot index, or UINT32_MAX */
    bool in_tree;     /* whether the slot is attached to the rax tree */
} KvSlotMeta;

/* Pool of free physical KV slots, managed as a stack for O(1) alloc/free. */
typedef struct {
    u32  n_slots;      /* total number of slots in the pool */
    u32  n_free;       /* number of currently free slots */
    u32 *free_stack;   /* [n_slots], stack of free slot indices */
} KvSlotPool;

/* Per-request view of the KV cache: logical positions mapped to physical slots. */
typedef struct {
    u32 *slots;        /* [ctx_size], logical position -> physical slot */
    u32  len;          /* current sequence length */
    u32  shared_len;   /* length of the matched and locked shared prefix */
    bool active;       /* whether this request currently holds the cache */
} KvRequestState;

/* KV cache statistics accumulated over the engine lifetime. */
typedef struct {
    u64 hit_tokens;      /* tokens served from the shared prefix cache */
    u64 computed_tokens; /* tokens that had to be computed from scratch */
    u64 evictions;       /* slots evicted to make room */
    u64 allocations;     /* successful slot allocations */
} KvRequstStats;

typedef struct {
    void *state;      /* packed recurrent/compressed state */
    u32   slot;       /* slot whose prefix this snapshot follows */
    u32   pos;        /* logical length of the snapshot */
    u64   last_used;
    bool  valid;
} KvStateSnapshot;

/* Radix-tree based prefix cache for KV slots. */
typedef struct KvRadixCache {
    rax            *tree;        /* token prefix -> slot + 1 */
    KvSlotPool      pool;
    KvSlotMeta     *meta;        /* [n_slots] */
    KvRequestState  req;
    KvCache        *kv;          /* points to GraphPlan.cache */
    u32             n_layer;
    u32             ctx_size;
    bool            enabled;
    unsigned char  *keybuf;      /* scratch buffer for key encoding */
    size_t          keybuf_cap;
    u32            *path;        /* [ctx_size] eviction path scratch */
    Session        *s;           /* owning session (state callbacks) */
    size_t          state_bytes; /* packed snapshot size, 0 = none */
    KvStateSnapshot *snaps;
    u32             n_snaps;
    u32             cap_snaps;
    u64             clock;       /* monotonically increasing LRU counter */
    /* stats */
    KvRequstStats stats;
} KvRadixCache;

KvRadixCache *kv_cache_create(Session *s, KvCache *kv, u32 ctx_size);
void kv_cache_destroy(KvRadixCache *c);

/* Release the previous request's references, then lock the longest
 * cached prefix of `tokens` into the request slot map.  Returns the
 * number of matched tokens. */
u32 kv_cache_match(KvRadixCache *c, const u32 *tokens, u32 n);

/* Map/allocate physical slots for positions [pos, pos+n).  `tokens`
 * is the full sequence starting at position 0.  Existing entries are
 * validated; a divergent token truncates the request path first. */
bool kv_cache_append(KvRadixCache *c, const u32 *tokens, u32 pos, u32 n);

/* Save/restore recurrent state snapshots for checkpointed prefixes. */
bool kv_cache_checkpoint(KvRadixCache *c, u32 pos);

/* Pick and load the deepest usable checkpoint at or below `matched`.
 * Returns the resume position; `restored` tells whether state was loaded. */
u32 kv_cache_restore(KvRadixCache *c, u32 matched, u32 n, bool *restored);

#endif
