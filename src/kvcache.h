#include "def.h"
#include "rax.h"

/* Per-slot metadata for the KV cache radix tree. */
typedef struct {
    u32 token;        /* token id stored in this slot */
    u32 parent;       /* slot of the previous token; U32_MAX for root */
    u32 child_count;  /* number of direct children, used to detect leaves */
    u32 ref_count;    /* number of requests currently using this slot */
    u64 last_used;    /* LRU timestamp */
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
    u64             clock;       /* monotonically increasing LRU counter */
    /* stats */
    KvRequstStats stats;
} KvRadixCache;
