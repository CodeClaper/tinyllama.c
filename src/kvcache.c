#include <string.h>
#include "kvcache.h"
#include "graph.h"
#include "mm.h"
#include "slog.h"

#define KV_SLOT_NONE UINT32_MAX
#define KV_SNAP_BUDGET (256ULL * 1024 * 1024)
#define KV_SNAP_MAX 16

static void snapshot_clear(KvRadixCache *c, u32 idx) {
    if (!c || !c->snaps || idx >= c->cap_snaps) return;
    KvStateSnapshot *sn = &c->snaps[idx];
    if (sn->valid && sn->slot < c->pool.n_slots &&
        c->meta[sn->slot].snap == idx)
        c->meta[sn->slot].snap = KV_SLOT_NONE;
    sn->valid = false;
    sn->slot = KV_SLOT_NONE;
    sn->pos = 0;
}

static u32 snapshot_select(KvRadixCache *c) {
    if (!c->snaps || c->cap_snaps == 0) return KV_SLOT_NONE;
    for (u32 i = 0; i < c->cap_snaps; i++)
        if (!c->snaps[i].valid) return i;

    u32 victim = KV_SLOT_NONE;
    u64 oldest = UINT64_MAX;
    for (u32 i = 0; i < c->cap_snaps; i++) {
        if (c->snaps[i].valid && c->snaps[i].last_used < oldest) {
            oldest = c->snaps[i].last_used;
            victim = i;
        }
    }
    if (victim != KV_SLOT_NONE) snapshot_clear(c, victim);
    return victim;
}

static void *key_find(KvRadixCache *c, size_t len) {
    void *v = raxFind(c->tree, c->keybuf, len);
    return v == raxNotFound ? NULL : v;
}

static void request_release(KvRadixCache *c, u32 keep) {
    if (keep > c->req.len) keep = c->req.len;
    for (u32 i = keep; i < c->req.len; i++) {
        u32 slot = c->req.slots[i];
        if (slot < c->pool.n_slots && c->meta[slot].ref_count > 0)
            c->meta[slot].ref_count--;
        c->req.slots[i] = KV_SLOT_NONE;
    }
    c->req.len = keep;
    if (keep == 0) {
        c->req.shared_len = 0;
        c->req.active = false;
    }
}

static void slot_free(KvRadixCache *c, u32 slot) {
    if (c->meta[slot].snap != KV_SLOT_NONE)
        snapshot_clear(c, c->meta[slot].snap);
    c->meta[slot].snap = KV_SLOT_NONE;
    c->meta[slot].in_tree = false;
    c->meta[slot].ref_count = 0;
    c->meta[slot].child_count = 0;
    c->meta[slot].parent = KV_SLOT_NONE;
    c->pool.free_stack[c->pool.n_free++] = slot;
}

static bool slot_evict_one(KvRadixCache *c) {
    u32 victim = KV_SLOT_NONE;
    u64 oldest = UINT64_MAX;
    for (u32 i = 0; i < c->pool.n_slots; i++) {
        KvSlotMeta *m = &c->meta[i];
        if (!m->in_tree || m->child_count != 0 || m->ref_count != 0)
            continue;
        if (m->last_used < oldest) {
            oldest = m->last_used;
            victim = i;
        }
    }
    if (victim == KV_SLOT_NONE) return false;

    u32 depth = 0;
    u32 *path = c->path;
    for (u32 cur = victim; cur != KV_SLOT_NONE; cur = c->meta[cur].parent) {
        if (depth >= c->ctx_size) return false;
        path[depth++] = c->meta[cur].token;
    }
    for (u32 i = 0; i < depth / 2; i++) {
        u32 tmp = path[i];
        path[i] = path[depth - 1 - i];
        path[depth - 1 - i] = tmp;
    }
    (void)raxRemove(c->tree, (unsigned char *)path,
                    (size_t)depth * sizeof(u32), NULL);

    u32 parent = c->meta[victim].parent;
    if (parent != KV_SLOT_NONE && c->meta[parent].child_count > 0)
        c->meta[parent].child_count--;
    slot_free(c, victim);
    c->stats.evictions++;
    return true;
}

static u32 slot_alloc(KvRadixCache *c, u32 token, u32 parent) {
    while (c->pool.n_free == 0) {
        if (!slot_evict_one(c)) return KV_SLOT_NONE;
    }
    u32 slot = c->pool.free_stack[--c->pool.n_free];
    KvSlotMeta *m = &c->meta[slot];
    m->token        = token;
    m->parent       = parent;
    m->child_count  = 0;
    m->ref_count    = 1;
    m->last_used    = ++c->clock;
    m->in_tree      = false;
    c->stats.allocations++;
    return slot;
}

KvRadixCache *kv_cache_create(Session *s, KvCache *kv, u32 ctx_size) {
    if (!kv || ctx_size == 0) return NULL;

    KvRadixCache *c = scalloc(1, sizeof(*c));
    c->tree = raxNew();
    c->kv = kv;
    c->s = s;
    c->n_layer = kv->n_layer;
    c->ctx_size = ctx_size;
    c->enabled = true;

    c->pool.n_slots = ctx_size;
    c->pool.free_stack = smalloc((size_t)ctx_size * sizeof(u32));
    for (u32 i = 0; i < ctx_size; i++) {
        u32 slot = ctx_size - 1 - i;
        c->pool.free_stack[i] = slot;
    }
    c->pool.n_free = ctx_size;

    c->meta = scalloc(ctx_size, sizeof(KvSlotMeta));
    for (u32 i = 0; i < ctx_size; i++) {
        c->meta[i].parent = KV_SLOT_NONE;
        c->meta[i].snap = KV_SLOT_NONE;
    }

    c->req.slots = smalloc((size_t)ctx_size * sizeof(u32));
    for (u32 i = 0; i < ctx_size; i++)
        c->req.slots[i] = KV_SLOT_NONE;

    c->keybuf_cap = (size_t)ctx_size * sizeof(u32);
    c->keybuf = smalloc(c->keybuf_cap);
    c->path = smalloc((size_t)ctx_size * sizeof(u32));

    if (s) {
        c->state_bytes = graph_state_bytes(s);
        if (c->state_bytes) {
            u64 max = KV_SNAP_BUDGET / (u64)c->state_bytes;
            if (max < 1) max = 1;
            if (max > KV_SNAP_MAX) max = KV_SNAP_MAX;
            c->cap_snaps = (u32)max;
            c->snaps = scalloc(c->cap_snaps, sizeof(*c->snaps));
            for (u32 i = 0; i < c->cap_snaps; i++)
                c->snaps[i].slot = KV_SLOT_NONE;
        }
    }

    if (!c->tree || !c->pool.free_stack || !c->meta ||
        !c->req.slots || !c->keybuf || !c->path) {
        kv_cache_destroy(c);
        return NULL;
    }
    return c;
}

void kv_cache_destroy(KvRadixCache *c) {
    if (!c) return;
    if (c->tree) raxFree(c->tree);
    sfree(c->pool.free_stack);
    sfree(c->meta);
    sfree(c->req.slots);
    sfree(c->keybuf);
    sfree(c->path);
    if (c->snaps) {
        for (u32 i = 0; i < c->cap_snaps; i++)
            sfree(c->snaps[i].state);
        sfree(c->snaps);
    }
    sfree(c);
}

u32 kv_cache_match(KvRadixCache *c, const u32 *tokens, u32 n) {
    if (!c || !c->enabled || !tokens || n == 0) return 0;
    if ((size_t)n > c->keybuf_cap / sizeof(u32))
        n = (u32)(c->keybuf_cap / sizeof(u32));

    request_release(c, 0);
    memcpy(c->keybuf, tokens, (size_t)n * sizeof(u32));

    u32 matched = 0;
    for (u32 i = 0; i < n; i++) {
        void *v = key_find(c, (size_t)(i + 1) * sizeof(u32));
        if (!v) break;
        u32 slot = (u32)((uintptr_t)v - 1);
        if (slot >= c->pool.n_slots || !c->meta[slot].in_tree) break;
        c->req.slots[i] = slot;
        c->meta[slot].ref_count++;
        c->meta[slot].last_used = ++c->clock;
        matched = i + 1;
    }

    c->req.len = matched;
    c->req.shared_len = matched;
    c->req.active = matched > 0;
    c->stats.hit_tokens += matched;
    if (n > matched) c->stats.computed_tokens += n - matched;
    return matched;
}

bool kv_cache_append(KvRadixCache *c, const u32 *tokens, u32 pos, u32 n) {
    if (!c || !c->enabled || n == 0) return true;
    if ((u64)pos + n > c->ctx_size) return false;
    if ((size_t)(pos + n) > c->keybuf_cap / sizeof(u32)) return false;

    memcpy(c->keybuf, tokens, (size_t)(pos + n) * sizeof(u32));

    for (u32 k = 0; k < n; k++) {
        u32 i = pos + k;
        if (i < c->req.len) {
            u32 old = c->req.slots[i];
            if (old < c->pool.n_slots && c->meta[old].in_tree &&
                c->meta[old].token == tokens[i]) {
                c->meta[old].last_used = ++c->clock;
                continue;
            }
            request_release(c, i);
        }
        if (i != c->req.len) return false;

        size_t key_len = (size_t)(i + 1) * sizeof(u32);
        void *v = key_find(c, key_len);
        if (v) {
            u32 slot = (u32)((uintptr_t)v - 1);
            if (slot >= c->pool.n_slots || !c->meta[slot].in_tree) return false;
            c->req.slots[i] = slot;
            c->meta[slot].ref_count++;
            c->meta[slot].last_used = ++c->clock;
            c->req.len = i + 1;
            continue;
        }

        u32 parent = i > 0 ? c->req.slots[i - 1] : KV_SLOT_NONE;
        if (i > 0 && parent == KV_SLOT_NONE) return false;
        u32 slot = slot_alloc(c, tokens[i], parent);
        if (slot == KV_SLOT_NONE) return false;

        c->req.slots[i] = slot;
        if (parent != KV_SLOT_NONE) c->meta[parent].child_count++;
        if (!raxInsert(c->tree, c->keybuf, key_len,
                       (void *)(uintptr_t)(slot + 1), NULL)) {
            if (parent != KV_SLOT_NONE && c->meta[parent].child_count > 0)
                c->meta[parent].child_count--;
            slot_free(c, slot);
            return false;
        }
        c->meta[slot].in_tree = true;
        c->req.len = i + 1;
    }
    return true;
}

bool kv_cache_checkpoint(KvRadixCache *c, u32 pos) {
    if (!c || !c->s || !c->state_bytes || !c->snaps ||
        pos == 0 || pos > c->req.len)
        return true;

    u32 slot = c->req.slots[pos - 1];
    if (slot >= c->pool.n_slots || !c->meta[slot].in_tree) return true;

    u32 idx = c->meta[slot].snap;
    if (idx < c->cap_snaps && c->snaps[idx].valid) {
        c->snaps[idx].pos = pos;
        c->snaps[idx].last_used = ++c->clock;
        return true;
    }

    idx = snapshot_select(c);
    if (idx == KV_SLOT_NONE) return true;
    KvStateSnapshot *sn = &c->snaps[idx];
    if (!sn->state) sn->state = smalloc(c->state_bytes);
    if (!graph_state_save(c->s, sn->state, c->state_bytes))
        return false;

    sn->valid = true;
    sn->slot = slot;
    sn->pos = pos;
    sn->last_used = ++c->clock;
    c->meta[slot].snap = idx;
    return true;
}

u32 kv_cache_restore(KvRadixCache *c, u32 matched, u32 n, bool *restored) {
    if (restored) *restored = false;
    if (!c || !c->s || !c->state_bytes || !c->snaps || matched == 0)
        return 0;

    for (u32 pos = matched; pos > 0; pos--) {
        if (pos >= n) continue;
        u32 slot = c->req.slots[pos - 1];
        if (slot >= c->pool.n_slots || !c->meta[slot].in_tree) continue;
        u32 idx = c->meta[slot].snap;
        if (idx >= c->cap_snaps || !c->snaps[idx].valid) continue;
        if (c->snaps[idx].pos != pos || !c->snaps[idx].state) continue;
        if (!graph_state_load(c->s, c->snaps[idx].state, c->state_bytes))
            return 0;
        c->snaps[idx].last_used = ++c->clock;
        if (restored) *restored = true;
        return pos;
    }
    return 0;
}
