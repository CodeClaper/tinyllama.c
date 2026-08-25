#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "asctx.h"
#include "def.h"

const uint8_t leftmost_one_pos[256] = {
	0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};

/* Leftmost 32bit word postion. */
static int leftmost_32_pos(uint32_t word) {
    int shift = 32 - 8;
    while ((word >> shift) == 0)
        shift -=8;
    return shift + leftmost_one_pos[(word >> shift) & 255];
}

static void fatal(const char *msg) {
    fprintf(stderr, "Fatal for: %s\n", msg);
    exit(1);
}

/* Index of AllocSet free size 
 * --------------------------
 * At this pointer we must compute ceil(log2(size >> ALLOC_MINBITS))
 * This is the same as
 *      leftmost_32_pos((size - 1) >> ALLOC_MINBITS) + 1
 * or equivalently
 *      leftmost_32_pos(size - 1) - ALLOC_MINBITS + 1
 * -------------------------
 * */
static inline int alloc_set_free_index(size_t size) {
    int idx = size > (1 << ALLOC_MINBITS) 
            ? leftmost_32_pos(size -1) - ALLOC_MINBITS + 1
            : 0;
    return idx;
}

static inline void alloc_chunk_set_mask_external(AllocChunk chunk) {
    chunk->mask = ALLOC_CHUNK_MAGIC | (((uint64_t) 1) << ALLOC_CHUNK_EXTERNAL_BASEBIT);
}

static inline void alloc_chunk_set_mask(AllocChunk chunk, AllocBlock block, size_t value) {
    size_t offset = (char *) chunk - (char *) block;
    chunk->mask = (((uint64_t) offset) << ALLOC_CHUNK_BLOCK_OFFSET_BASEBIT) | ((uint64_t) value) << ALLOC_CHUNK_VALUE_BASEBIT;
}

static inline bool alloc_chunk_is_external(AllocChunk chunk) {
    return CHUNK_IS_EXTERNAL(chunk->mask);
}

static inline AllocBlock alloc_chunk_get_block(AllocChunk chunk) {
    return (AllocBlock) ((char *) chunk - CHUNK_GET_OFFSET(chunk->mask));
}

AllocSetContext *alloc_set_memory_context_create(u32 block_size) {
    AllocSetContext *set;
    size_t size;
    AllocBlock block;

    /* Allocate the context header and the first (keeper) block
     * together; block_size is the payload capacity of the keeper. */
    size = MAXALIGN(ALLOC_SET_CXT_SIZE + block_size);

    set = (AllocSet) malloc(size);
    if (!set) fatal("Out of memory");

    block          = KEEPER_ALLOC_BLOCK(set);
    block->set     = set;
    block->pres    = NULL;
    block->next    = NULL;
    block->freeptr = ((char *) block) + ALLOC_BLOCK_SIZE;
    block->endptr  = ((char *) set) + size;

    set->blocks = block;
    set->next_block_size = MAXALIGN(block_size);
    memset(set->free_list, 0, sizeof(set->free_list));

    return set;
}

void *alloc_set_alloc_chunk_from_block(AllocSetContext *context, AllocBlock block, size_t chksize) {
    AllocChunk chunk = (AllocChunk) block->freeptr;
    block->freeptr += (chksize + ALLOC_CHUNK_SIZE);
    chunk->next = NULL;
    alloc_chunk_set_mask(chunk, block, chksize);
    return CHUNK_GET_POINTER(chunk);
}

static void *alloc_set_alloc_new_block(AllocSetContext *context, size_t chksize) {
    size_t blk_size;
    AllocSet set = (AllocSet) context;

    blk_size = set->next_block_size;

    /* Grow the block size for the next block, up to the cap. */
    set->next_block_size = blk_size > DEFAULT_MAX_BLOCK_SIZE / 2
                           ? DEFAULT_MAX_BLOCK_SIZE : blk_size * 2;

    /* The block must be able to hold the requested chunk. */
    if (blk_size < chksize + ALLOC_BLOCK_SIZE + ALLOC_CHUNK_SIZE)
        blk_size = chksize + ALLOC_BLOCK_SIZE + ALLOC_CHUNK_SIZE;

    AllocBlock block = (AllocBlock) malloc(blk_size);
    if (!block) fatal("Out of memory");

    block->freeptr = ((char *) block) + ALLOC_BLOCK_SIZE;
    block->endptr  = ((char *) block) + blk_size;
    block->set     = set;
    block->next    = set->blocks;
    block->pres    = NULL;
    
    if (block->next) block->next->pres = block;
    set->blocks = block;

    return alloc_set_alloc_chunk_from_block(context, block, chksize);
}

static void *alloc_set_alloc_large(AllocSetContext *context, size_t size) {
    AllocSet set = (AllocSet) context;
    size_t blk_size;
    AllocBlock block;
    AllocChunk chunk;
    
    blk_size = MAXALIGN(size + ALLOC_BLOCK_SIZE + ALLOC_CHUNK_SIZE);
    block = (AllocBlock) malloc(blk_size);
    if (!block) fatal("Out of memory");

    block->set = set;
    block->freeptr = block->endptr = ((char *) block) + blk_size;
    chunk = (AllocChunk) (((char *) block) + ALLOC_BLOCK_SIZE);
    /* Mark the Chunk as externally managed. */
    alloc_chunk_set_mask_external(chunk);

    if (set->blocks != NULL) {
        block->pres = set->blocks;
        block->next = set->blocks->next;
        if (block->next != NULL) block->next->pres = block;
        set->blocks->next = block;
    } else {
        block->pres = NULL;
        block->next = NULL;
        set->blocks = block;
    }

    return CHUNK_GET_POINTER(chunk);
}

void *alloc_set_alloc(AllocSetContext *context, size_t size) {
    size_t     chksize, freesize;
    int        fdx;
    AllocChunk chunk;

    if (size > ALLOC_CHUNK_LIMIT) return alloc_set_alloc_large(context, size);

    AllocSet   set   = (AllocSet) context;
    AllocBlock block = set->blocks;

    /* Find there is enough chunk in FreeList. */
    fdx = alloc_set_free_index(size);
    chunk = set->free_list[fdx];
    if (chunk) {
        set->free_list[fdx] = chunk->next;
        return CHUNK_GET_POINTER(chunk);
    }

    chksize = CHUNK_GET_SIZE_FROM_FREE_LIST_IDX(fdx);

    /* Get block free space size. 
     * If it has enough free space, allocate from current block. 
     * Otherwise, generate new block and allocate from that. */
    freesize = block->endptr - block->freeptr;

    if (freesize < chksize + ALLOC_CHUNK_SIZE) return alloc_set_alloc_new_block(context, chksize);
    else return alloc_set_alloc_chunk_from_block(context, block, chksize);
}

void *alloc_set_realloc(void *ptr, size_t size) {
    AllocChunk chunk = POINTER_GET_CHUNK(ptr);
    AllocBlock block;
    AllocSet   set;
    size_t     old_size;

    /* Way to external chunk. */
    if (alloc_chunk_is_external(chunk)) {
        size_t blksize;

        block = CHUNK_EXTERNAL_GET_BLOCK(chunk);
        set = block->set;
        blksize = MAXALIGN(size + ALLOC_BLOCK_SIZE + ALLOC_CHUNK_SIZE);

        block = realloc(block, blksize);
        if (!block) fatal("Out of memory");

        block->freeptr = block->endptr = ((char *) block) + blksize;
        chunk = (AllocChunkData *) (((char *) block) + ALLOC_BLOCK_SIZE);

        if (block->pres) block->pres->next = block;
        else set->blocks = block;
        if (block->next) block->next->pres = block;

        return CHUNK_GET_POINTER(chunk);
    }

    /* Way to normal chunk. */
    block = alloc_chunk_get_block(chunk);
    set = block->set;
    old_size = CHUNK_GET_VALUE(chunk->mask);

    if (old_size >= size) return ptr; 
    else {
        void *nptr = alloc_set_alloc((AllocSetContext *) set, size);
        memcpy(nptr, ptr, old_size);
        alloc_set_free(ptr);
        return nptr;
    }
}

void alloc_set_free(void *ptr) {
    AllocSet   set;
    AllocChunk chunk = POINTER_GET_CHUNK(ptr);

    if (alloc_chunk_is_external(chunk)) {
        AllocBlock block = CHUNK_EXTERNAL_GET_BLOCK(chunk);
        set = block->set;
        if (block->pres) block->pres->next = block->next;
        else set->blocks = block->next;
        if (block->next) block->next->pres = block->pres;
        free(block);
    } else {
        AllocBlock block = alloc_chunk_get_block(chunk); 
        set = block->set;
        int fdx = alloc_set_free_index(CHUNK_GET_VALUE(chunk->mask));
        chunk->next = set->free_list[fdx];
        set->free_list[fdx] = chunk;
    }
}

void alloc_set_reset(AllocSetContext *context) {
    AllocSet   set   = (AllocSet) context;
    AllocBlock block = set->blocks;

    /* Clean the free list*/
    memset(set->free_list, 0, sizeof(set->free_list));

    /* Reset first block as header. */
    set->blocks = KEEPER_ALLOC_BLOCK(set);

    while (block != NULL) {
        AllocBlock next = block->next;
        if (IS_KEEPER_BLOCK(set, block)) {
            /* Way to handle first block. */
            char *start = ((char *) block) + ALLOC_BLOCK_SIZE;
            block->freeptr = start;
            block->next = NULL;
            block->pres = NULL;
        } else {
            /* Way to handle other block. */
            free(block);
        }
        block = next;
    }
}

void alloc_set_memory_context_delete(AllocSetContext *context) {
    alloc_set_reset(context);
    free((AllocSet) context);
}
