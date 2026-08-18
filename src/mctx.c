#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mctx.h"
#include "asctx.h"
#include "utils.h"

/* 
 * Current MemoryContext. 
 */
MemoryContext CURRENT_MEMORY_CONTEXT = NULL;

/* 
 * Store start-up and initialization Info
 */
MemoryContext TOP_MEMORY_CONTEXT = NULL;

/* 
 * Store start-up load object info.
 */
MemoryContext SIDE_MEMORY_CONTEXT = NULL;


static MemoryContextMethods mctx_methods[] = {
    /* AllocSetContext implements. */
    [ALLOC_SET_ID].alloc = AllocSetAlloc,
    [ALLOC_SET_ID].free = AllocSetFree,
    [ALLOC_SET_ID].realloc = AllocSetRealloc,
    [ALLOC_SET_ID].reset = AllocSetReset,
    [ALLOC_SET_ID].delete_context = AllocSetDelete
};

/* MemoryContextInit.
 * Start up the memory-context subsystem. */
void MemoryContextInit(void) {
    TOP_MEMORY_CONTEXT = AllocSetMemoryContextCreate(NULL, "TopMemoryContext", DEFAULT_MAX_BLOCK_SIZE);
    SIDE_MEMORY_CONTEXT = AllocSetMemoryContextCreate(TOP_MEMORY_CONTEXT, "SideMemoryContext", DEFAULT_MAX_BLOCK_SIZE);
    MemoryContextSwitchTo(TOP_MEMORY_CONTEXT);
}


/* Create MemoryContext.
 * Thist abstract function not really to create MemoryContext and it just
 * make up base info and link to others MemoryContext.
 * */
void MemoryContextCreate(MemoryContext node, MemoryContext parent, 
                         const char *name, ContextType type, 
                         MemoryContextMethodID id) {
    /* Make up base Info. */
    node->name = name;
    node->type = type;
    node->parent = parent;
    node->allocated_size = 0;
    node->firstChild = NULL;
    node->presChild = NULL;
    node->nextChild = NULL;
    node->context_methods = &mctx_methods[id];
    
    /* Link node to peer nodes. */
    if (parent != NULL) {
        if (parent->firstChild != NULL) 
            parent->firstChild->presChild = node;
        parent->firstChild = node;
    }
}

/* MemoryContextReset. 
 * Release all space allocate within a context and also its children contexts. */
void MemoryContextReset(MemoryContext context) {
    context->context_methods->reset(context);
}

/* MemoryContext set its parent. */
void MemoryContextSetParent(MemoryContext context, MemoryContext new_parent) {
    Assert(context);
    Assert(context != new_parent);

    if (context->parent == new_parent)
        return;

    /* Delink. */
    if (context->parent) {
        MemoryContext parent = context->parent;
        
        if (context->presChild) 
            context->presChild->nextChild = context->nextChild;
        else 
        {
            Assert(parent->firstChild == context);
            parent->firstChild = context->nextChild;
        }

        if (context->nextChild) 
            context->nextChild->presChild = context->presChild;
    }

    if (new_parent) {
        context->parent = new_parent;
        context->presChild = NULL;
        context->nextChild = new_parent->firstChild;
        if (new_parent->firstChild) 
            new_parent->firstChild->presChild = context;
        new_parent->firstChild = context;
    } else {
        context->parent = NULL;
        context->presChild = NULL;
        context->nextChild = NULL;
    }
}

/* Delete the MemoryContext only. */
static void MemoryContextDeleteOnly(MemoryContext context) {
    /* Delink the parent. */
    MemoryContextSetParent(context, NULL);
    context->context_methods->delete_context(context);
}

/* Delete the MemoryContext. */
void MemoryContextDelete(MemoryContext context) {
    Assert(context);
    MemoryContext curcontext;
    curcontext = context;

    for(;;) {
        MemoryContext parentcontext;
        while (curcontext->firstChild != NULL)  {
            curcontext = curcontext->firstChild;
        }
        parentcontext = curcontext->parent;
        MemoryContextDeleteOnly(curcontext);
        if (context == curcontext) break;
        curcontext = parentcontext;
    }
}

/* Switch to MemoryContext. */
void *MemoryContextSwitchTo(MemoryContext currentConext) {
    /* Not allowd null. */
    Assert(currentConext != NULL);
    /* Not allowed parallel compute there. */
    // Assert(GetComputeMode() != PARALLEL_COMPUTE);
    MemoryContext old = CURRENT_MEMORY_CONTEXT;
    CURRENT_MEMORY_CONTEXT = currentConext;
    return old;
}

/* Alloc from MemoryContext. */
void *MemoryContextAlloc(size_t size) {
    MemoryContext context = CURRENT_MEMORY_CONTEXT;
    return context->context_methods->alloc(context, size);
}

/* Free from MemoryContext. */
void MemoryContextFree(void *ptr) {
    MemoryContext context = CURRENT_MEMORY_CONTEXT;
    context->context_methods->free(ptr);
}

/* Calloc from MemoryContext. */
void *MemoryContextCalloc(size_t n, size_t s) {
    size_t size = n * s;
    void *p = MemoryContextAlloc(size);
    memset(p, 0, size);
    return p;
}

/* Realloc from MemoryContext. */
void *MemoryContextRealloc(void *pointer, size_t size) {
    MemoryContext context = CURRENT_MEMORY_CONTEXT;
    return context->context_methods->realloc(pointer, size);
}

/* Strdup from MemoryContext. */
char *MemoryContextStrdup(char *str) {
    size_t len = strlen(str) + 1;
    char *nstr = MemoryContextAlloc(len);
    memcpy(nstr, str, len);
    return nstr;
}
