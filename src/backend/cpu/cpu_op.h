#ifndef __CPU_OP_H__
#define __CPU_OP_H__

#include "../../def.h"

/*
 * cpu_graph_op — CPU twin of gpu_graph_op().
 *
 * The executor calls it once per node with host-resident pointers; the
 * op_table below implements every GraphOp with the scalar / SIMD kernels
 * in core.c.  Returns false for unhandled ops (OP_INPUT / OP_H2D / OP_D2H
 * stay with the executor) or structural failures.
 */
bool cpu_graph_op(OpCtx *c);

#endif
