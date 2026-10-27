#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#include "helpers/rassert.cl"
#include "../defines.h"

// Reduction of a flattened digit-major histogram.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* current_sums,
    __global uint* next_sums,
    uint current_n)
{
    const uint i = get_global_id(0);
    const uint left = 2u * i;
    if (left >= current_n) return;
    uint sum = current_sums[left];
    if (left + 1u < current_n) sum += current_sums[left + 1u];
    next_sums[i] = sum;
}

// Work-efficient exclusive scan of 1024 values using 256 work-items.
// Higher hierarchy levels scan their block sums in place. All reads precede
// the first barrier, so input == prefix is safe, including a partial tile.
// Padding avoids 32-bank conflicts during the Blelloch tree traversal.
#define SCAN_INDEX(i) ((i) + ((i) >> 5u))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scan_blocks(
    __global const uint* input,
    __global uint* prefix,
    __global uint* block_sums,
    uint count)
{
    const uint lid = get_local_id(0);
    const uint block = get_group_id(0);
    const uint base = block * RADIX_SCAN_BLOCK_SIZE + 4u * lid;
    uint4 values = (uint4)(0u);
    if (base + 3u < count) {
        values = vload4(0, input + base);
    } else {
        if (base < count) values.s0 = input[base];
        if (base + 1u < count) values.s1 = input[base + 1u];
        if (base + 2u < count) values.s2 = input[base + 2u];
    }

    __local uint sums[GROUP_SIZE + GROUP_SIZE / 32];
    sums[SCAN_INDEX(lid)] = values.s0 + values.s1 + values.s2 + values.s3;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1u; stride < GROUP_SIZE; stride <<= 1u) {
        const uint right = (lid + 1u) * (stride << 1u) - 1u;
        if (right < GROUP_SIZE)
            sums[SCAN_INDEX(right)] += sums[SCAN_INDEX(right - stride)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0u) {
        // The top level has no consumer of its total, hence needs no buffer.
        if (count > RADIX_SCAN_BLOCK_SIZE)
            block_sums[block] = sums[SCAN_INDEX(GROUP_SIZE - 1u)];
        sums[SCAN_INDEX(GROUP_SIZE - 1u)] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        const uint right = (lid + 1u) * (stride << 1u) - 1u;
        if (right < GROUP_SIZE) {
            const uint left = sums[SCAN_INDEX(right - stride)];
            sums[SCAN_INDEX(right - stride)] = sums[SCAN_INDEX(right)];
            sums[SCAN_INDEX(right)] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint offset = sums[SCAN_INDEX(lid)];
    const uint4 result = offset + (uint4)(0u, values.s0,
        values.s0 + values.s1, values.s0 + values.s1 + values.s2);
    if (base + 3u < count) {
        vstore4(result, 0, prefix + base);
    } else {
        if (base < count) prefix[base] = result.s0;
        if (base + 1u < count) prefix[base + 1u] = result.s1;
        if (base + 2u < count) prefix[base + 2u] = result.s2;
    }
}
