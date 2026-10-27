#pragma once

#include "kernels/defines.h"
#include "kernels/kernels.h"

// All kernels use the context's in-order queue. The caller must wait for the
// result (e.g. synchronous scatter or readback) before stopping its timer.
class RadixHistogramScan {
public:
    explicit RadixHistogramScan(unsigned int count)
        : scan_(ocl::getRadixSort02GlobalPrefixesScanSumReduction(), "radix_sort_scan_blocks"),
          add_(ocl::getRadixSort03GlobalPrefixesScanAccumulation(), "radix_sort_scan_add_offsets"),
          counts_{count}
    {
        if (count == 0) throw std::runtime_error("Scan count must be positive");
        while (count > RADIX_SCAN_BLOCK_SIZE) {
            count = (count + RADIX_SCAN_BLOCK_SIZE - 1u) / RADIX_SCAN_BLOCK_SIZE;
            levels_.emplace_back(count);
            counts_.push_back(count);
        }
    }

    void exec(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& prefix)
    {
        for (size_t level = 0; level < counts_.size(); ++level) {
            const auto& src = level == 0 ? input : levels_[level - 1];
            auto& dst = level == 0 ? prefix : levels_[level - 1];
            // The top scan does not write a sum; reuse a valid buffer argument.
            auto& sums = level < levels_.size() ? levels_[level] : prefix;
            scan_.execAsynchronized(workSize(counts_[level]), src, dst, sums, counts_[level]);
        }
        for (size_t level = levels_.size(); level > 0; --level) {
            auto& dst = level == 1 ? prefix : levels_[level - 2];
            add_.execAsynchronized(workSize(counts_[level - 1]),
                dst, levels_[level - 1], counts_[level - 1]);
        }
    }

private:
    static gpu::WorkSize workSize(unsigned int count)
    {
        const auto blocks = (count + RADIX_SCAN_BLOCK_SIZE - 1u) / RADIX_SCAN_BLOCK_SIZE;
        return gpu::WorkSize(GROUP_SIZE, blocks * GROUP_SIZE);
    }

    ocl::KernelSource scan_;
    ocl::KernelSource add_;
    std::vector<unsigned int> counts_;
    std::vector<gpu::gpu_mem_32u> levels_;
};
