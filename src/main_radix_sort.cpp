#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "radix_scan.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>
#include <cstdlib>

namespace {
int problemSize()
{
    const char* value = std::getenv("TASK05_SIZE");
    if (!value) return 100000000;
    size_t end = 0;
    const auto n = std::stoull(value, &end);
    if (value[end] != '\0' || n == 0 || n > 100000000)
        throw std::runtime_error("TASK05_SIZE must be in [1, 100000000]");
    return static_cast<int>(n);
}

void testHistogramScan()
{
    FastRandom random;
    for (unsigned int n : {1u, 10u, 16u, 255u, 256u, 257u, 1023u, 1024u,
            1025u, 6104u, 1048575u, 1048576u, 1048577u, 6250000u}) {
        RadixHistogramScan scan(n);
        gpu::gpu_mem_32u input(n), output(n);
        std::vector<unsigned int> values(n);
        for (unsigned int pattern = 0; pattern < 3; ++pattern) {
            for (unsigned int i = 0; i < n; ++i)
                values[i] = pattern == 0 ? random.next(0, 256)
                    : pattern == 1 ? 1u : (i % 7u == 0 ? 256u : 0u);
            input.writeN(values.data(), n);
            // Reuse buffers with different data to expose stale prefix values.
            scan.exec(input, output);
            const auto actual = output.readVector();
            unsigned int sum = 0;
            for (unsigned int i = 0; i < n; ++i) {
                rassert(actual[i] == sum, 6235161, n, pattern, i, actual[i], sum);
                sum += values[i];
            }
            rassert(input.readVector() == values, 6235162, n, pattern);
        }
        std::cout << "PASS histogram exclusive scan n=" << n << " patterns=3" << std::endl;
    }
}
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DGPU_CUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    if (std::getenv("TASK05_SCAN_TEST")) {
        testHistogramScan();
        return;
    }
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    const int n = problemSize();
    const char* scan_env = std::getenv("TASK05_SCAN");
    const std::string scan_mode = scan_env ? scan_env : "hierarchical";
    if (scan_mode != "baseline" && scan_mode != "hierarchical")
        throw std::runtime_error("TASK05_SCAN must be baseline or hierarchical");
    const bool baseline_scan = scan_mode == "baseline";
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    const char* pattern_env = std::getenv("TASK05_PATTERN");
    const std::string pattern = pattern_env ? pattern_env : "random";
    if (pattern != "random" && pattern != "full-u32" &&
        pattern != "duplicates" && pattern != "equal" && pattern != "descending")
        throw std::runtime_error("Unknown TASK05_PATTERN");

    unsigned int bit_count = 0;

    for (unsigned int v = static_cast<unsigned int>(max_value);
        v > 0;
        v >>= 1) {
        ++bit_count;
    }

    if (bit_count == 0) {
        bit_count = 1;
    }
    if (pattern != "random") bit_count = 32;

    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
        if (pattern == "full-u32")
            as[i] = (as[i] << 1u) | static_cast<unsigned int>(r.next(0, 1));
        else if (pattern == "duplicates")
            as[i] = (as[i] % 16u) * 0x11111111u;
        else if (pattern == "equal")
            as[i] = 0xffffffffu;
        else if (pattern == "descending")
            as[i] = 0xffffffffu - static_cast<unsigned int>(i);
    }
    std::cout << "n=" << n << " max_value=" << max_value
              << " pattern=" << pattern << " scan=" << scan_mode << std::endl;

    if (n > 1) {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
        // Guarantee duplicates independently of random index selection.
        as[n - 1] = as[0];
    }
    if (pattern == "full-u32" && n >= 4) {
        as[0] = 0u;
        as[1] = 0x80000000u;
        as[2] = 0xffffffffu;
        as[n - 1] = 0xffffffffu;
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double elapsed = t.elapsed();
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / elapsed << " GB/s (" << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u work_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);

    gpu::gpu_mem_32u local_one_rank_gpu(n);

    const unsigned int num_groups =
        (n + GROUP_SIZE - 1) / GROUP_SIZE;
    const unsigned int hist_size = 16u * num_groups;

    gpu::gpu_mem_32u block_one_counts_gpu(hist_size);
    gpu::gpu_mem_32u global_one_prefix_gpu(hist_size);

    std::vector<gpu::gpu_mem_32u> levels;
    RadixHistogramScan histogram_scan(hist_size);
    
    if (baseline_scan) {
        for (unsigned int current_n = hist_size; current_n > 1;) {
            current_n = (current_n + 1) / 2;
            levels.emplace_back(current_n);
        }
    }

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    //buffer1_gpu.fill(255);
    //buffer2_gpu.fill(255);
    //buffer3_gpu.fill(255);
    //buffer4_gpu.fill(255);
    //buffer_output_gpu.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            const unsigned int passes = (bit_count + 3u) / 4u;
            const gpu::gpu_mem_32u* current_input = &input_gpu;
            gpu::gpu_mem_32u* current_output =
                (passes & 1u) ? &buffer_output_gpu : &work_gpu;

            for (unsigned int pass = 0u; pass < passes; ++pass) {
                const unsigned int shift = 4u * pass;
                ocl_radixSort01LocalCounting.exec(
                    gpu::WorkSize(GROUP_SIZE, n),
                    *current_input, local_one_rank_gpu, block_one_counts_gpu,
                    static_cast<unsigned int>(n), num_groups, shift);

                if (baseline_scan) {
                    unsigned int current_n = hist_size;
                    for (size_t level = 0; level < levels.size(); ++level) {
                        const unsigned int next_n = (current_n + 1u) / 2u;
                        const gpu::gpu_mem_32u& src = level == 0u
                            ? block_one_counts_gpu : levels[level - 1u];
                        ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                            gpu::WorkSize(GROUP_SIZE, next_n),
                            src, levels[level], current_n);
                        current_n = next_n;
                    }

                    ocl_fillBufferWithZeros.exec(
                        gpu::WorkSize(GROUP_SIZE, hist_size),
                        global_one_prefix_gpu, hist_size);
                    ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                        gpu::WorkSize(GROUP_SIZE, hist_size),
                        block_one_counts_gpu, global_one_prefix_gpu, hist_size, 0u);
                    for (size_t level = 0; level < levels.size(); ++level) {
                        ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                            gpu::WorkSize(GROUP_SIZE, hist_size),
                            levels[level], global_one_prefix_gpu, hist_size,
                            static_cast<unsigned int>(level + 1u));
                    }
                } else {
                    // Three local scans + two offset additions for 6,250,000
                    // counts. Reuse block sums as their own scanned prefixes.
                    // The queue is in order; scatter below waits for completion.
                    histogram_scan.exec(block_one_counts_gpu, global_one_prefix_gpu);
                }

                ocl_radixSort04Scatter.exec(
                    gpu::WorkSize(GROUP_SIZE, n),
                    *current_input, local_one_rank_gpu, global_one_prefix_gpu,
                    *current_output, static_cast<unsigned int>(n),
                    num_groups, shift);

                if (pass + 1u < passes) {
                    current_input = current_output;
                    current_output = (current_output == &buffer_output_gpu)
                        ? &work_gpu : &buffer_output_gpu;
                }
            }
        } else if (context.type() == gpu::Context::TypeCUDA) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::radix_sort_01_local_counting();
            // cuda::radix_sort_02_global_prefixes_scan_sum_reduction();
            // cuda::radix_sort_03_global_prefixes_scan_accumulation();
            // cuda::radix_sort_04_scatter();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // vk_fillBufferWithZeros.exec();
            // vk_radixSort01LocalCounting.exec();
            // vk_radixSort02GlobalPrefixesScanSumReduction.exec();
            // vk_radixSort03GlobalPrefixesScanAccumulation.exec();
            // vk_radixSort04Scatter.exec();
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
    std::cout << "median_ms=" << stats::median(times) * 1000.0 << " measurements=" << times.size() << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
    std::cout << "PASS: std::sort and unchanged input_gpu, n=" << n << std::endl;
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            exit_code = 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
