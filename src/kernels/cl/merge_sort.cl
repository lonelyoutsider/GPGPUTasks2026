#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Функция lower_bound для работы с глобальной памятью
uint lower_bound(__global const uint* arr, uint size, uint value) {
    uint left = 0;
    uint right = size; // Интервал [left, right)

    while (left < right) {
        // Защита от переполнения left + right для больших массивов
        uint mid = left + ((right - left) >> 1);

        if (arr[mid] < value) {
            left = mid + 1; // Искомый элемент находится правее mid
        } else {
            right = mid;    // Искомый элемент равен mid или левее mid
        }
    }
    return left; // Возвращает индекс первого элемента >= value (или size, если такого нет)
}

    // Возвращает индекс первого элемента, который строго больше, чем value.
// Если таких элементов нет, возвращает размер массива (size).
uint upper_bound(__global const uint* arr, uint size, uint value) {
    uint low = 0;
    uint high = size;

    while (low < high) {
        uint mid = low + (high - low) / 2;

        if (value >= arr[mid]) {
            low = mid + 1; // Искомый элемент находится правее mid
        } else {
            high = mid;    // Искомый элемент это mid или левее
        }
    }
    return low;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint i = get_global_id(0);

    if (i >= (uint)n) {
        return;
    }

    unsigned int k = sorted_k;

    uint base = (i / (2u * k)) * (2u * k);

    const uint count = (uint)n;
    uint mid = min(base + k, count);
    uint end = min(base + (2u * k), count);
    unsigned int pos = 0;
    const uint x = input_data[i];

    if (i < mid) {
        pos = i + lower_bound(input_data + mid, end - mid, x);
    } else {
        pos = base + (i - mid) + upper_bound(input_data + base, mid - base, x);
    }

    output_data[pos] = x;
}
