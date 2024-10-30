#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h> 
#include <time.h>
#include <tmmintrin.h>
#include <immintrin.h>

#define NUM_RUNS 100  // Number of times to run each test to find the minimum cycle count

// Function to perform addition using uint64_t
uint64_t SingleScalar(uint64_t count, uint64_t* input_data) {
    uint64_t total_sum = 0;
    for (uint64_t i = 0; i < count; i++) {
        total_sum += input_data[i];
    }
    return total_sum;
}

// Function to perform addition using uint64_t with loop unrolling by 2
uint64_t Unroll2Scalar(uint64_t count, uint64_t* input_data) {
    uint64_t total_sum = 0;
    for (uint64_t i = 0; i < count; i += 2) {
        total_sum += input_data[i];
        total_sum += input_data[i + 1];
    }
    return total_sum;
}

// Function to perform addition using uint64_t with loop unrolling by 4
uint64_t Unroll4Scalar(uint64_t count, uint64_t* input_data) {
    uint64_t total_sum = 0;
    for (uint64_t i = 0; i < count; i += 4) {
        total_sum += input_data[i];
        total_sum += input_data[i + 1];
        total_sum += input_data[i + 2];
        total_sum += input_data[i + 3];
    }
    return total_sum;
}

// Function to perform addition using SIMD with SSE2 for 64-bit integers
uint64_t __attribute__((target("sse2"))) Simd128(uint64_t count, uint64_t* input_data) {
    __m128i total_sum = _mm_setzero_si128();
    uint64_t i;
    for (i = 0; i + 2 <= count; i += 2) {
        __m128i data = _mm_loadu_si128((__m128i*)&input_data[i]);
        total_sum = _mm_add_epi64(total_sum, data);
    }
    uint64_t result[2];
    _mm_storeu_si128((__m128i*)result, total_sum);
    uint64_t final_sum = result[0] + result[1];

    for (; i < count; i++) {
        final_sum += input_data[i];
    }
    return final_sum;
}

// Function to perform addition using SIMD with AVX2 for 64-bit integers
uint64_t __attribute__((target("avx2"))) Simd256(uint64_t count, uint64_t* input_data) {
    __m256i total_sum = _mm256_setzero_si256();
    uint64_t i;
    for (i = 0; i + 4 <= count; i += 4) {
        __m256i data = _mm256_loadu_si256((__m256i*)&input_data[i]);
        total_sum = _mm256_add_epi64(total_sum, data);
    }
    uint64_t result[4];
    _mm256_storeu_si256((__m256i*)result, total_sum);
    uint64_t final_sum = result[0] + result[1] + result[2] + result[3];

    for (; i < count; i++) {
        final_sum += input_data[i];
    }
    return final_sum;
}

// Function to measure CPU cycles and calculate the CPU clock speed
uint64_t measure_cycles(uint64_t (*func)(uint64_t, uint64_t*), uint64_t* input_data, uint64_t size, double* cpu_clock) {
    uint64_t min_cycles = UINT64_MAX;
    struct timespec start_time, end_time;

    for (int run = 0; run < NUM_RUNS; run++) {
        uint32_t start_low, start_high, end_low, end_high;
        uint64_t start, end, cycles;
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);  // Get start time in nanoseconds
        asm volatile ("rdtsc" : "=a" (start_low), "=d" (start_high));
        func(size, input_data);
        asm volatile ("rdtsc" : "=a" (end_low), "=d" (end_high));
        clock_gettime(CLOCK_MONOTONIC, &end_time);  // Get end time in nanoseconds

        start = ((uint64_t)start_high << 32) | start_low;
        end = ((uint64_t)end_high << 32) | end_low;
        cycles = end - start;

        uint64_t elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000 + (end_time.tv_nsec - start_time.tv_nsec);
        *cpu_clock = (double)cycles / (elapsed_ns / 1e9);  // Calculate CPU clock in Hz

        if (cycles < min_cycles) {
            min_cycles = cycles;
        }
    }

    return min_cycles;
}

void run_test(const char* func_name, uint64_t (*func)(uint64_t, uint64_t*), uint64_t* sizes, int num_sizes) {
    printf("\nRunning tests for function: %s\n", func_name);
    printf("=======================================================================================================\n");
    printf("%-20s%-25s%-20s%-20s%-15s\n", "Test Size", "Result", "CPU Cycles", "CPU Clock (GHz)", "Adds per Cycle");
    printf("-------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        uint64_t size = sizes[i];
        uint64_t* input_data = malloc(size * sizeof(uint64_t));
        if (input_data == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for size %" PRIu64 "\n", size);
            exit(EXIT_FAILURE);
        }

        for (uint64_t j = 0; j < size; j++) {
            input_data[j] = j;
        }

        double cpu_clock = 0;
        uint64_t cycles = measure_cycles(func, input_data, size, &cpu_clock);
        double adds_per_cycle = (double)size / cycles;
        uint64_t result = func(size, input_data);

        printf("%-20" PRIu64 "%-25" PRIu64 "%-20" PRIu64 "%-20.3f%-15.6f\n", size, result, cycles, cpu_clock / 1e9, adds_per_cycle);
        free(input_data);
    }

    printf("=======================================================================================================\n");
}

int main() {
    uint64_t test_sizes[] = {5000, 20000, 312500, 6000000, 25000000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    run_test("SingleScalar", SingleScalar, test_sizes, num_sizes);
    run_test("Unroll2Scalar", Unroll2Scalar, test_sizes, num_sizes);
    run_test("Unroll4Scalar", Unroll4Scalar, test_sizes, num_sizes);
    run_test("Simd128", Simd128, test_sizes, num_sizes);
    run_test("Simd256", Simd256, test_sizes, num_sizes);

    return 0;
}
