#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h> 
#include <time.h>
#include <tmmintrin.h>
#include <immintrin.h>


typedef unsigned int u32;

#define NUM_RUNS 100  // Number of times to run each test to find the minimum cycle count

// Function prototypes
u32 SingleScalar(u32 count, u32* input_data);
u32 Unroll2Scalar(u32 count, u32* input_data);
u32 Unroll4Scalar(u32 count, u32* input_data);
u32 Simd128(u32 count, u32* input_data);
u32 Simd256(u32 count, u32* input_data);

uint64_t measure_cycles(u32 (*func)(u32, u32*), u32* input_data, u32 size, double* cpu_clock);

void run_test(const char* func_name, u32 (*func)(u32, u32*), u32* sizes, int num_sizes);

// Function to perform addition using u32
u32 SingleScalar(u32 count, u32* input_data) {
    u32 total_sum = 0;
    for (u32 i = 0; i < count; i++) {
        total_sum += input_data[i];
    }
    return total_sum;
}

// Function to perform addition using u32 with loop unrolling by 2
u32 Unroll2Scalar(u32 count, u32* input_data) {
    u32 total_sum = 0;
    for (u32 i = 0; i < count; i += 2) {
        total_sum += input_data[i];
        total_sum += input_data[i + 1];
    }
    return total_sum;
}

// Function to perform addition using u32 with loop unrolling by 4
u32 Unroll4Scalar(u32 count, u32* input_data) {
    u32 total_sum = 0;
    for (u32 i = 0; i < count; i += 4) {
        total_sum += input_data[i];
        total_sum += input_data[i + 1];
        total_sum += input_data[i + 2];
        total_sum += input_data[i + 3];
    }
    return total_sum;
}

// Function to perform si128 addition using SIMD
u32 __attribute__((target("ssse3"))) Simd128(u32 count, u32* input_data) {
    __m128i total_sum = _mm_setzero_si128();
    u32 i;
    for (i = 0; i + 4 <= count; i += 4) {
        __m128i data = _mm_loadu_si128((__m128i*)&input_data[i]);
        total_sum = _mm_add_epi32(total_sum, data);
    }
    total_sum = _mm_hadd_epi32(total_sum, total_sum);
    total_sum = _mm_hadd_epi32(total_sum, total_sum);
    u32 result = _mm_cvtsi128_si32(total_sum);
    for (; i < count; i++) {
        result += input_data[i];
    }
    return result;
}

// Function to perform si256 addition using AVX2
u32 __attribute__((target("avx2"))) Simd256(u32 count, u32* input_data) {
    __m256i total_sum = _mm256_setzero_si256();
    u32 i;
    for (i = 0; i + 8 <= count; i += 8) {
        __m256i data = _mm256_loadu_si256((__m256i*)&input_data[i]);
        total_sum = _mm256_add_epi32(total_sum, data);
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(total_sum), _mm256_extracti128_si256(total_sum, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    u32 result = _mm_cvtsi128_si32(sum128);
    for (; i < count; i++) {
        result += input_data[i];
    }
    return result;
}

// Function to measure CPU cycles and calculate the CPU clock speed
uint64_t measure_cycles(u32 (*func)(u32, u32*), u32* input_data, u32 size, double* cpu_clock) {
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

void run_test(const char* func_name, u32 (*func)(u32, u32*), u32* sizes, int num_sizes) {
    printf("\nRunning tests for function: %s\n", func_name);
    printf("=======================================================================================================\n");
    printf("%-20s%-25s%-20s%-20s%-15s\n", "Test Size", "Result", "CPU Cycles", "CPU Clock (GHz)", "Adds per Cycle");
    printf("-------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        u32 size = sizes[i];
        u32* input_data = malloc(size * sizeof(u32));
        if (input_data == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for size %u\n", size);
            exit(EXIT_FAILURE);
        }

        for (u32 j = 0; j < size; j++) {
            input_data[j] = j;
        }

        double cpu_clock = 0;
        uint64_t cycles = measure_cycles(func, input_data, size, &cpu_clock);
        double adds_per_cycle = (double)size / cycles;
        u32 result = func(size, input_data);

        printf("%-20u%-25u%-20" PRIu64 "%-20.3f%-15.6f\n", size, result, cycles, cpu_clock / 1e9, adds_per_cycle);
        free(input_data);
    }

    printf("=======================================================================================================\n");
}

int main() {
    u32 test_sizes[] = {5000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    run_test("SingleScalar", SingleScalar, test_sizes, num_sizes);
    run_test("Unroll2Scalar", Unroll2Scalar, test_sizes, num_sizes);
    run_test("Unroll4Scalar", Unroll4Scalar, test_sizes, num_sizes);
    run_test("Simd128", Simd128, test_sizes, num_sizes);
    run_test("Simd256", Simd256, test_sizes, num_sizes);

    return 0;
}
