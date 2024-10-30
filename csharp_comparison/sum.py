import time
import numpy

# Assumed CPU frequency in cycles per second
CPU_FREQUENCY_HZ = 4_400_000_000  # 4.4 GHz for All-Core Turbo
NUM_RUNS = 2  # Number of times to run each test

def SingleScalar(count, input_data):
    total_sum = 0
    for i in range(count):
        total_sum += input_data[i]
    return total_sum

def SingleScalarNoRange(_, input_data):
    total_sum = 0
    for i in input_data:
        total_sum += i
    return total_sum

def NumpySum(_, input_data):
    return numpy.sum(input_data)

def BuiltinSum(_, input_data):
    return sum(input_data)

def run_test(func, sizes):
    print(f"\nRunning tests for function: {func.__name__}")
    print("=" * 100)
    print(f"{'Test Size':<20}{'Result':<25}{'Time Taken (s)':<20}{'CPU Cycles':<15}{'Adds per Cycle':<15}")
    print("-" * 100)

    for size in sizes:
        input_data = list(range(size))
        min_elapsed_time_s = float("inf")  # Initialize with infinity
        result = None

        # Run the function multiple times and keep the minimum time
        for _ in range(NUM_RUNS):
            start_time = time.perf_counter_ns()
            result = func(size, input_data)
            end_time = time.perf_counter_ns()
            elapsed_time_s = (end_time - start_time) / 1_000_000_000  # Convert ns to seconds
            if elapsed_time_s < min_elapsed_time_s:
                min_elapsed_time_s = elapsed_time_s

        # Calculate CPU cycles and adds per cycle based on the fastest time
        cpu_cycles = min_elapsed_time_s * CPU_FREQUENCY_HZ
        adds_per_cycle = size / cpu_cycles

        # Print results for this test size
        print(f"{size:<20}{result:<25}{min_elapsed_time_s:<20.6f}{int(cpu_cycles):<15}{adds_per_cycle:<15.6f}")
    print("=" * 100)

if __name__ == "__main__":
    test_sizes = [5000, 20000, 312500, 6000000, 25000000]
    run_test(SingleScalar, test_sizes)
    run_test(SingleScalarNoRange, test_sizes)
    run_test(NumpySum, test_sizes)
    run_test(BuiltinSum, test_sizes)
