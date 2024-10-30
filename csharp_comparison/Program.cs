using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Seventy.ComputerEnhance.Sum;

internal static class Program
{
    private const int NumRuns = 100; // Number of runs for measuring minimum cycle count
    private const double CpuFrequencyGHz = 4.4; // Assume an approximate CPU frequency in GHz

    private static ulong SingleScalar(ulong count, ulong[] inputData)
    {
        ulong totalSum = 0;
        for (ulong i = 0; i < count; i++)
        {
            totalSum += inputData[i];
        }
        return totalSum;
    }

    private static ulong LinqSum(ulong count, ulong[] inputData)
    {
        return (ulong)inputData.Select(x => (long)x).Sum();
    }

    private static ulong LinqAggregate(ulong count, ulong[] inputData)
    {
        return inputData.Aggregate(0UL, (a, b) => a + b);
    }

    private static ulong Unroll2Scalar(ulong count, ulong[] inputData)
    {
        ulong totalSum = 0;
        for (ulong i = 0; i < count; i += 2)
        {
            totalSum += inputData[i] + inputData[i + 1];
        }
        return totalSum;
    }

    private static ulong Unroll4Scalar(ulong count, ulong[] inputData)
    {
        ulong totalSum = 0;
        for (ulong i = 0; i < count; i += 4)
        {
            totalSum += inputData[i] + inputData[i + 1] + inputData[i + 2] + inputData[i + 3];
        }
        return totalSum;
    }

    private static ulong ParallelSingleScalar(ulong count, ulong[] inputData)
    {
        ulong totalSum = 0;
        var lockObj = new object();

        Parallel.For(0, (int)count, () => 0UL, (i, state, localSum) =>
            {
                localSum += inputData[i];
                return localSum;
            },
            localSum =>
            {
                lock (lockObj)
                {
                    totalSum += localSum;
                }
            });

        return totalSum;
    }

    private static ulong ParallelUnroll4Scalar(ulong count, ulong[] inputData)
    {
        ulong totalSum = 0;
        var lockObj = new object();

        Parallel.For(0, (int)count / 4, () => 0UL, (i, state, localSum) =>
            {
                var index = i * 4;
                if (index + 3 >= (int)count) return localSum;
            
                localSum += inputData[index];
                localSum += inputData[index + 1];
                localSum += inputData[index + 2];
                localSum += inputData[index + 3];
            
                return localSum;
            },
            localSum =>
            {
                lock (lockObj)
                {
                    totalSum += localSum;
                }
            });

        for (var i = (count / 4) * 4; i < count; i++)
        {
            totalSum += inputData[i];
        }

        return totalSum;
    }

    private static unsafe ulong Simd256(ulong count, ulong[] inputData)
    {
        var totalSum = Vector256<ulong>.Zero;
        ulong i;

        fixed (ulong* dataPtr = inputData)
        {
            for (i = 0; i + 4 <= count; i += 4)
            {
                var data = Avx.LoadVector256(dataPtr + i);
                totalSum = Avx2.Add(totalSum, data);
            }

            var result = new ulong[4];
            fixed (ulong* resultPtr = result)
            {
                Avx.Store(resultPtr, totalSum);
            }

            var finalSum = result[0] + result[1] + result[2] + result[3];

            for (; i < count; i++)
            {
                finalSum += inputData[i];
            }

            return finalSum;
        }
    }
    
    private static unsafe ulong ParallelUnroll4Simd256(ulong count, ulong[] inputData)
    {
        var numThreads = Environment.ProcessorCount;
        var partialSums = new ulong[numThreads];

        Parallel.For(0, numThreads, threadIndex =>
        {
            var chunkSize = (int)(count / (ulong)numThreads);
            var start = (ulong)threadIndex * (ulong)chunkSize;
            var end = (threadIndex == numThreads - 1) ? count : start + (ulong)chunkSize;

            var totalSum = Vector256<ulong>.Zero;

            fixed (ulong* dataPtr = inputData)
            {
                // Unroll loop by 4
                ulong i;
                for (i = start; i + 16 <= end; i += 16)
                {
                    var data1 = Avx.LoadVector256(dataPtr + i);
                    var data2 = Avx.LoadVector256(dataPtr + i + 4);
                    var data3 = Avx.LoadVector256(dataPtr + i + 8);
                    var data4 = Avx.LoadVector256(dataPtr + i + 12);

                    totalSum = Avx2.Add(totalSum, data1);
                    totalSum = Avx2.Add(totalSum, data2);
                    totalSum = Avx2.Add(totalSum, data3);
                    totalSum = Avx2.Add(totalSum, data4);
                }

                // Sum up the SIMD register values into an array for final accumulation
                var result = new ulong[4];
                fixed (ulong* resultPtr = result)
                {
                    Avx.Store(resultPtr, totalSum);
                }

                var partialSum = result[0] + result[1] + result[2] + result[3];

                // Handle any remaining elements in the chunk that couldn’t fit into 16-element chunks
                for (; i < end; i++)
                {
                    partialSum += inputData[i];
                }

                partialSums[threadIndex] = partialSum;
            }
        });

        // Aggregate the partial sums from each thread
        ulong finalSum = 0;
        // ReSharper disable once LoopCanBeConvertedToQuery
        foreach (var partialSum in partialSums)
        {
            finalSum += partialSum;
        }

        return finalSum;
    }


    private static double MeasureCycles(Func<ulong, ulong[], ulong> func, ulong count, ulong[] inputData, out double elapsedTimeSeconds)
    {
        var minTicks = long.MaxValue;

        for (var run = 0; run < NumRuns; run++)
        {
            var startTicks = Stopwatch.GetTimestamp();
            func(count, inputData);
            var endTicks = Stopwatch.GetTimestamp();

            var elapsedTicks = endTicks - startTicks;
            minTicks = Math.Min(minTicks, elapsedTicks);
        }

        elapsedTimeSeconds = minTicks / (double)Stopwatch.Frequency;
        return elapsedTimeSeconds * CpuFrequencyGHz * 1e9;
    }

    private static void RunTest(string funcName, Func<ulong, ulong[], ulong> func, ulong[] sizes)
    {
        Console.WriteLine($"\nRunning tests for function: {funcName}");
        Console.WriteLine(new string('=', 100));
        Console.WriteLine("{0,-20}{1,-25}{2,-20}{3,-15}{4,-15}", "Test Size", "Result", "Time Taken (s)", "CPU Cycles", "Adds per Cycle");
        Console.WriteLine(new string('-', 100));

        foreach (var size in sizes)
        {
            var inputData = new ulong[size];
            for (ulong i = 0; i < size; i++)
            {
                inputData[i] = i;
            }

            var cycles = MeasureCycles(func, size, inputData, out var elapsedTimeSeconds);
            var addsPerCycle = size / cycles;
            var result = func(size, inputData);

            Console.WriteLine("{0,-20}{1,-25}{2,-20:F6}{3,-15:F0}{4,-15:F6}", size, result, elapsedTimeSeconds, cycles, addsPerCycle);
        }

        Console.WriteLine(new string('=', 100));
    }

    private static void Main()
    {
        ulong[] testSizes = { 5000, 20000, 312500, 6000000, 25000000 };

        RunTest("SingleScalar", SingleScalar, testSizes);
        // RunTest("LinqSum", LinqSum, testSizes);
        // RunTest("LinqAggregate", LinqAggregate, testSizes);
        // RunTest("Unroll2Scalar", Unroll2Scalar, testSizes);
        // RunTest("Unroll4Scalar", Unroll4Scalar, testSizes);
        // RunTest("ParallelSingleScalar", ParallelSingleScalar, testSizes);
        RunTest("ParallelUnroll4Scalar", ParallelUnroll4Scalar, testSizes);
        RunTest("Simd256", Simd256, testSizes);
        RunTest("ParallelUnroll4Simd256", ParallelUnroll4Simd256, testSizes);
    }
}