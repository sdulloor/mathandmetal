TunableOp Benchmark Tool
========================

This tool allows you to run benchmarks for matrix multiplication operations and analyze the results, particularly focusing on the impact of PyTorch's TunableOP feature.

Requirements:
-------------
- Python 3.10+
- PyTorch
- Click
- Matplotlib
- NumPy

Installation:
-------------
1. Ensure you have Python 3.10 or later installed.
2. Install the required packages:
   pip install -r requirements.txt

Usage:
------
The tool provides two main commands: 'run' for running benchmarks and 'plot' for visualizing results.

1. Running a benchmark:
   python bench_tunableop.py run --config <config_file.json> --output <output_file.csv>

   Options:
   --config: Path to the JSON configuration file (required)
   --output: Name of the output CSV file (default: benchmark_results.csv)

2. Plotting results:
   python bench_tunableop.py plot <input_file.csv> --output <output_directory>

   Arguments:
   input_file.csv: Path to the CSV file containing benchmark results

   Options:
   --output: Directory to save the generated plots (default: results/)

Configuration File:
-------------------
The configuration file (JSON format) should contain the following fields:
{
    "test_type": "gemm",
    "dtypes": ["fp32", "fp16", "bf16"],
    "sizes": {
        "M": [64, 128, 256, 512, 1024, 2048, 4096],
        "N": [64, 128, 256, 512, 1024, 2048, 4096],
        "K": [64, 128, 256, 512, 1024, 2048, 4096]
    },
    "warmup_iterations": 10,
    "test_iterations": 10000
}

Output:
-------
1. The 'run' command produces a CSV file with benchmark results.
2. The 'plot' command generates three PNG files:
   - matmul_performance_tflops.png: Shows TFLOPs performance for tuned and untuned cases
   - matmul_performance_improvement.png: Visualizes percentage improvement with tuning
   - matmul_performance_top_bottom_improvement.png: Highlights top improvements and degradations

Notes:
------
- The benchmark runs each configuration twice: once with TunableOP enabled and once disabled.
- Ensure you have an AMD MI series GPU (tested only on MI300) for hardware acceleration.
- Large matrix sizes or a high number of iterations may lead to long execution times.