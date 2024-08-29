import torch
import csv
import os
import matplotlib.pyplot as plt
from itertools import product
import sys
import subprocess
import click
import numpy as np

def time_matmul(M, N, K):
    n_iter = 10000
    n_warmup = 10
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    A = torch.rand(M, K, device="cuda")
    B = torch.rand(K, N, device="cuda")
    for i in range(n_iter + n_warmup):
        if i == n_warmup:
            t0.record()
        C = A @ B
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000
    return n_iter/dt, dt

def run_benchmark(config):
    M_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    N_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    K_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    results = []
    total_experiments = len(list(product(M_sizes, N_sizes, K_sizes)))
    experiment_count = 0

    for M, N, K in product(M_sizes, N_sizes, K_sizes):
        experiment_count += 1
        click.echo(f"Running experiment {experiment_count}/{total_experiments}: M={M}, N={N}, K={K}")
        iter_per_sec, elapsed_time = time_matmul(M, N, K)
        click.echo(f"Tuning enabled: {config} -> M: {M}, N: {N}, K: {K}, iter_per_sec: {iter_per_sec}, elapsed_time: {elapsed_time}")
        results.append({
            'M': M,
            'N': N,
            'K': K,
            'config': config,
            'iter_per_sec': iter_per_sec,
            'elapsed_time': elapsed_time
        })

    return results

def save_results(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['M', 'N', 'K', 'config', 'iter_per_sec', 'elapsed_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(filename, output_dir="results/"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    data = {'w/o tuning': [], 'with tuning': []}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            config = 'with tuning' if row['config'] == '1' else 'w/o tuning'
            M, N, K = int(row['M']), int(row['N']), int(row['K'])
            n_iter = 10000  # Make sure this matches the n_iter in your benchmark function
            flops = 2 * M * N * K * n_iter
            tflops = (flops / 1e12) / float(row['elapsed_time'])
            data[config].append({
                'size': M * N * K,
                'shape': (M, N, K),
                'tflops': tflops
            })

    # Sort data by size
    for config in data:
        data[config].sort(key=lambda x: x['size'])

    # Prepare data for plotting
    sizes = [p['size'] for p in data['w/o tuning']]
    shapes = [p['shape'] for p in data['w/o tuning']]
    tflops_wo = [p['tflops'] for p in data['w/o tuning']]
    tflops_w = [p['tflops'] for p in data['with tuning']]

    # Calculate percentage improvement
    percent_improvement = [(w - wo) / wo * 100 for w, wo in zip(tflops_w, tflops_wo)]

    # Figure 1: TFLOPs Performance
    plt.figure(figsize=(16, 10), dpi=300)
    scatter_wo = plt.scatter(sizes, tflops_wo, label='w/o tuning', alpha=0.7)
    scatter_w = plt.scatter(sizes, tflops_w, label='with tuning', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Matrix Size (M*N*K)', fontsize=12)
    plt.ylabel('TFLOPs', fontsize=12)
    plt.title('MatMul Performance', fontsize=14)
    plt.legend(fontsize=10)

    # Annotate best and worst points only for 'with tuning'
    tflops_list = tflops_w
    top_idx = np.argmax(tflops_list)
    bottom_idx = np.argmin(tflops_list)
    
    for idx, point_type in [(top_idx, 'Best'), (bottom_idx, 'Worst')]:
        x_pos = sizes[idx]
        y_pos = tflops_list[idx]
        
        # Adjust y position for annotation to avoid overlap
        y_pos *= 1.1 if point_type == 'Best' else 0.9
        
        plt.annotate(f'{point_type}: ({shapes[idx][0]},{shapes[idx][1]},{shapes[idx][2]})',
                     (x_pos, y_pos),
                     xytext=(5, 5), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matmul_performance_tflops.png'))
    plt.close()

    # Figure 2: Percentage Improvement
    plt.figure(figsize=(16, 10), dpi=300)
    scatter = plt.scatter(sizes, percent_improvement, c=percent_improvement, cmap='viridis')
    plt.xscale('log')
    plt.xlabel('Matrix Size (M*N*K)', fontsize=12)
    plt.ylabel('Percentage Improvement (%)', fontsize=12)
    plt.title('Performance Improvement with Tuning', fontsize=14)
    plt.colorbar(scatter, label='Percentage Improvement (%)')

    # Annotate top 5 improvements
    top_5_indices = np.argsort(percent_improvement)[-5:][::-1]
    for idx in top_5_indices:
        plt.annotate(f'({shapes[idx][0]},{shapes[idx][1]},{shapes[idx][2]})\n{percent_improvement[idx]:.2f}%',
                     (sizes[idx], percent_improvement[idx]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matmul_performance_improvement.png'))
    plt.close()

    # Figure 3: Top 5 and Bottom 5 Configurations based on improvement/degradation
    plt.figure(figsize=(16, 10), dpi=300)
    
    # Get indices of top 5 improvements and top 5 degradations
    top_5_improvements = np.argsort(percent_improvement)[-5:][::-1]
    top_5_degradations = np.argsort(percent_improvement)[:5]
    indices = np.concatenate([top_5_improvements, top_5_degradations])

    x = np.arange(10)
    improvements = [percent_improvement[i] for i in indices]

    plt.bar(x, improvements, color=['green' if i >= 0 else 'red' for i in improvements])

    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Matrix Configurations', fontsize=12)
    plt.ylabel('Percentage Improvement/Degradation (%)', fontsize=12)
    plt.title('Top 5 Improvements and Degradations with Tuning', fontsize=14)
    plt.xticks(x, [f'({shapes[i][0]},{shapes[i][1]},{shapes[i][2]})' for i in indices], rotation=45, ha='right')

    for i, v in enumerate(improvements):
        plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        # Add TFLOPs information
        plt.text(i, v, f'w/o: {tflops_wo[indices[i]]:.2f}\nw: {tflops_w[indices[i]]:.2f}', 
                 ha='center', va='top' if v >= 0 else 'bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matmul_performance_top_bottom_improvement.png'))
    plt.close()

    # Print top 5 and bottom 5 configurations
    print("Top 5 configurations (M, N, K) for 'with tuning':")
    for idx in np.argsort(tflops_w)[-5:][::-1]:
        print(f"{shapes[idx]}: {tflops_w[idx]:.2f} TFLOPs")

    print("\nBottom 5 configurations (M, N, K) for 'with tuning':")
    for idx in np.argsort(tflops_w)[:5]:
        print(f"{shapes[idx]}: {tflops_w[idx]:.2f} TFLOPs")

    print("\nTop 5 configurations (M, N, K) for 'w/o tuning':")
    for idx in np.argsort(tflops_wo)[-5:][::-1]:
        print(f"{shapes[idx]}: {tflops_wo[idx]:.2f} TFLOPs")

    print("\nBottom 5 configurations (M, N, K) for 'w/o tuning':")
    for idx in np.argsort(tflops_wo)[:5]:
        print(f"{shapes[idx]}: {tflops_w[idx]:.2f} TFLOPs")

@click.group()
def cli():
    """MatMul Benchmark Tool: Run benchmarks and plot results for matrix multiplication operations."""
    pass

@cli.command()
@click.option('--output', default='benchmark_results.csv', help='Output CSV file name')
def bench(output):
    """Run the benchmark and produce the results in a CSV file.

    Example:
    python script.py bench --output my_results.csv
    """
    if os.environ.get('PYTORCH_TUNABLEOP_RUNNING') != '1':
        # This is the parent process
        results = []
        for config in ['0', '1']:  # '0' for disabled, '1' for enabled
            env = os.environ.copy()
            env['PYTORCH_TUNABLEOP_VERBOSE'] = '1'
            env['PYTORCH_TUNABLEOP_FILENAME'] = 'src/matmul_result.csv'
            env['PYTORCH_TUNABLEOP_ENABLED'] = config
            env['PYTORCH_TUNABLEOP_RUNNING'] = '1'  # Flag to indicate we're in the subprocess

            click.echo(f"Running benchmark with PYTORCH_TUNABLEOP_ENABLED={config}")

            # Run this script as a subprocess with the modified environment
            subprocess.run([sys.executable, __file__, "bench", "--output", f"temp_{config}.csv"], env=env, check=True)

            # Read results from the temporary CSV file
            with open(f"temp_{config}.csv", 'r') as f:
                reader = csv.DictReader(f)
                new_results = list(reader)
                results.extend(new_results)
            
            # Remove temporary file
            os.remove(f"temp_{config}.csv")

        # Save combined results
        save_results(results, output)
        click.echo(f"Benchmark completed. Results saved to {output}")
    else:
        # This is the child process
        results = run_benchmark(os.environ['PYTORCH_TUNABLEOP_ENABLED'])
        save_results(results, output)

@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--output', default='matmul_performance.png', help='Output plot file name')
def plot(input, output):
    """Read the CSV file and produce the plot.

    Example:
    python script.py plot benchmark_results.csv --output my_plot.png
    """
    plot_results(input)
    click.echo(f"Plot saved as {output}")

if __name__ == '__main__':
    cli()