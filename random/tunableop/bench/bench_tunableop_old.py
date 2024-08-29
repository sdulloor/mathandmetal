import torch
import csv
import os
import matplotlib.pyplot as plt
from itertools import product
import sys
import subprocess

def time_matmul(M, N, K):
    n_iter = 10000  # Reduced number of iterations for multiple experiments
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
        print(f"Running experiment {experiment_count}/{total_experiments}: M={M}, N={N}, K={K}")
        iter_per_sec, elapsed_time = time_matmul(M, N, K)
        print(f"Tuning enabled: {config} -> M: {M}, N: {N}, K: {K}, iter_per_sec: {iter_per_sec}, elapsed_time: {elapsed_time}")
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

def plot_results(filename):
    data = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            config = row['config']
            if config not in data:
                data[config] = []
            data[config].append({
                'size': int(row['M']) * int(row['N']) * int(row['K']),
                'iter_per_sec': float(row['iter_per_sec'])
            })

    plt.figure(figsize=(12, 8))
    for config, points in data.items():
        sizes = [p['size'] for p in points]
        iter_per_sec = [p['iter_per_sec'] for p in points]
        plt.scatter(sizes, iter_per_sec, label=config)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Matrix Size (M*N*K)')
    plt.ylabel('Iterations per Second')
    plt.title('MatMul Performance')
    plt.legend()
    plt.savefig('matmul_performance.png')
    plt.close()

def main():
    if os.environ.get('PYTORCH_TUNABLEOP_RUNNING') != '1':
        # This is the parent process
        results = []
        for config in ['disabled', 'enabled']:
            env = os.environ.copy()
            env['PYTORCH_TUNABLEOP_VERBOSE'] = '1'
            env['PYTORCH_TUNABLEOP_FILENAME'] = 'src/matmul_result.csv'
            env['PYTORCH_TUNABLEOP_ENABLED'] = '1' if config == 'enabled' else '0'
            env['PYTORCH_TUNABLEOP_RUNNING'] = '1'  # Flag to indicate we're in the subprocess
            
            print(f"Running benchmark with PYTORCH_TUNABLEOP_ENABLED={env['PYTORCH_TUNABLEOP_ENABLED']}")
            
            # Run this script as a subprocess with the modified environment
            subprocess.run([sys.executable, __file__], env=env, check=True)
            
            # Read results from the CSV file
            with open('benchmark_results.csv', 'r') as f:
                reader = csv.DictReader(f)
                new_results = list(reader)
                results.extend(new_results)
            
            # Save combined results after each config
            save_results(results, 'combined_benchmark_results.csv')
            print(f"Updated combined results saved after {config} config")
            
            # Optionally, update the plot after each config
            #plot_results('combined_benchmark_results.csv')
            #print(f"Updated plot saved after {config} config")
        
        # Final plot (although it's the same as the last intermediate plot)
        plot_results('combined_benchmark_results.csv')
        print("Benchmark completed. Final results and plot saved.")
    else:
        # This is the child process
        results = run_benchmark(os.environ['PYTORCH_TUNABLEOP_ENABLED'])
        save_results(results, 'benchmark_results.csv')

if __name__ == "__main__":
    main()