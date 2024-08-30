import torch
from itertools import product

def time_matmul(M, N, K, dtype, warmup_iterations, test_iterations):
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map[dtype]

    A = torch.rand(M, K, device="cuda", dtype=torch_dtype)
    B = torch.rand(K, N, device="cuda", dtype=torch_dtype)

    # Warmup
    for _ in range(warmup_iterations):
        C = A @ B

    # Benchmark
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    
    t0.record()
    for _ in range(test_iterations):
        C = A @ B
    t1.record()
    
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000  # Convert to seconds
    return test_iterations/dt, dt

def run_gemm_benchmark(config, tuning_enabled):
    results = []
    total_experiments = len(list(product(config['sizes']['M'], config['sizes']['N'], config['sizes']['K'], config['dtypes'])))
    experiment_count = 0

    for M, N, K, dtype in product(config['sizes']['M'], config['sizes']['N'], config['sizes']['K'], config['dtypes']):
        experiment_count += 1
        print(f"Running experiment {experiment_count}/{total_experiments}: M={M}, N={N}, K={K}, dtype={dtype}")
        iter_per_sec, elapsed_time = time_matmul(M, N, K, dtype, config['warmup_iterations'], config['test_iterations'])
        
        # Calculate TFLOPs
        flops = 2 * M * N * K * config['test_iterations']  # Multiply-adds count as 2 operations
        tflops = (flops / 1e12) / elapsed_time
        
        print(f"Tuning enabled: {tuning_enabled} -> M: {M}, N: {N}, K: {K}, dtype: {dtype}, "
              f"iter_per_sec: {iter_per_sec:.2f}, elapsed_time: {elapsed_time:.4f}, TFLOPs: {tflops:.2f}")
        
        results.append({
            'M': M,
            'N': N,
            'K': K,
            'dtype': dtype,
            'config': tuning_enabled,
            'iter_per_sec': iter_per_sec,
            'elapsed_time': elapsed_time
        })

    return results