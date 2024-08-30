import click
import json
import os
import subprocess
import sys
import csv
from bench_gemm import run_gemm_benchmark
from plot_gemm import plot_gemm_results

@click.group()
def cli():
    """TunableOp Benchmark Tool: Run benchmarks and plot results for various operations."""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration file')
@click.option('--output', default='benchmark_results.csv', help='Output CSV file name')
def run(config, output):
    """Run the benchmark based on the provided configuration."""
    with open(config, 'r') as f:
        config_data = json.load(f)

    test_type = config_data['test_type']
    if test_type != 'gemm':
        click.echo(f"Test type {test_type} is not implemented yet.")
        return

    if os.environ.get('PYTORCH_TUNABLEOP_RUNNING') != '1':
        # This is the parent process
        results = []
        for tuning in ['0', '1']:  # '0' for disabled, '1' for enabled
            env = os.environ.copy()
            env['PYTORCH_TUNABLEOP_VERBOSE'] = '1'
            env['PYTORCH_TUNABLEOP_FILENAME'] = 'src/matmul_result.csv'
            env['PYTORCH_TUNABLEOP_ENABLED'] = tuning
            env['PYTORCH_TUNABLEOP_RUNNING'] = '1'

            click.echo(f"Running benchmark with PYTORCH_TUNABLEOP_ENABLED={tuning}")

            # Run this script as a subprocess with the modified environment
            subprocess.run([sys.executable, __file__, "run", "--config", config, "--output", f"temp_{tuning}.csv"], env=env, check=True)

            # Read results from the temporary CSV file
            with open(f"temp_{tuning}.csv", 'r') as f:
                results.extend(list(csv.DictReader(f)))

            # Remove temporary file
            os.remove(f"temp_{tuning}.csv")

        # Save combined results
        with open(output, 'w', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        click.echo(f"Benchmark completed. Results saved to {output}")
    else:
        # This is the child process
        results = run_gemm_benchmark(config_data, os.environ['PYTORCH_TUNABLEOP_ENABLED'])
        with open(output, 'w', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--output', default='results', help='Output directory for plots')
def plot(input, output):
    """Read the CSV file and produce the plots."""
    plot_gemm_results(input, output)
    click.echo(f"Plots saved in directory: {output}")

if __name__ == '__main__':
    cli()