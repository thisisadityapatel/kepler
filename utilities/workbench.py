#!/usr/bin/env python3
"""
Kepler LLM Workbench - Interactive model selection, serving, and benchmarking for macOS.

This script orchestrates the entire workflow:
1. Interactive model selection from available GGUF files
2. Docker container building and management
3. Model serving via llama.cpp
4. Performance benchmarking with industry-standard metrics
"""

import argparse
import sys
from pathlib import Path

from benchmark import (
    HARD_QUESTIONS_BENCHMARK,
    PERFORMANCE_BENCHMARK,
    QUICK_BENCHMARK,
    STANDARD_BENCHMARK,
    BenchmarkRunner,
)
from docker_manager import create_llama_container
from model_selector import find_gguf_models, select_model_interactive
from progress_tracker import create_workbench_tracker, status_error, status_info


def main():
    parser = argparse.ArgumentParser(
        description="Kepler LLM Workbench - Run, serve and benchmark LLM models on macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - select model and run standard benchmark
  python workbench.py

  # Quick benchmark with specific model
  python workbench.py --model models/qwen2.5-0.5b-instruct-q5_k_m.gguf --benchmark quick

  # Performance benchmark with custom port
  python workbench.py --model models/my-model.gguf --benchmark performance --port 8081
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to GGUF model file (skips interactive selection)",
    )

    parser.add_argument(
        "--benchmark",
        "-b",
        choices=["quick", "standard", "performance", "hard", "skip"],
        default="standard",
        help="Benchmark type to run (default: standard)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port for model server (default: 8080)",
    )

    parser.add_argument(
        "--ctx-size",
        "-c",
        type=int,
        default=4096,
        help="Context size for model (default: 4096)",
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="b7531",
        help="llama.cpp version to use (default: b7531)",
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    parser.add_argument(
        "--no-serve", action="store_true", help="Skip model serving (for testing)"
    )

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        models = find_gguf_models()
        if models:
            print("📋 Available GGUF models:")
            for model in models:
                rel_path = model.relative_to(Path.cwd() / "models")
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  • {rel_path} ({size_mb:.1f} MB)")
        else:
            print("❌ No GGUF models found in models/ directory")
        return

    # Create progress tracker
    tracker = create_workbench_tracker()

    container = None
    try:
        # Step 1: Model Selection
        with tracker.step("Model Selection"):
            if args.model:
                model_path = Path(args.model)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            else:
                model_path = select_model_interactive()
                if not model_path:
                    status_info("No model selected. Exiting.")
                    sys.exit(0)

        # Extract model info
        model_name = model_path.stem
        repo_id = f"local/{model_name}"

        # Step 2: Benchmark Selection (if not specified via args)
        if not args.benchmark or args.benchmark == "standard":
            print("\n🎯 Select Benchmark Type:")
            print("1. Quick (2 iterations, simple questions)")
            print("2. Standard (3 iterations, medium complexity)")
            print("3. Performance (5 iterations, comprehensive)")
            print("4. Hard Questions (3 challenging questions, 3 iterations each)")
            print("5. Skip benchmarking")

            while True:
                try:
                    choice = input("\nSelect benchmark (1-5): ").strip()
                    if choice == "1":
                        selected_benchmark = "quick"
                        break
                    elif choice == "2":
                        selected_benchmark = "standard"
                        break
                    elif choice == "3":
                        selected_benchmark = "performance"
                        break
                    elif choice == "4":
                        selected_benchmark = "hard"
                        break
                    elif choice == "5":
                        selected_benchmark = "skip"
                        break
                    else:
                        print("Please enter a number between 1-5")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    sys.exit(0)
        else:
            selected_benchmark = args.benchmark

        if not args.no_serve:
            # Step 2: Docker Setup
            with tracker.step("Docker Setup"):
                container, docker_cmd = create_llama_container(
                    model_path=model_path,
                    port=args.port,
                    ctx_size=args.ctx_size,
                    version=args.version,
                )

            # Step 3: Container Start
            with tracker.step("Container Start"):
                if not container.start_container(docker_cmd):
                    raise RuntimeError("Failed to start container")

            # Step 4: Health Check
            with tracker.step("Health Check"):
                if not container.wait_for_ready(timeout=120):
                    raise RuntimeError("Model server did not become ready")

        # Step 5: Benchmarking
        if selected_benchmark != "skip":
            with tracker.step("Benchmarking"):
                benchmark_configs = {
                    "quick": [QUICK_BENCHMARK],
                    "standard": [STANDARD_BENCHMARK],
                    "performance": [PERFORMANCE_BENCHMARK],
                    "hard": HARD_QUESTIONS_BENCHMARK,
                }

                configs = benchmark_configs[selected_benchmark]

                # Set host/port for all configs
                for config in configs:
                    config.host = "localhost"
                    config.port = args.port

                runner = BenchmarkRunner()

                if selected_benchmark == "hard":
                    # Run multiple benchmarks for hard questions
                    results = runner.run_multiple_benchmarks(
                        repo_id=repo_id,
                        model_path=str(model_path),
                        engine="llama-server",
                        configs=configs,
                    )
                    benchmark_result = results  # List of results
                else:
                    # Single benchmark for other types
                    result = runner.run_benchmark(
                        repo_id=repo_id,
                        model_path=str(model_path),
                        engine="llama-server",
                        config=configs[0],
                    )
                    benchmark_result = result  # Single result

            # Step 6: Save Results
            with tracker.step("Save Results"):
                if isinstance(benchmark_result, list):
                    # Save multiple results
                    results_files = []
                    for result in benchmark_result:
                        results_file = runner.save_results(result)
                        results_files.append(results_file)
                else:
                    # Save single result
                    results_file = runner.save_results(benchmark_result)

        else:
            status_info("Skipping benchmark as requested")
            status_info(f"Model server running at: http://localhost:{args.port}")
            if not args.no_serve:
                input("\nPress Enter to stop the server...")
            benchmark_result = None

    except KeyboardInterrupt:
        status_info("Interrupted by user")
        benchmark_result = None

    except Exception as e:
        status_error(f"Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup first
        if container and container.is_running():
            container.stop_container()

        # Show workflow summary
        tracker.print_summary()

        # Show final benchmark results at the very end
        if "benchmark_result" in locals() and benchmark_result:
            print("\n")
            if isinstance(benchmark_result, list):
                # Print summary for each hard question result
                for i, result in enumerate(benchmark_result, 1):
                    print(f"\n📋 Hard Question {i} Results:")
                    runner.print_summary(result)
            else:
                # Single result
                runner.print_summary(benchmark_result)


if __name__ == "__main__":
    main()
