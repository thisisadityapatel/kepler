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

from model_selector import select_model_interactive, find_gguf_models
from docker_manager import create_llama_container, DockerContainer
from benchmark import BenchmarkRunner, QUICK_BENCHMARK, STANDARD_BENCHMARK, PERFORMANCE_BENCHMARK
from common import BACKEND_REGISTRY


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
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to GGUF model file (skips interactive selection)"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        choices=["quick", "standard", "performance", "skip"],
        default="standard",
        help="Benchmark type to run (default: standard)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port for model server (default: 8080)"
    )
    
    parser.add_argument(
        "--ctx-size", "-c",
        type=int,
        default=4096,
        help="Context size for model (default: 4096)"
    )
    
    parser.add_argument(
        "--version", "-v",
        type=str,
        default="b7531",
        help="llama.cpp version to use (default: b7531)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Skip model serving (for testing)"
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
    
    print("🚀 Welcome to Kepler LLM Workbench!")
    print("   macOS-optimized LLM serving and benchmarking")
    
    # Step 1: Model Selection
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        print(f"📁 Using specified model: {model_path.name}")
    else:
        model_path = select_model_interactive()
        if not model_path:
            print("👋 No model selected. Exiting.")
            sys.exit(0)
    
    # Extract model info
    model_name = model_path.stem
    repo_id = f"local/{model_name}"  # Since we're using local files
    
    print(f"\n-- Selected model: {model_name}")
    print(f"-- Path: {model_path}")
    
    container = None
    try:
        if not args.no_serve:
            # Step 2: Container Setup and Model Serving
            print(f"\n-- Setting up Docker container...")
            container, docker_cmd = create_llama_container(
                model_path=model_path,
                port=args.port,
                ctx_size=args.ctx_size,
                version=args.version
            )
            
            # Start the container
            if not container.start_container(docker_cmd):
                print("-- Failed to start container")
                sys.exit(1)
            
            # Wait for model to be ready
            if not container.wait_for_ready(timeout=120):
                print("-- Model server did not become ready")
                sys.exit(1)
            
            print(f"-- Model server is running on http://localhost:{args.port}")
        
        # Step 3: Benchmarking
        if args.benchmark != "skip":
            print(f"\n-- Starting {args.benchmark} benchmark...")
            
            # Select benchmark configuration
            benchmark_configs = {
                "quick": QUICK_BENCHMARK,
                "standard": STANDARD_BENCHMARK, 
                "performance": PERFORMANCE_BENCHMARK
            }
            
            config = benchmark_configs[args.benchmark]
            config.host = "localhost"
            config.port = args.port
            
            # Run benchmark
            runner = BenchmarkRunner()
            result = runner.run_benchmark(
                repo_id=repo_id,
                model_path=str(model_path),
                engine="llama-server",
                config=config
            )
            
            # Save and display results
            results_file = runner.save_results(result)
            runner.print_summary(result)
            
            print(f"\n-- Benchmark complete!")
            print(f"-- Results saved to: {results_file}")
        
        else:
            print(f"\n-- Skipping benchmark as requested")
            print(f"-- Model server running at: http://localhost:{args.port}")
            print("-- Test with: curl -X POST http://localhost:8080/completion -H 'Content-Type: application/json' -d '{\"prompt\":\"Hello\",\"n_predict\":10}'")
            
            if not args.no_serve:
                input("\n-- Press Enter to stop the server...")
    
    except KeyboardInterrupt:
        print("\n-- Interrupted by user")
    
    except Exception as e:
        print(f"\n-- Error: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        if container and container.is_running():
            print("\n-- Cleaning up...")
            container.stop_container()
    
    print("\n-- Thanks for using Kepler LLM Workbench!")


if __name__ == "__main__":
    main()