"""Comprehensive benchmarking module for LLM models."""

import json
import time
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from run_bench import bench_once_llama
from common import PERF_ROOT


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    prompt_set: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    iterations: int = 3
    host: str = "localhost"
    port: int = 8080


@dataclass
class BenchmarkIteration:
    """Single benchmark iteration result."""
    wall_s: float
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    tok_per_s: float
    generation_tok_per_s: float
    ttft_ms: float
    prefill_ms: float
    generation_ms: float


@dataclass
class SystemInfo:
    """System information for benchmarks."""
    platform: str
    architecture: str
    processor: str
    
    @classmethod
    def get_current_system(cls):
        return cls(
            platform=platform.system(),
            architecture=platform.machine(),
            processor=platform.processor() or platform.machine()
        )


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark results."""
    median_wall_s: float
    median_tok_per_s: float
    median_ttft_ms: float
    median_generation_tok_per_s: float


@dataclass
class ModelBenchmarkResult:
    """Complete benchmark result for a model."""
    timestamp: str
    repo_id: str
    model_ref: str
    engine: str
    system_info: SystemInfo
    config: BenchmarkConfig
    iterations: List[BenchmarkIteration]
    summary: BenchmarkSummary
    mode: str = "text-only"
    environment: str = "stable"


class BenchmarkRunner:
    """Runs comprehensive benchmarks on LLM models."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or PERF_ROOT
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_benchmark(self, repo_id: str, model_path: str, engine: str = "llama-server",
                     config: BenchmarkConfig = None) -> ModelBenchmarkResult:
        """Run complete benchmark for a model."""
        if config is None:
            config = BenchmarkConfig(
                prompt_set="default",
                prompt="Explain quantum computing to a 10-year-old.",
                max_tokens=100,
                temperature=0.7,
                iterations=3
            )
        
        print(f"-- Starting benchmark for {repo_id}")
        print(f"-- Engine: {engine}")
        print(f"-- Prompt: {config.prompt[:50]}...")
        print(f"-- Iterations: {config.iterations}")
        
        iterations = []
        
        # Run benchmark iterations
        for i in range(config.iterations):
            print(f"-- Run {i + 1}/{config.iterations}", end=" ")
            
            try:
                result = bench_once_llama(
                    prompt=config.prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    host=config.host,
                    port=config.port
                )
                
                iteration = BenchmarkIteration(
                    wall_s=result["wall_s"],
                    output_text=result["output_text"],
                    prompt_tokens=result["prompt_tokens"],
                    generated_tokens=result["generated_tokens"],
                    tok_per_s=result["tok_per_s"],
                    generation_tok_per_s=result["generation_tok_per_s"],
                    ttft_ms=result["ttft_ms"],
                    prefill_ms=result["ttft_ms"],  # Use TTFT as prefill for now
                    generation_ms=result["generation_ms"]
                )
                
                iterations.append(iteration)
                print(f"-- {result['tok_per_s']:.1f} tok/s")
                
            except Exception as e:
                print(f"-- Failed: {e}")
                continue
        
        if not iterations:
            raise RuntimeError("All benchmark iterations failed")
        
        # Calculate summary statistics
        summary = self._calculate_summary(iterations)
        
        # Create complete result
        benchmark_result = ModelBenchmarkResult(
            timestamp=datetime.now().isoformat(),
            repo_id=repo_id,
            model_ref=model_path,
            engine=engine,
            system_info=SystemInfo.get_current_system(),
            config=config,
            iterations=iterations,
            summary=summary
        )
        
        return benchmark_result
    
    def _calculate_summary(self, iterations: List[BenchmarkIteration]) -> BenchmarkSummary:
        """Calculate summary statistics from iterations."""
        wall_times = sorted([i.wall_s for i in iterations])
        tok_per_s_values = sorted([i.tok_per_s for i in iterations])
        ttft_values = sorted([i.ttft_ms for i in iterations])
        gen_tok_per_s_values = sorted([i.generation_tok_per_s for i in iterations])
        
        n = len(iterations)
        median_idx = n // 2
        
        return BenchmarkSummary(
            median_wall_s=wall_times[median_idx],
            median_tok_per_s=tok_per_s_values[median_idx],
            median_ttft_ms=ttft_values[median_idx],
            median_generation_tok_per_s=gen_tok_per_s_values[median_idx]
        )
    
    def save_results(self, result: ModelBenchmarkResult) -> Path:
        """Save benchmark results to JSON file with proper naming."""
        # Create filename: perf/2025-12-09_repo_name_engine_mode.json
        timestamp = datetime.fromisoformat(result.timestamp).strftime("%Y-%m-%d")
        
        # Clean repo_id for filename
        repo_clean = result.repo_id.replace("/", "_").replace(".", "_")
        engine_clean = result.engine.replace("-", "_")
        
        filename = f"{timestamp}_{repo_clean}_{engine_clean}_{result.mode}.json"
        file_path = self.results_dir / filename
        
        # Convert to dict for JSON serialization
        data = asdict(result)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"-- Results saved to: {file_path}")
        return file_path
    
    def print_summary(self, result: ModelBenchmarkResult):
        """Print benchmark summary to console."""
        print(f"\n" + "="*60)
        print(f"📊 BENCHMARK RESULTS: {result.repo_id}")
        print("="*60)
        print(f"Model: {result.model_ref}")
        print(f"Engine: {result.engine}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Iterations: {len(result.iterations)}")
        
        print(f"\n🚀 Performance Metrics (Median):")
        print(f"   Tokens/sec:     {result.summary.median_tok_per_s:.2f}")
        print(f"   TTFT (ms):      {result.summary.median_ttft_ms:.1f}")
        print(f"   Wall Time:      {result.summary.median_wall_s:.2f}s")
        print(f"   Gen Tokens/sec: {result.summary.median_generation_tok_per_s:.2f}")
        
        print(f"\n💻 System Info:")
        print(f"   Platform:       {result.system_info.platform}")
        print(f"   Architecture:   {result.system_info.architecture}")
        print(f"   Processor:      {result.system_info.processor}")
        
        print("="*60)


# Default benchmark configurations
QUICK_BENCHMARK = BenchmarkConfig(
    prompt_set="quick",
    prompt="Write a short Python function to find the maximum number in a list.",
    max_tokens=50,
    temperature=0.7,
    iterations=2
)

STANDARD_BENCHMARK = BenchmarkConfig(
    prompt_set="standard", 
    prompt="Explain quantum computing in simple terms and provide a real-world analogy.",
    max_tokens=100,
    temperature=0.7,
    iterations=3
)

PERFORMANCE_BENCHMARK = BenchmarkConfig(
    prompt_set="performance",
    prompt="Write a comprehensive guide on machine learning algorithms, including examples and use cases for each type.",
    max_tokens=200,
    temperature=0.7,
    iterations=5
)