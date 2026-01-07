"""Comprehensive benchmarking module for LLM models."""

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
from common import CONFIG_PATH, PERF_ROOT
from run_bench import bench_once_llama


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
            processor=platform.processor() or platform.machine(),
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

    def __init__(self):
        self.results_dir = PERF_ROOT
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_multiple_benchmarks(
        self,
        repo_id: str,
        model_path: str,
        engine: str = "llama-server",
        configs: List[BenchmarkConfig] = [],
    ) -> List[ModelBenchmarkResult]:
        """Run multiple benchmark configurations."""
        if not configs:
            return []

        results = []
        for config in configs:
            try:
                result = self.run_benchmark(repo_id, model_path, config, engine)
                results.append(result)
            except Exception:
                continue
        return results

    def run_benchmark(
        self,
        repo_id: str,
        model_path: str,
        config: BenchmarkConfig,
        engine: str = "llama-server",  # Default to llama-server
    ) -> ModelBenchmarkResult:
        """Run complete benchmark for a model."""

        iterations = []

        for _ in range(config.iterations):
            try:
                result = bench_once_llama(
                    prompt=config.prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    host=config.host,
                    port=config.port,
                )

                iteration = BenchmarkIteration(
                    wall_s=result["wall_s"],
                    output_text=result["output_text"],
                    prompt_tokens=result["prompt_tokens"],
                    generated_tokens=result["generated_tokens"],
                    tok_per_s=result["tok_per_s"],
                    generation_tok_per_s=result["generation_tok_per_s"],
                    ttft_ms=result["ttft_ms"],
                    prefill_ms=result["ttft_ms"],
                    generation_ms=result["generation_ms"],
                )

                iterations.append(iteration)
            except Exception:
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
            summary=summary,
        )

        return benchmark_result

    def _calculate_summary(
        self, iterations: List[BenchmarkIteration]
    ) -> BenchmarkSummary:
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
            median_generation_tok_per_s=gen_tok_per_s_values[median_idx],
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

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return file_path

    def print_summary(self, result: ModelBenchmarkResult):
        """Print benchmark summary to console."""
        print("\n" + "=" * 60)
        print(f"📊 BENCHMARK RESULTS: {result.repo_id}")
        print("=" * 60)
        print(f"Model: {result.model_ref}")
        print(f"Engine: {result.engine}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Iterations: {len(result.iterations)}")

        print("\nPerformance Metrics (Median):")
        print(f"  Tokens/sec:     {result.summary.median_tok_per_s:.2f}")
        print(f"  TTFT (ms):      {result.summary.median_ttft_ms:.1f}")
        print(f"  Wall Time:      {result.summary.median_wall_s:.2f}s")
        print(f"  Gen Tokens/sec: {result.summary.median_generation_tok_per_s:.2f}")

        print("\nSystem Info:")
        print(f"  Platform:       {result.system_info.platform}")
        print(f"  Architecture:   {result.system_info.architecture}")
        print(f"  Processor:      {result.system_info.processor}")
        print("=" * 60)


def load_config():
    """Load full config from YAML file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def load_benchmark_questions():
    """Load benchmark questions from config file."""
    config = load_config()
    return config.get("benchmark_questions", {}).get("hard_questions", [])


def load_benchmark_config(benchmark_name: str) -> BenchmarkConfig:
    """Load a specific benchmark configuration from config file."""
    config = load_config()
    benchmark_data = config.get(benchmark_name, {})

    if not benchmark_data:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in config")

    return BenchmarkConfig(
        prompt_set=benchmark_data.get("prompt_set", benchmark_name),
        prompt=benchmark_data["prompt"],
        max_tokens=benchmark_data.get("max_tokens", 100),
        temperature=benchmark_data.get("temperature", 0.7),
        iterations=benchmark_data.get("iterations", 3),
        host=benchmark_data.get("host", "localhost"),
        port=benchmark_data.get("port", 8080),
    )


def create_hard_questions_benchmark() -> List[BenchmarkConfig]:
    """Create benchmark configs from hard questions in config file."""
    questions = load_benchmark_questions()
    benchmarks = []

    for i, q in enumerate(questions, 1):
        config = BenchmarkConfig(
            prompt_set=f"hard_question_{i}",
            prompt=q["prompt"],
            max_tokens=q["max_tokens"],
            temperature=0.7,
            iterations=3,
        )
        benchmarks.append(config)

    return benchmarks


# Load benchmark configurations from config file
QUICK_BENCHMARK = load_benchmark_config("quick_benchmark")
STANDARD_BENCHMARK = load_benchmark_config("standard_benchmark")
PERFORMANCE_BENCHMARK = load_benchmark_config("performance_benchmark")
HARD_QUESTIONS_BENCHMARK = create_hard_questions_benchmark()
