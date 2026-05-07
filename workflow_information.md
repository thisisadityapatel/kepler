The big idea

  Kepler answers one question: "How fast does this model run on my Mac, across the engines I could use?" It does that by running every compatible engine
  sequentially on the same model, in clean GPU-isolated subprocesses, then printing a comparison table. One format at a time — you tell it --format gguf or
  --format mlx and it picks engines from there.

  What happens when you type kepler benchmark Qwen/Qwen2.5-0.5B-Instruct --format gguf --mode quick

  1. CLI parses args (src/kepler/cli/app.py:34). Validates --format is set, picks a preset (quick → 2 iterations of 50 tokens), figures out which engines
  support GGUF (engines_for_format(GGUF) → [llamacpp, ollama, ik_llama]).
  2. Resolver runs once (src/kepler/models/resolver.py:23). For each candidate engine it asks: is this engine actually installed (probe in
  engines/registry.py:36)? If not, mark it unavailable and move on. If installed, what artifact does it need? For GGUF, the resolver checks the original HF
  repo, then falls back to <owner>/<name>-GGUF, picks the best quant via pick_gguf_file (Q5_K_M > Q4_K_M > Q8_0 > …, case-insensitive), and downloads it once
   via hf_hub_download. Returns a list of ResolvedEngine(engine_name, spec, status) — one per requested engine.
  3. Orchestrator drives the comparison (src/kepler/benchmark/orchestrator.py:36). For each ResolvedEngine with status="ready":
    - Spawn a fresh subprocess via multiprocessing.get_context("spawn") (benchmark/lifecycle.py:22).
    - Inside the child: import only that engine's module, call engine.load(model), run a warmup iteration (discarded), then preset.iterations timed
  iterations. Each iteration produces an EngineResult with timing fields.
    - The child sends {"iterations": [...], "host": ..., "port": ...} back to the parent over a Pipe and exits. Process exit is what releases the GPU — no
  gc.collect(), no manual cache clears.
    - Parent collects results, computes medians (vec_median matches the legacy C++ exactly — sort, take v[v.size()/2]), builds a BenchmarkResult, writes one
  JSON to perf/.
    - Sleeps --cooldown seconds (default 8) so Apple Silicon can shed thermal load before the next engine.
  4. Comparison report (src/kepler/benchmark/comparison.py:34). Aggregates all per-engine summaries into one ComparisonReport. Picks the winner by
  median_tok_per_s, generates a one-line narrative ("llamacpp leads at 83.1 tok/s on Apple M4"), saves a comparison_*.json, and renders a Rich table with the
   best value per column highlighted (ui/report.py).

  The subsystems

  - engines/ — Each engine is a class implementing InferenceEngine (load, infer, unload). They're not asked to clean up perfectly because they don't have to
  — the subprocess wrapping kills the whole interpreter when done.
  - models/ — ModelSpec is the value object an engine needs to load (local_path, gguf_filename, or ollama_tag). resolver.py is the only place that knows
  about HF Hub. downloader.py is just thin wrappers around huggingface_hub.
  - benchmark/ — lifecycle.py is the subprocess-isolation primitive. runner.py is "warmup + N iterations." orchestrator.py is the sequencer. results.py
  matches the legacy JSON schema field-for-field. comparison.py is the new layer the C++ version didn't have.
  - ui/ — Rich console helpers. Pure presentation, no logic.
  - config/loader.py — Reads config/models.yaml (preserved verbatim from the C++ version) into typed dataclasses.

  The piece that took the most thought: subprocess isolation

  Two engines running on the same Metal device at the same time would corrupt the benchmark — they'd contend for compute and memory bandwidth. So the
  orchestrator runs them strictly sequentially. But even sequentially, in-process MLX/llama.cpp can leave GPU caches and Python references hanging around, so
   engine #2 starts in a polluted environment.

  The fix is brutal and simple: each engine runs in a fresh spawn-context subprocess (lifecycle.run_in_subprocess). The child does the work, sends results
  back over a Pipe, exits. When the OS reaps that process, every byte it allocated — Python objects, MLX/Metal buffers, the lot — is gone. The next engine
  gets a clean room.

  verify_released (parent side) confirms the child PID is reaped, no descendants survive, and any bound port is closed before the next engine starts.

  The data flow

  CLI args → preset + engine list
          ↓
  Resolver → [ResolvedEngine(llamacpp, spec, ready),
              ResolvedEngine(ollama, spec, ready),
              ResolvedEngine(ik_llama, None, unavailable)]
          ↓
  Orchestrator (per ready engine):
      fork → child runs warmup + N iters → returns EngineResult[]
      parent builds BenchmarkResult → writes perf/*.json
      cooldown
          ↓
  Comparison.build_report → perf/comparison_*.json + Rich table