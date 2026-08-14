# Test made with AI to speed up validation, same style as test_agent.py:
# plain asserts + prints, no pytest, runs fully offline (no GPU required -
# the no-GPU fallback path is exercised directly on machines without one).
import json
import os
import shutil
import tempfile
import time

from llmadapt.benchmark import BenchmarkResult, HardwareBenchmark

cache_dir = tempfile.mkdtemp(prefix="llmadapt_bench_test_")

try:
    # Test 1: run() returns a fully populated BenchmarkResult
    result = HardwareBenchmark.run(cpu_time_budget_s=0.2, cache_dir=cache_dir)
    assert isinstance(result, BenchmarkResult)
    assert result.cpu_gflops > 0
    assert result.cpu_gflops_measured_with in ("numpy", "pure_python")
    assert result.cpu_cores >= 1
    assert result.system_ram_gb > 0
    assert result.gpu_vram_gb >= 0
    assert result.pcie_bandwidth_gbps >= 0
    assert result.gpu_tflops >= 0
    assert len(result.fingerprint) == 16
    print("PASS: run() returns a fully populated BenchmarkResult")

    # Test 2: to_dict() is JSON-serializable end to end
    as_json = json.dumps(result.to_dict())
    assert json.loads(as_json)["fingerprint"] == result.fingerprint
    print("PASS: BenchmarkResult.to_dict() is JSON-serializable")

    # Test 3: cache file is written to the given cache_dir (not the real home dir)
    cache_file = os.path.join(cache_dir, "benchmark_cache.json")
    assert os.path.exists(cache_file), "expected the benchmark to write its cache file"
    print("PASS: benchmark cache is written to the given cache_dir")

    # Test 4: a second call within the TTL hits the cache instead of re-measuring
    start = time.perf_counter()
    cached_result = HardwareBenchmark.run(cpu_time_budget_s=0.2, cache_dir=cache_dir)
    elapsed = time.perf_counter() - start
    assert cached_result.fingerprint == result.fingerprint
    assert cached_result.timestamp == result.timestamp, (
        "a cache hit should return the exact cached result, not re-measure"
    )
    assert elapsed < 0.05, f"cache hit took {elapsed:.3f}s - expected a near-instant read, not a re-benchmark"
    print("PASS: repeat run() within the TTL hits the cache instead of re-measuring")

    # Test 5: force=True bypasses the cache and re-measures
    forced_result = HardwareBenchmark.run(force=True, cpu_time_budget_s=0.2, cache_dir=cache_dir)
    assert forced_result.timestamp > result.timestamp, "force=True should re-measure, not reuse the cached timestamp"
    print("PASS: force=True bypasses the cache")

    # Test 6: an expired TTL is treated as a cache miss
    stale_result = HardwareBenchmark.run(cpu_time_budget_s=0.2, cache_ttl_s=0, cache_dir=cache_dir)
    assert stale_result.timestamp > forced_result.timestamp, "cache_ttl_s=0 should force a fresh measurement"
    print("PASS: expired cache_ttl_s is treated as a cache miss")

    # Test 7: fingerprint is stable across repeated calls on the same machine
    fp1 = HardwareBenchmark.run(cpu_time_budget_s=0.1, cache_dir=cache_dir).fingerprint
    fp2 = HardwareBenchmark.run(cpu_time_budget_s=0.1, cache_dir=cache_dir).fingerprint
    assert fp1 == fp2
    print("PASS: fingerprint is stable across repeated runs on the same machine")

    # Test 8: no-GPU machines get honest zeros instead of a guessed fallback number
    if result.gpu_name == "none":
        assert result.gpu_tflops == 0.0
        assert result.pcie_bandwidth_gbps == 0.0
        assert result.pcie_bandwidth_measured is False
        assert result.gpu_tflops_measured is False
        print("PASS: no GPU detected -> gpu_tflops/pcie_bandwidth are honest zeros, not a guess")
    else:
        print(f"SKIP: GPU detected ({result.gpu_name}) - no-GPU zero-fallback path not exercised on this machine")

    print("\nAll checks passed.")
finally:
    shutil.rmtree(cache_dir, ignore_errors=True)
