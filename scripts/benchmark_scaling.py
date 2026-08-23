"""Benchmark API scaling with 3-layer decomposition and concurrency sweep."""

import time
import psutil
import os
import requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import uuid
import concurrent.futures
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from api.main import _predict_single, get_state, _get_properties
from api.model_loader import ModelState

def create_synthetic_snapshots(n: int, decision_path: str, explanation_path: str, meta_path: str):
    np.random.seed(42)
    decision_df = pd.DataFrame({
        "property_id": [f"prop_{i}" for i in range(n)],
        "census_tract": [f"tract_{i % 100}" for i in range(n)],
        "ward": [f"ward_{i % 10}" for i in range(n)],
        "p_lead_calibrated": np.random.uniform(0, 1, n).astype(np.float32),
        "uncertainty_score": np.random.uniform(0, 0.5, n).astype(np.float32),
        "equity_boost": np.random.uniform(0, 0.2, n).astype(np.float32),
        "evi": np.random.uniform(0, 5000, n).astype(np.float32),
    })
    Path(decision_path).parent.mkdir(parents=True, exist_ok=True)
    decision_df.to_parquet(decision_path)
    
    explanation_df = pd.DataFrame({
        "property_id": decision_df["property_id"],
        "shap_features_json": ['[{"feature": "age", "contribution": 0.5}]'] * n
    })
    explanation_df.to_parquet(explanation_path)
    
    from datetime import UTC, datetime, timedelta
    now = datetime.now(UTC)
    meta = {
        "snapshot_id": "snap-bench",
        "generated_at": now.isoformat(),
        "valid_until": (now + timedelta(hours=1)).isoformat(),
        "model_version": "v1.0",
        "feature_version": "v1.0",
        "policy_version": "v1.0",
        "policy_parameters_hash": "abc",
        "data_cutoff": now.isoformat(),
        "code_version": "bench"
    }
    Path(meta_path).write_text(json.dumps(meta))
    return decision_df["property_id"].tolist()

def benchmark_three_layers(prop_ids, n):
    decision_path = "data/processed/decision_snapshot.parquet"
    expl_path = "data/processed/explanation_snapshot.parquet"
    
    print("\n--- A. Direct Snapshot Lookup (Storage/Indexing) ---")
    decision_df = pd.read_parquet(decision_path)
    
    lats = []
    for _ in range(100):
        pid = prop_ids[np.random.randint(0, n)]
        t0 = time.time()
        # 1. Decision lookup
        row_mask = decision_df["property_id"] == pid
        row = decision_df[row_mask].iloc[0]
        # 2. Explanation lookup (PyArrow out-of-core)
        table = pq.read_table(expl_path, filters=[("property_id", "==", pid)])
        json_str = table.column("shap_features_json")[0].as_py()
        top_f = json.loads(json_str)
        lats.append((time.time() - t0) * 1000)
    print(f"Layer A Latency (N=1M): p50={np.percentile(lats, 50):.2f}ms, p99={np.percentile(lats, 99):.2f}ms")

    print("\n--- B. Application Function (_predict_single) ---")
    state = ModelState()
    state.model = object()
    lats = []
    for _ in range(100):
        pid = prop_ids[np.random.randint(0, n)]
        row_mask = decision_df["property_id"] == pid
        row = decision_df[row_mask].iloc[0]
        
        t0 = time.time()
        res = _predict_single(row, state, features_df=decision_df)
        lats.append((time.time() - t0) * 1000)
    print(f"Layer B Latency (N=1M): p50={np.percentile(lats, 50):.2f}ms, p99={np.percentile(lats, 99):.2f}ms")
    
    return np.percentile(lats, 50)

def benchmark_http_sweep(prop_ids, n):
    print("\n--- C. HTTP Endpoint Concurrency Sweep ---")
    process = subprocess.Popen([
        ".venv/Scripts/uvicorn.exe", "api.main:app", "--port", "8000"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        time.sleep(5) # Wait for startup
        
        # Cold start
        t0 = time.time()
        requests.get(f"http://127.0.0.1:8000/v1/properties/{prop_ids[0]}/prediction")
        print(f"Cold start (HTTP): {(time.time() - t0)*1000:.2f} ms")
        
        results = []
        for c in [1, 2, 4, 8, 16, 32, 64, 100]:
            req_per_worker = max(1, 1000 // c)
            total_reqs = c * req_per_worker
            
            def worker():
                local_lats = []
                session = requests.Session()
                for _ in range(req_per_worker):
                    pid = prop_ids[np.random.randint(0, n)]
                    t = time.time()
                    resp = session.get(f"http://127.0.0.1:8000/v1/properties/{pid}/prediction")
                    if resp.status_code == 200:
                        local_lats.append((time.time() - t) * 1000)
                return local_lats
                
            t_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
                futures = [executor.submit(worker) for _ in range(c)]
                latencies = []
                for f in concurrent.futures.as_completed(futures):
                    latencies.extend(f.result())
            t_end = time.time()
            
            throughput = len(latencies) / (t_end - t_start)
            if latencies:
                p50 = np.percentile(latencies, 50)
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)
            else:
                p50 = p95 = p99 = 0.0
                
            p = psutil.Process(process.pid)
            rss_mb = p.memory_info().rss / (1024 * 1024)
            cpu = p.cpu_percent(interval=0.1)
            
            results.append({
                "concurrency": c, "throughput": throughput,
                "p50": p50, "p95": p95, "p99": p99, "rss": rss_mb, "cpu": cpu
            })
            
            print(f"C={c:<3} | Tput: {throughput:6.1f} r/s | p50: {p50:6.1f} ms | p95: {p95:6.1f} ms | CPU: {cpu:5.1f}% | RSS: {rss_mb:4.1f} MB")
            
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    n = 1_000_000
    decision_path = "data/processed/decision_snapshot.parquet"
    if not Path(decision_path).exists():
        prop_ids = create_synthetic_snapshots(
            n,
            decision_path,
            "data/processed/explanation_snapshot.parquet",
            "data/processed/decision_snapshot_meta.json"
        )
    else:
        df = pd.read_parquet(decision_path)
        prop_ids = df["property_id"].tolist()
        
    benchmark_three_layers(prop_ids, n)
    benchmark_http_sweep(prop_ids, n)
