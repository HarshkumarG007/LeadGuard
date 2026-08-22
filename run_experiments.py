import json
import subprocess
import shutil
import numpy as np
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    feature_sets = ["intrinsic", "intrinsic_geo", "intrinsic_geo_process", "full"]
    
    # 1. Fixed Geographic Split
    geographic_results = {}
    print("\n" + "="*50)
    print("   PHASE C2: GEOGRAPHIC ABLATION")
    print("="*50)
    
    for fset in feature_sets:
        print(f"\n--- Geographic | Feature Set: {fset} ---")
        out_dir = f"artifacts/c2_geographic_{fset}"
        run_cmd([".venv/Scripts/python.exe", "src/leadguard/models/baseline.py", "--output-dir", f"{out_dir}/models/baseline", "--reports-dir", f"{out_dir}/reports", "--sample", "--split-mode", "geographic"])
        run_cmd([".venv/Scripts/python.exe", "src/leadguard/models/xgboost_model.py", "--output-dir", f"{out_dir}/models/xgboost", "--reports-dir", f"{out_dir}/reports", "--sample", "--split-mode", "geographic", "--feature-set", fset])
        
        with open(f"{out_dir}/models/xgboost/metrics.json") as f:
            xgb_metrics = json.load(f)
        with open(f"{out_dir}/reports/baseline_metrics.json") as f:
            base_metrics = json.load(f)
        
        geographic_results[fset] = {
            "xgb_pr_auc": xgb_metrics["pr_auc"],
            "heuristic_pr_auc": base_metrics["heuristic"]["pr_auc"]
        }

    # 2. Rolling Temporal Folds (C4)
    # Fold 1: Train < 2023, Cal=2023, Test=2024
    # Fold 2: Train < 2024, Cal=2024, Test=2025
    folds = [
        {"name": "fold_1_test_2024", "cutoff": "2023-01-01", "test_start": "2024-01-01", "test_end": "2025-01-01"},
        {"name": "fold_2_test_2025", "cutoff": "2024-01-01", "test_start": "2025-01-01", "test_end": "2026-01-01"},
    ]
    
    temporal_results = {fset: [] for fset in feature_sets}
    
    print("\n" + "="*50)
    print("   PHASE C4: ROLLING TEMPORAL ABLATION")
    print("="*50)
    
    for fold in folds:
        for fset in feature_sets:
            print(f"\n--- Temporal | Fold: {fold['name']} | Feature Set: {fset} ---")
            out_dir = f"artifacts/c4_{fold['name']}_{fset}"
            args = [
                "--sample", "--split-mode", "temporal",
                "--cutoff-date", fold["cutoff"],
                "--test-start-date", fold["test_start"],
                "--test-end-date", fold["test_end"],
                "--min-test-rows", "5" # Sample data is small
            ]
            run_cmd([".venv/Scripts/python.exe", "src/leadguard/models/baseline.py", "--output-dir", f"{out_dir}/models/baseline", "--reports-dir", f"{out_dir}/reports"] + args)
            run_cmd([".venv/Scripts/python.exe", "src/leadguard/models/xgboost_model.py", "--output-dir", f"{out_dir}/models/xgboost", "--reports-dir", f"{out_dir}/reports", "--feature-set", fset] + args)
            
            try:
                with open(f"{out_dir}/models/xgboost/metrics.json") as f:
                    xgb_metrics = json.load(f)
                with open(f"{out_dir}/reports/baseline_metrics.json") as f:
                    base_metrics = json.load(f)
                
                temporal_results[fset].append({
                    "fold": fold["name"],
                    "xgb_pr_auc": xgb_metrics["pr_auc"],
                    "heuristic_pr_auc": base_metrics["heuristic"]["pr_auc"]
                })
            except Exception as e:
                print(f"Error loading metrics for {fold['name']} {fset}: {e}")

    # 3. Print the Summary Table (C5)
    print("\n\n" + "="*60)
    print("FINAL ABLATION RESULTS")
    print("="*60)
    print(f"{'Feature Set':<25} | {'Geo PR-AUC':<12} | {'Temp PR-AUC (Mean)':<18} | {'Temp Folds'}")
    print("-" * 80)
    for fset in feature_sets:
        geo = geographic_results[fset]["xgb_pr_auc"]
        
        temp_fold_vals = [r["xgb_pr_auc"] for r in temporal_results[fset]]
        temp_mean = np.mean(temp_fold_vals) if temp_fold_vals else 0
        
        folds_str = ", ".join([f"{v:.4f}" for v in temp_fold_vals])
        print(f"{fset:<25} | {geo:.4f}       | {temp_mean:.4f}             | [{folds_str}]")
        
    print("\nHeuristic Baseline (Year Built only):")
    geo_h = geographic_results["full"]["heuristic_pr_auc"]
    temp_h_vals = [r["heuristic_pr_auc"] for r in temporal_results["full"]]
    temp_h_mean = np.mean(temp_h_vals) if temp_h_vals else 0
    folds_h_str = ", ".join([f"{v:.4f}" for v in temp_h_vals])
    print(f"{'heuristic (year_built)':<25} | {geo_h:.4f}       | {temp_h_mean:.4f}             | [{folds_h_str}]")

    # Compute Delta
    print("\nDelta (XGBoost FULL - Heuristic):")
    geo_delta = geographic_results["full"]["xgb_pr_auc"] - geo_h
    temp_delta_vals = [xgb - h for xgb, h in zip(
        [r["xgb_pr_auc"] for r in temporal_results["full"]],
        [r["heuristic_pr_auc"] for r in temporal_results["full"]]
    )]
    temp_delta_mean = np.mean(temp_delta_vals) if temp_delta_vals else 0
    folds_delta_str = ", ".join([f"{v:+.4f}" for v in temp_delta_vals])
    print(f"{'Delta':<25} | {geo_delta:+.4f}       | {temp_delta_mean:+.4f}             | [{folds_delta_str}]")

if __name__ == "__main__":
    main()
