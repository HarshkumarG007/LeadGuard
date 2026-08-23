import sys
from pathlib import Path

def calculate_evi(c_base, v, n_inspected, n_valid_outcomes):
    """
    EVI = InfoValue * ValidOutcomes - Cost * Inspected
    """
    return v * n_valid_outcomes - c_base * n_inspected

def calculate_break_even(v, n_inspected, n_valid_outcomes):
    """
    c* = (v * n_valid_outcomes) / n_inspected
    """
    if n_inspected == 0:
        return float('inf')
    return (v * n_valid_outcomes) / n_inspected

def evaluate_robustness(c_star, c_base):
    if c_base == 0:
        return float('inf'), "VERY ROBUST"
        
    r_c = (c_star - c_base) / c_base
    
    if r_c < 0.10:
        status = "FRAGILE"
    elif r_c < 0.25:
        status = "SENSITIVE"
    elif r_c < 0.50:
        status = "ROBUST"
    else:
        status = "VERY ROBUST"
        
    return r_c, status

def run_sensitivity():
    print("=== ECONOMIC SENSITIVITY DISCOVERY ===")
    # These are dummy baseline assumptions for the purpose of S0 discovery
    # In S1, the real parameters are frozen in the contract.
    c_base = 100.0  # Assumed cost per inspection
    v = 5000.0      # Assumed value of finding lead
    n_inspected = 100
    n_valid_outcomes = 10 # 10% hit rate
    
    base_evi = calculate_evi(c_base, v, n_inspected, n_valid_outcomes)
    c_star = calculate_break_even(v, n_inspected, n_valid_outcomes)
    
    r_c, status = evaluate_robustness(c_star, c_base)
    
    report = f"""# EVI Sensitivity Discovery Report

> **DISCLAIMER:** This artifact is an interpretation layer. It discovers fragility but does NOT redefine the frozen S1 acceptance criterion.

## Baseline Assumptions (S0 Simulation)
- Base inspection cost ($c_{{base}}$): ${c_base:.2f}
- Base info value ($v$): ${v:.2f}
- Assumed inspections: {n_inspected}
- Assumed hit rate: {n_valid_outcomes}/{n_inspected} ({n_valid_outcomes/n_inspected*100:.1f}%)

## Expected EVI
Expected Value of Information under base assumptions: **${base_evi:.2f}**

## Break-Even Discovery
Break-even inspection cost ($c^*$): **${c_star:.2f}**

## Normalized Robustness ($R_c$)
$R_c = (c^* - c_{{base}}) / c_{{base}} = {r_c*100:.1f}\\%$

**Deployment Policy Guidance**: {status}

### Interpretation
- **FRAGILE (< 10%)**: The economic advantage evaporates easily.
- **SENSITIVE (10-25%)**: Minor operational delays can flip the utility.
- **ROBUST (25-50%)**: Structurally sound unless major assumptions are flawed.
- **VERY ROBUST (> 50%)**: The policy can withstand massive estimation errors.
"""

    report_path = Path("reports/s0/EVI_SENSITIVITY_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(report)
    print(f"Sensitivity artifact saved to {report_path}")

if __name__ == "__main__":
    run_sensitivity()
