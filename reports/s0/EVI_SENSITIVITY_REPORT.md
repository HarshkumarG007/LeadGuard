# EVI Sensitivity Discovery Report

> **DISCLAIMER:** This artifact is an interpretation layer. It discovers fragility but does NOT redefine the frozen S1 acceptance criterion.

## Baseline Assumptions (S0 Simulation)
- Base inspection cost ($c_{base}$): $100.00
- Base info value ($v$): $5000.00
- Assumed inspections: 100
- Assumed hit rate: 10/100 (10.0%)

## Expected EVI
Expected Value of Information under base assumptions: **$40000.00**

## Break-Even Discovery
Break-even inspection cost ($c^*$): **$500.00**

## Normalized Robustness ($R_c$)
$R_c = (c^* - c_{base}) / c_{base} = 400.0\%$

**Deployment Policy Guidance**: VERY ROBUST

### Interpretation
- **FRAGILE (< 10%)**: The economic advantage evaporates easily.
- **SENSITIVE (10-25%)**: Minor operational delays can flip the utility.
- **ROBUST (25-50%)**: Structurally sound unless major assumptions are flawed.
- **VERY ROBUST (> 50%)**: The policy can withstand massive estimation errors.
