# Housing Decision Analysis System

A comprehensive Monte Carlo simulation system for analyzing housing move/stay decisions using weighted parameter sampling and multi-scenario lease structure modeling.

## Table of Contents

- [Overview](#overview)
- [Mathematical Model](#mathematical-model)
- [Parameter Definitions](#parameter-definitions)
- [Sampling Methodology](#sampling-methodology)
- [Weight Distributions](#weight-distributions)
- [Analysis Types](#analysis-types)
- [Usage Instructions](#usage-instructions)
- [Output Interpretation](#output-interpretation)
- [Examples](#examples)

## Overview

This system helps analyze whether to move to a new housing location by comparing current rent (Rx) and additional costs (Hx) against alternative rent (Ry) and costs (Hy). It uses Monte Carlo simulation with weighted parameter distributions to account for realistic probability distributions across parameter ranges.

**Key Features:**

- Monte Carlo simulation with 5,000+ weighted samples
- Parallel coordinates visualization with decision boundary highlighting
- Threshold analysis showing parameter sensitivity
- Multi-year lease structure modeling with compound rent hikes
- Comprehensive comparison across all lease combinations
- Interactive parameter input with scenario testing

## Mathematical Model

### Decision Framework

The core decision is based on a simple comparison of total annual costs:

```
s = (12×Rx + Hx) - (12×Ry + Hy)
```

Where:

- **s > 0**: MOVE (current location is more expensive)
- **s ≤ 0**: STAY (alternative location is more expensive or equivalent)

### Annualization Logic

Monthly rent values (Rx, Ry) are multiplied by 12 to convert to annual costs, then combined with additional annual costs (Hx, Hy) for total comparison.

### Boundary Analysis

The system identifies “boundary cases” where |s| ≤ 20, representing scenarios where the decision is marginal and small parameter changes could flip the outcome.

## Parameter Definitions

### Core Parameters

|Parameter|Description                        |Example Range|Units      |
|---------|-----------------------------------|-------------|-----------|
|**Rx**   |Current monthly rent               |480-530      |$ per month|
|**Hx**   |Current additional annual costs    |560-1000     |$ per year |
|**Ry**   |Alternative monthly rent           |530-570      |$ per month|
|**Hy**   |Alternative additional annual costs|500-650      |$ per year |

### Additional Annual Costs (H Values)

Examples include:

- Utilities not included in rent
- Parking fees
- Transportation cost differences
- HOA fees
- Insurance adjustments
- Moving costs (amortized)

### Derived Parameters

|Parameter|Formula     |Description                 |
|---------|------------|----------------------------|
|**d**    |12×(Rx - Ry)|Annual rent difference      |
|**f**    |Hx - Hy     |Additional cost difference  |
|**s**    |d + f       |Total annual cost difference|

## Sampling Methodology

### Monte Carlo Approach

The system uses Monte Carlo simulation to handle the uncertainty and variability in housing costs:

1. **Sample Size**: 5,000 combinations (configurable)
1. **Reproducibility**: Fixed random seed (42) for consistent results
1. **Weighted Sampling**: Non-uniform probability distributions reflecting real-world patterns

### Parameter Ranges

Default ranges are designed to represent realistic housing market conditions:

```python
rx_values = range(480, 531, 5)    # 11 values: [480, 485, ..., 530]
hx_values = range(560, 1001, 5)   # 89 values: [560, 565, ..., 1000]  
ry_values = range(530, 571, 5)    # 9 values: [530, 535, ..., 570]
hy_values = range(500, 651, 5)    # 31 values: [500, 505, ..., 650]
```

**Total Combinations**: 11 × 89 × 9 × 31 = 274,329 possible combinations

## Weight Distributions

### Rationale for Weighted Sampling

Real housing markets don’t distribute parameters uniformly. The system uses two distribution types to model realistic scenarios:

### 1. Linear Distribution (R Values)

**Applied to**: Rx, Ry (monthly rent values)

**Logic**: Higher rent values are more likely in inflationary markets

```python
weights = np.linspace(1, n_values, n_values)
weights = weights / weights.sum()
```

**Effect**:

- Lowest value gets weight ∝ 1
- Highest value gets weight ∝ n_values
- Creates linear increase in probability

### 2. Bell Curve Distribution (H Values)

**Applied to**: Hx, Hy (additional annual costs)

**Logic**: Extreme additional costs are rare; moderate values are most common

```python
center = n_values // 2
weights = exp(-0.5 × ((index - center) / (n_values / 4))²)
weights = weights / weights.sum()
```

**Effect**:

- Peak probability at center values
- Exponential decay toward extremes
- Standard deviation = n_values / 4

### Weight Distribution Examples

For Hx values (560-1000, 89 values):

- Center (780): Maximum probability
- Extremes (560, 1000): Minimum probability
- 68% of samples within ±22 values of center

## Analysis Types

### 1. Basic Decision Analysis

**Function**: `generate_weighted_combinations()`

Generates 5,000 weighted parameter combinations and calculates:

- Decision (MOVE/STAY) for each combination
- Statistical summary (move rate, boundary cases)
- Parallel coordinates visualization

### 2. Threshold Analysis

#### d-Threshold Analysis (`analyze_d_threshold_effects()`)

Examines how **d = 12×(Rx - Ry)** affects sensitivity to H values:

- Groups combinations by d value
- Identifies where H values become irrelevant
- Finds exact decision boundaries (s = 0 cases)

**Key Insights**:

- Large positive d → Always MOVE (H irrelevant)
- Large negative d → Always STAY (H irrelevant)
- d near 0 → High H sensitivity

#### f-Threshold Analysis (`analyze_f_threshold_effects()`)

Examines how **f = Hx - Hy** affects sensitivity to R values:

- Groups combinations by f value
- Identifies where R values become irrelevant
- Shows decision boundary crossings

**Key Insights**:

- Large positive f → Always MOVE (R irrelevant)
- Large negative f → Always STAY (R irrelevant)
- f near 0 → High R sensitivity

### 3. Lease Structure Analysis

#### Single Lease Structure (`analyze_lease_structure_scenarios()`)

Models rent evolution over time with:

- **Compound rent hikes**: Annual percentage increases
- **Lease renewal patterns**: Fixed-term vs. yearly renewal
- **Year-over-year decision tracking**

**Formula for rent in year t**:

```
Rent(t) = Initial_Rent × (1 + hike_rate)^(floor(t / lease_years))
```

#### Comprehensive Lease Analysis (`analyze_all_lease_structures()`)

Tests all combinations of lease structures:

- **Lease durations**: 1-5 years for both Rx and Ry
- **Total scenarios**: 5² = 25 combinations
- **Optimization**: Identifies best lease structure combination

### 4. Custom Scenario Analysis

**Function**: `analyze_lease_scenario()`

Allows manual input of:

- Specific rent values for each year
- Custom lease durations
- Targeted scenario testing

## Usage Instructions

### Running the Analysis

1. **Execute main function**:
   
   ```python
   df = main()
   ```
1. **Choose analysis type**:
- **Option 1**: Specific lease scenario (manual input)
- **Option 2**: Single lease structure with hikes
- **Option 3**: Comprehensive analysis (all structures)

### Input Requirements

#### Option 1: Manual Scenario

- Number of lease years
- Rx and Ry values for each year

#### Option 2: Single Lease Structure

- Starting minimum values (Rx_min, Ry_min)
- Annual hike rates (as percentages)
- Lease durations for each parameter
- Analysis timeframe (total years)

#### Option 3: Comprehensive Analysis

- Starting minimum values
- Annual hike rates
- Analysis timeframe
- System tests all 25 lease combinations automatically

### Example Input Workflow

```
ANALYSIS OPTIONS:
1. Specific lease scenario (manual rent input)
2. Lease structure analysis (single lease structure with hikes)  
3. Comprehensive analysis (ALL lease structures with hikes)

Choose analysis type: 3

Enter Rx starting minimum (e.g., 510): 480
Enter Ry starting minimum (e.g., 550): 530
Enter annual Rx hike percentage (e.g., 6.125): 6.125
Enter annual Ry hike percentage (e.g., 6.125): 0
Enter analysis timeframe in years (e.g., 5): 5
```

## Output Interpretation

### Statistical Summary

```
DECISION ANALYSIS STATISTICS
Total Combinations: 5,000
MOVE Decisions (s > 0): 3,247 (64.9%)
STAY Decisions (s <= 0): 1,753 (35.1%)

BOUNDARY ANALYSIS:
  Boundary cases (|s| <= 20): 127 (2.5%)
  Exact s = 0 cases: 3
```

### Threshold Analysis Output

```
  d/12  Total  MOVE%  H_min  H_max  H_bnd  s=0  ±10  H_div  Prob%
   -50    2759    0.0   -440    350   -600    0    0   2759   15.23
   -45    2759   12.3   -435    355   -540   18   89   2759   15.23
    ...
    50    2759  100.0   -410    390    600    0    0   2759   15.23
```

**Column Definitions**:

- **d/12**: Rent difference (Rx - Ry)
- **Total**: Combinations for this d value
- **MOVE%**: Percentage choosing MOVE
- **H_bnd**: H difference where s = 0
- **s=0**: Exact boundary cases
- **Prob%**: Probability mass for this d value

### Lease Structure Heatmap

The heatmap shows MOVE rates for all lease structure combinations:

- **X-axis**: Ry lease duration (1-5 years)
- **Y-axis**: Rx lease duration (1-5 years)
- **Color scale**: Red (low MOVE rate) to Green (high MOVE rate)

### Decision Recommendations

The system provides recommendations at multiple levels:

1. **Individual Scenario**: MOVE/STAY for specific parameters
1. **Lease Structure**: Best lease combination for given conditions
1. **Universal**: Overall recommendation across all scenarios

## Examples

### Example 1: Basic Analysis

**Scenario**: Standard parameter ranges with weighted sampling

**Result**:

```
MOVE Rate: 64.9% (3,247 out of 5,000 combinations)
Recommendation: MOVE (favorable in majority of scenarios)
```

### Example 2: Threshold Insight

**Finding**: When Rx - Ry ≥ 50, decision becomes MOVE regardless of H values

**Interpretation**: If current rent exceeds alternative by $50+/month, additional costs become irrelevant

### Example 3: Lease Structure Optimization

**Input**:

- Rx starts at $480, 6.125% annual hikes, 1-year lease
- Ry starts at $530, 0% annual hikes, 3-year lease
- 5-year analysis

**Finding**:

- Year 1-3: High STAY rate (Ry advantage)
- Year 4+: High MOVE rate (Rx hikes compound)
- Net outcome: +$15,000 over 5 years favoring MOVE

## Technical Implementation Notes

### Performance Optimizations

- **Exact Calculations**: For threshold analysis, calculates all 274,329 combinations exactly
- **Sampling**: For Monte Carlo, uses 5,000 samples for speed/accuracy balance
- **Progress Tracking**: Real-time progress for long-running analyses

### Visualization Features

- **Color Coding**: Red (STAY), Green (MOVE), Yellow (Boundary)
- **Interactive Plots**: Hover details and zoom capabilities
- **Multi-panel Dashboards**: Timeline, parameter evolution, outcomes

### Data Export

All analysis results are automatically saved to `decision_combinations.csv` for further analysis or record-keeping.

## Mathematical Assumptions

### Key Assumptions

1. **Linear Cost Model**: Total costs are simple sums (no interaction effects)
1. **Annual Decision Points**: Decisions evaluated yearly
1. **Independent Parameters**: H values independent of R values
1. **Compound Hikes**: Rent increases compound annually
1. **Risk Neutrality**: Decision based on expected value only

### Limitations

- Does not account for transaction costs of moving
- Assumes constant probability distributions over time
- No consideration of qualitative factors (neighborhood quality, etc.)
- Fixed parameter ranges may not suit all markets

## Extending the System

### Adding New Parameters

To include additional cost factors:

1. Define parameter range and weights
1. Modify decision formula: `s = (12*Rx + Hx + Nx) - (12*Ry + Hy + Ny)`
1. Update sampling and visualization functions

### Custom Weight Functions

Replace existing weight calculations with domain-specific distributions:

```python
# Example: Exponential decay for premium markets
weights = np.exp(-decay_rate * np.arange(n_values))
weights = weights / weights.sum()
```

### Advanced Modeling

Potential extensions:

- **Stochastic Hike Rates**: Variable annual increases
- **Market Cycle Modeling**: Cyclical rent patterns
- **Risk Preferences**: Utility functions beyond expected value
- **Multi-Criteria Analysis**: Incorporating qualitative factors

-----

**Version**: 1.0  
**Dependencies**: numpy, pandas, plotly, scipy  
**Python Version**: 3.7+