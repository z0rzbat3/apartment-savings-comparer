def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot with boundary highlighting"""
    
    # Create a custom color mapping for boundary highlighting
    # 0 = STAY (red), 1 = MOVE (green), 2 = BOUNDARY (yellow)
    def get_color_value(row):
        if abs(row['s']) <= 20:  # Boundary case
            return 2
        elif row['s'] > 0:  # MOVE
            return 1
        else:  # STAY
            return 0
    
    df['color_value'] = df.apply(get_color_value, axis=1)
    
    # Create the parallel coordinates plot with custom colorscale
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df['color_value'],
            colorscale=[[0, '#dc3545'], [0.5, '#28a745'], [1, '#ffc107']],  # Red, Green, Yellow
            showscale=True,
            colorbar=dict(
                title="Decision Type",
                tickvals=[0, 1, 2],
                ticktext=["STAY", "MOVE", "BOUNDARY"]
            )
        ),
        dimensions=[
            dict(
                range=[df['Rx'].min()-5, df['Rx'].max()+5],
                label='Rx',
                values=df['Rx'],
                tickvals=sorted(df['Rx'].unique()),
                ticktext=[str(x) for x in sorted(df['Rx'].unique())]
            ),
            dict(
                range=[df['Hx'].min()-25, df['Hx'].max()+25],
                label='Hx',
                values=df['Hx'],
                tickvals=[1000, 1200, 1400, 1600, 1700],
                ticktext=['1000', '1200', '1400', '1600', '1700']
            ),
            dict(
                range=[df['Ry'].min()-5, df['Ry'].max()+5],
                label='Ry',
                values=df['Ry'],
                tickvals=sorted(df['Ry'].unique()),
                ticktext=[str(x) for x in sorted(df['Ry'].unique())]
            ),
            dict(
                range=[df['Hy'].min()-25, df['Hy'].max()+25],
                label='Hy',
                values=df['Hy'],
                tickvals=[500, 700, 900, 1100, 1200],
                ticktext=['500', '700', '900', '1100', '1200']
            ),
            dict(
                range=[df['s'].min()-10, df['s'].max()+10],
                label='s = (12*Rx+Hx)-(12*Ry+Hy)',
                values=df['s'],
                tickvals=[-200, 0, 200, 400, 600, 800, 1000, 1200],
                ticktext=['-200', '0', '200', '400', '600', '800', '1000', '1200']
            )
        ]
    ))
    
    # Calculate statistics
    total_combinations = len(df)
    move_count = len(df[df['Decision'] == 'MOVE'])
    stay_count = len(df[df['Decision'] == 'STAY'])
    move_percentage = (move_count / total_combinations) * 100
    
    fig.update_layout(
        title=dict(
            text=f'Move/Stay Decision Analysis - {total_combinations:,} Weighted Samples<br>' +
                 f'R: Linear↗ | H: Bell🔔 | ' +
                 f'<span style="color: #28a745">MOVE: {move_count:,} ({move_percentage:.1f}%)</span> | ' +
                 f'<span style="color: #dc3545">STAY: {stay_count:,} ({100-move_percentage:.1f}%)</span>',
            x=0.5,
            font=dict(size=16)
        ),
        font=dict(size=12),
        height=600,
        margin=dict(l=80, r=80, t=120, b=80)
    )
    
    return fig

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def generate_weighted_combinations(n_samples=5000):
    """Generate weighted parameter combinations using different distributions"""
    np.random.seed(42)  # For reproducible results
    
    # Define parameter ranges
    rx_values = np.array(range(480, 531, 5))  # [480, 485, 490, ..., 530]
    hx_values = np.array(range(560, 1001, 5))  # [560, 565, 570, ..., 1000]
    ry_values = np.array(range(530, 571, 5))  # [530, 535, 540, ..., 570]
    hy_values = np.array(range(500, 651, 5))  # [500, 505, 510, ..., 650]
    
    # Linear weights for R values (higher values more likely)
    rx_weights = np.linspace(1, len(rx_values), len(rx_values))
    rx_weights = rx_weights / rx_weights.sum()
    
    ry_weights = np.linspace(1, len(ry_values), len(ry_values))
    ry_weights = ry_weights / ry_weights.sum()
    
    # Bell curve (normal) weights for H values (middle values more likely)
    hx_center = len(hx_values) // 2
    hx_weights = np.exp(-0.5 * ((np.arange(len(hx_values)) - hx_center) / (len(hx_values) / 4)) ** 2)
    hx_weights = hx_weights / hx_weights.sum()
    
    hy_center = len(hy_values) // 2
    hy_weights = np.exp(-0.5 * ((np.arange(len(hy_values)) - hy_center) / (len(hy_values) / 4)) ** 2)
    hy_weights = hy_weights / hy_weights.sum()
    
    # Generate weighted samples
    combinations = []
    for _ in range(n_samples):
        rx = np.random.choice(rx_values, p=rx_weights)
        hx = np.random.choice(hx_values, p=hx_weights)
        ry = np.random.choice(ry_values, p=ry_weights)
        hy = np.random.choice(hy_values, p=hy_weights)
        
        # Calculate decision score
        s = (12*rx + hx) - (12*ry + hy)
        decision = "MOVE" if s > 0 else "STAY"
        
        combinations.append({
            'Rx': rx,
            'Hx': hx, 
            'Ry': ry,
            'Hy': hy,
            's': s,
            'Decision': decision,
            'Decision_Numeric': 1 if s > 0 else 0
        })
    
    return pd.DataFrame(combinations)

def calculate_top_probable_combinations():
    """Calculate the exact probabilities and return top combinations"""
    # Define parameter ranges
    rx_values = np.array(range(480, 531, 5))  # [480, 485, 490, ..., 530]
    hx_values = np.array(range(560, 1001, 5))  # [560, 565, 570, ..., 1000]
    ry_values = np.array(range(530, 571, 5))  # [530, 535, 540, ..., 570]
    hy_values = np.array(range(500, 651, 5))  # [500, 505, 510, ..., 650]
    
    # Calculate weights (same as in weighted function)
    rx_weights = np.linspace(1, len(rx_values), len(rx_values))
    rx_weights = rx_weights / rx_weights.sum()
    
    ry_weights = np.linspace(1, len(ry_values), len(ry_values))
    ry_weights = ry_weights / ry_weights.sum()
    
    hx_center = len(hx_values) // 2
    hx_weights = np.exp(-0.5 * ((np.arange(len(hx_values)) - hx_center) / (len(hx_values) / 4)) ** 2)
    hx_weights = hx_weights / hx_weights.sum()
    
    hy_center = len(hy_values) // 2
    hy_weights = np.exp(-0.5 * ((np.arange(len(hy_values)) - hy_center) / (len(hy_values) / 4)) ** 2)
    hy_weights = hy_weights / hy_weights.sum()
    
    # Calculate all combinations with their probabilities
    all_combinations = []
    for i, rx in enumerate(rx_values):
        for j, hx in enumerate(hx_values):
            for k, ry in enumerate(ry_values):
                for l, hy in enumerate(hy_values):
                    # Joint probability = product of individual probabilities
                    probability = rx_weights[i] * hx_weights[j] * ry_weights[k] * hy_weights[l]
                    
                    cx = 12*rx + hx
                    cy = 12*ry + hy
                    s = cx - cy
                    decision = "MOVE" if s > 0 else "STAY"
                    
                    all_combinations.append({
                        'Rx': rx,
                        'Hx': hx,
                        'Ry': ry,
                        'Hy': hy,
                        's': s,
                        'Decision': decision,
                        'Probability': probability
                    })
    
    # Convert to DataFrame and sort by probability
    df_all = pd.DataFrame(all_combinations)
    df_sorted = df_all.sort_values('Probability', ascending=False)
    
    return df_sorted

def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot showing MOVE/STAY decisions"""
    
    # Create figure with single trace for all data
    fig = go.Figure()
    
    # Single trace colored by MOVE/STAY decision
    fig.add_trace(go.Parcoords(
        line=dict(
            color=df['Decision_Numeric'],
            colorscale=[[0, '#dc3545'], [1, '#28a745']],  # Red for STAY, Green for MOVE
            showscale=True,
            colorbar=dict(
                title="Decision",
                tickvals=[0, 1],
                ticktext=["STAY", "MOVE"]
            )
        ),
        dimensions=[
            dict(
                range=[df['Rx'].min()-5, df['Rx'].max()+5],
                label='Rx',
                values=df['Rx'],
                tickvals=sorted(df['Rx'].unique()),
                ticktext=[str(x) for x in sorted(df['Rx'].unique())]
            ),
            dict(
                range=[df['Hx'].min()-25, df['Hx'].max()+25],
                label='Hx', 
                values=df['Hx'],
                tickvals=[560, 650, 700, 750, 800, 850, 900, 950, 1000],
                ticktext=['560', '650', '700', '750', '800', '850', '900', '950', '1000']
            ),
            dict(
                range=[df['Ry'].min()-5, df['Ry'].max()+5],
                label='Ry',
                values=df['Ry'],
                tickvals=sorted(df['Ry'].unique()),
                ticktext=[str(x) for x in sorted(df['Ry'].unique())]
            ),
            dict(
                range=[df['Hy'].min()-25, df['Hy'].max()+25],
                label='Hy',
                values=df['Hy'],
                tickvals=[500, 525, 550, 575, 600, 625, 650],
                ticktext=['500', '525', '550', '575', '600', '625', '650']
            ),
            dict(
                range=[df['s'].min()-10, df['s'].max()+10],
                label='s = (12*Rx+Hx)-(12*Ry+Hy)',
                values=df['s'],
                tickvals=[-100, 0, 100, 200, 300, 400, 500],
                ticktext=['-100', '0', '100', '200', '300', '400', '500']
            )
        ]
    ))
    
    # Calculate statistics
    total_combinations = len(df)
    move_count = len(df[df['Decision'] == 'MOVE'])
    stay_count = len(df[df['Decision'] == 'STAY'])
    move_percentage = (move_count / total_combinations) * 100
    
    fig.update_layout(
        title=dict(
            text=f'Move/Stay Decision Analysis - {total_combinations:,} Weighted Samples<br>' +
                 f'R: Linear↗ | H: Bell🔔 | ' +
                 f'<span style="color: #28a745">MOVE: {move_count:,} ({move_percentage:.1f}%)</span> | ' +
                 f'<span style="color: #dc3545">STAY: {stay_count:,} ({100-move_percentage:.1f}%)</span>',
            x=0.5,
            font=dict(size=16)
        ),
        font=dict(size=12),
        height=600,
        margin=dict(l=80, r=80, t=120, b=80)
    )
    
    return fig

def print_statistics(df):
    """Print detailed statistics including boundary cases"""
    total = len(df)
    move_count = len(df[df['Decision'] == 'MOVE'])
    stay_count = len(df[df['Decision'] == 'STAY'])
    
    # Boundary cases analysis
    boundary_df = df[abs(df['s']) <= 20]
    boundary_count = len(boundary_df)
    exact_zero = len(df[df['s'] == 0])
    
    print("="*60)
    print("DECISION ANALYSIS STATISTICS")
    print("="*60)
    print(f"Total Combinations: {total:,}")
    print(f"MOVE Decisions (s > 0): {move_count:,} ({move_count/total*100:.1f}%)")
    print(f"STAY Decisions (s <= 0): {stay_count:,} ({stay_count/total*100:.1f}%)")
    print()
    print("BOUNDARY ANALYSIS:")
    print(f"  Boundary cases (|s| <= 20): {boundary_count:,} ({boundary_count/total*100:.1f}%)")
    print(f"  Exact s = 0 cases: {exact_zero}")
    print()
    print("s Value Statistics:")
    print(f"  Minimum s: {df['s'].min()}")
    print(f"  Maximum s: {df['s'].max()}")
    print(f"  Mean s: {df['s'].mean():.1f}")
    print(f"  Median s: {df['s'].median():.1f}")
    print(f"  Standard deviation: {df['s'].std():.1f}")
    print()
    print("Parameter Ranges:")
    print(f"  Rx: {df['Rx'].min()} to {df['Rx'].max()}")
    print(f"  Hx: {df['Hx'].min()} to {df['Hx'].max()}")
    print(f"  Ry: {df['Ry'].min()} to {df['Ry'].max()}")
    print(f"  Hy: {df['Hy'].min()} to {df['Hy'].max()}")
    
    # Show some boundary cases
    if boundary_count > 0:
        print()
        print("Sample Boundary Cases (|s| <= 20):")
        boundary_sample = boundary_df.nsmallest(10, 's')[['Rx', 'Hx', 'Ry', 'Hy', 's', 'Decision']]
        print(boundary_sample.to_string(index=False))
    
    # Show exact s=0 cases if any
    if exact_zero > 0:
        print()
        print("Exact s = 0 Cases:")
        zero_cases = df[df['s'] == 0][['Rx', 'Hx', 'Ry', 'Hy', 's', 'Decision']]
        print(zero_cases.to_string(index=False))

def analyze_d_threshold_effects():
    """Analyze how d=Rx-Ry affects H value sensitivity across all combinations"""
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS: d = Rx - Ry EFFECTS ON H VALUE SENSITIVITY")
    print("="*80)
    
    # Get all combinations with probabilities
    df_all = calculate_top_probable_combinations()
    
    # Calculate d = 12*(Rx - Ry), but display as d/12 for R interpretation
    df_all['d'] = 12*(df_all['Rx'] - df_all['Ry'])
    df_all['d_display'] = df_all['d'] / 12  # This is just Rx - Ry for easier interpretation
    df_all['Hx_minus_Hy'] = df_all['Hx'] - df_all['Hy']
    
    # Group by d value and analyze H sensitivity
    d_analysis = []
    for d_val in sorted(df_all['d'].unique()):
        d_subset = df_all[df_all['d'] == d_val]
        
        # Count total combinations for this d
        total_combinations = len(d_subset)
        
        # Count MOVE vs STAY
        move_count = len(d_subset[d_subset['Decision'] == 'MOVE'])
        stay_count = len(d_subset[d_subset['Decision'] == 'STAY'])
        
        # H difference ranges for this d
        h_diff_min = d_subset['Hx_minus_Hy'].min()
        h_diff_max = d_subset['Hx_minus_Hy'].max()
        
        # Find decision boundary in H space (where s crosses 0)
        # s = d + (Hx - Hy) = 0, so Hx - Hy = -d
        h_boundary = -d_val
        
        # Count how many H combinations give MOVE vs STAY
        boundary_crossings = len(d_subset[d_subset['s'] == 0])
        near_boundary = len(d_subset[abs(d_subset['s']) <= 10])  # Within ±10
        
        # Calculate H diversity (unique H combinations)
        h_combinations = d_subset.groupby(['Hx', 'Hy']).size().reset_index()
        h_diversity = len(h_combinations)
        
        # Total probability mass for this d
        prob_mass = d_subset['Probability'].sum()
        
        d_analysis.append({
            'd': d_val,
            'd_display': d_val / 12,  # Show as R difference for easier interpretation
            'total_combos': total_combinations,
            'move_count': move_count,
            'stay_count': stay_count,
            'move_pct': move_count/total_combinations*100,
            'h_diff_min': h_diff_min,
            'h_diff_max': h_diff_max,
            'h_boundary': h_boundary,
            'boundary_crossings': boundary_crossings,
            'near_boundary': near_boundary,
            'h_diversity': h_diversity,
            'prob_mass': prob_mass
        })
    
    # Display analysis
    print(f"{'d/12':>6} {'Total':>6} {'MOVE%':>6} {'H_min':>6} {'H_max':>6} {'H_bnd':>6} {'s=0':>4} {'±10':>4} {'H_div':>5} {'Prob%':>6}")
    print("-" * 82)
    
    for analysis in d_analysis:
        print(f"{analysis['d_display']:>6.0f} {analysis['total_combos']:>6d} {analysis['move_pct']:>6.1f} "
              f"{analysis['h_diff_min']:>6d} {analysis['h_diff_max']:>6d} {analysis['h_boundary']:>6d} "
              f"{analysis['boundary_crossings']:>4d} {analysis['near_boundary']:>4d} "
              f"{analysis['h_diversity']:>5d} {analysis['prob_mass']*100:>6.2f}")
    
    # Find threshold insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Always MOVE (100% MOVE regardless of H)
    always_move = [a for a in d_analysis if a['move_pct'] == 100.0]
    if always_move:
        d_vals = [str(int(a['d_display'])) for a in always_move]
        print(f"ALWAYS MOVE (H doesn't matter): Rx-Ry = {', '.join(d_vals)}")
    
    # Always STAY (0% MOVE regardless of H)
    always_stay = [a for a in d_analysis if a['move_pct'] == 0.0]
    if always_stay:
        d_vals = [str(int(a['d_display'])) for a in always_stay]
        print(f"ALWAYS STAY (H doesn't matter): Rx-Ry = {', '.join(d_vals)}")
    
    # High H sensitivity (many boundary crossings)
    high_sensitivity = [a for a in d_analysis if a['boundary_crossings'] > 0]
    if high_sensitivity:
        d_vals = [str(int(a['d_display'])) for a in high_sensitivity]
        print(f"HIGH H SENSITIVITY (exact s=0 cases): Rx-Ry = {', '.join(d_vals)}")
    
    # Most probable d values
    top_d_by_prob = sorted(d_analysis, key=lambda x: x['prob_mass'], reverse=True)[:3]
    print(f"\nMOST PROBABLE Rx-Ry VALUES:")
    for i, analysis in enumerate(top_d_by_prob, 1):
        print(f"  #{i}: Rx-Ry={int(analysis['d_display'])}, {analysis['prob_mass']*100:.2f}% of probability mass")
    
    return df_all, d_analysis

def analyze_f_threshold_effects():
    """Analyze how f=Hx-Hy affects R value sensitivity across all combinations"""
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS: f = Hx - Hy EFFECTS ON R VALUE SENSITIVITY")
    print("="*80)
    
    # Get all combinations with probabilities
    df_all = calculate_top_probable_combinations()
    
    # Calculate f = Hx - Hy and d = 12*(Rx - Ry)
    df_all['f'] = df_all['Hx'] - df_all['Hy']
    df_all['d'] = 12*(df_all['Rx'] - df_all['Ry'])
    
    # Group by f value and analyze R sensitivity
    f_analysis = []
    for f_val in sorted(df_all['f'].unique()):
        f_subset = df_all[df_all['f'] == f_val]
        
        # Count total combinations for this f
        total_combinations = len(f_subset)
        
        # Count MOVE vs STAY
        move_count = len(f_subset[f_subset['Decision'] == 'MOVE'])
        stay_count = len(f_subset[f_subset['Decision'] == 'STAY'])
        
        # R difference ranges for this f
        d_min = f_subset['d'].min()
        d_max = f_subset['d'].max()
        
        # Find decision boundary in R space (where s crosses 0)
        # s = d + f = 0, so d = -f
        r_boundary = -f_val
        
        # Count how many R combinations give MOVE vs STAY
        boundary_crossings = len(f_subset[f_subset['s'] == 0])
        near_boundary = len(f_subset[abs(f_subset['s']) <= 10])  # Within ±10
        
        # Calculate R diversity (unique R combinations)
        r_combinations = f_subset.groupby(['Rx', 'Ry']).size().reset_index()
        r_diversity = len(r_combinations)
        
        # Total probability mass for this f
        prob_mass = f_subset['Probability'].sum()
        
        # Check if this f value makes R irrelevant
        r_irrelevant = False
        if f_val > 0:  # Positive f favors MOVE
            # R irrelevant if even worst R (d_min) still gives MOVE
            if d_min + f_val > 0:
                r_irrelevant = True
        else:  # Negative f favors STAY  
            # R irrelevant if even best R (d_max) still gives STAY
            if d_max + f_val <= 0:
                r_irrelevant = True
        
        f_analysis.append({
            'f': f_val,
            'total_combos': total_combinations,
            'move_count': move_count,
            'stay_count': stay_count,
            'move_pct': move_count/total_combinations*100,
            'd_min': d_min,
            'd_max': d_max,
            'r_boundary': r_boundary,
            'boundary_crossings': boundary_crossings,
            'near_boundary': near_boundary,
            'r_diversity': r_diversity,
            'prob_mass': prob_mass,
            'r_irrelevant': r_irrelevant
        })
    
    # Display analysis
    print(f"{'f':>4} {'Total':>6} {'MOVE%':>6} {'d_min':>6} {'d_max':>6} {'R_bnd':>6} {'s=0':>4} {'±10':>4} {'R_div':>5} {'Prob%':>6} {'R_Irr':>5}")
    print("-" * 90)
    
    for analysis in f_analysis:
        r_irr_mark = "YES" if analysis['r_irrelevant'] else "NO"
        print(f"{analysis['f']:>4d} {analysis['total_combos']:>6d} {analysis['move_pct']:>6.1f} "
              f"{analysis['d_min']:>6d} {analysis['d_max']:>6d} {analysis['r_boundary']:>6d} "
              f"{analysis['boundary_crossings']:>4d} {analysis['near_boundary']:>4d} "
              f"{analysis['r_diversity']:>5d} {analysis['prob_mass']*100:>6.2f} {r_irr_mark:>5s}")
    
    # Find threshold insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Always MOVE (100% MOVE regardless of R)
    always_move = [a for a in f_analysis if a['move_pct'] == 100.0]
    if always_move:
        f_vals = [str(a['f']) for a in always_move]
        print(f"ALWAYS MOVE (R doesn't matter): f = {', '.join(f_vals)}")
    
    # Always STAY (0% MOVE regardless of R)
    always_stay = [a for a in f_analysis if a['move_pct'] == 0.0]
    if always_stay:
        f_vals = [str(a['f']) for a in always_stay]
        print(f"ALWAYS STAY (R doesn't matter): f = {', '.join(f_vals)}")
    
    # R irrelevant cases
    r_irrelevant = [a for a in f_analysis if a['r_irrelevant']]
    if r_irrelevant:
        f_vals = [str(a['f']) for a in r_irrelevant]
        print(f"R VALUES IRRELEVANT: f = {', '.join(f_vals)}")
    
    # High R sensitivity (many boundary crossings)
    high_sensitivity = [a for a in f_analysis if a['boundary_crossings'] > 0]
    if high_sensitivity:
        f_vals = [str(a['f']) for a in high_sensitivity]
        print(f"HIGH R SENSITIVITY (exact s=0 cases): f = {', '.join(f_vals)}")
    
    # Most probable f values
    top_f_by_prob = sorted(f_analysis, key=lambda x: x['prob_mass'], reverse=True)[:3]
    print(f"\nMOST PROBABLE f VALUES:")
    for i, analysis in enumerate(top_f_by_prob, 1):
        print(f"  #{i}: f={analysis['f']}, {analysis['prob_mass']*100:.2f}% of probability mass")
    
    # Theoretical thresholds
    print(f"\nTHEORETICAL THRESHOLDS (given d range {f_analysis[0]['d_min']} to {f_analysis[0]['d_max']}):")
    d_range_min = min([a['d_min'] for a in f_analysis])
    d_range_max = max([a['d_max'] for a in f_analysis])
    always_move_threshold = -d_range_min  # f > this makes R irrelevant for MOVE
    always_stay_threshold = -d_range_max   # f < this makes R irrelevant for STAY
    
    print(f"  f > {always_move_threshold} -> ALWAYS MOVE (R irrelevant)")
    print(f"  f < {always_stay_threshold} -> ALWAYS STAY (R irrelevant)")
    
    return df_all, f_analysis


def analyze_lease_structure_scenarios(rx_min, ry_min, rx_hike_rate, ry_hike_rate, rx_lease_years, ry_lease_years, analysis_timeframe):
    """
    Analyze all parameter combinations across lease structure scenarios
    
    Parameters:
    - rx_min: Starting minimum Rx value
    - ry_min: Starting minimum Ry value
    - rx_hike_rate: Annual Rx hike rate (decimal)
    - ry_hike_rate: Annual Ry hike rate (decimal)  
    - rx_lease_years: Rx lease duration before renewal
    - ry_lease_years: Ry lease duration before renewal
    - analysis_timeframe: Total years to analyze
    
    Returns:
    - Comprehensive analysis across all scenarios
    """
    # Parameter ranges starting from user-defined minimums
    rx_base_values = np.array(range(rx_min, rx_min + 51, 5))  # 50-point range
    ry_base_values = np.array(range(ry_min, ry_min + 41, 5))  # 40-point range
    hx_values = np.array(range(560, 1001, 5))
    hy_values = np.array(range(500, 651, 5))
    
    # Calculate H weights (same as existing)
    hx_center = len(hx_values) // 2
    hx_weights = np.exp(-0.5 * ((np.arange(len(hx_values)) - hx_center) / (len(hx_values) / 4)) ** 2)
    hx_weights = hx_weights / hx_weights.sum()
    
    hy_center = len(hy_values) // 2
    hy_weights = np.exp(-0.5 * ((np.arange(len(hy_values)) - hy_center) / (len(hy_values) / 4)) ** 2)
    hy_weights = hy_weights / hy_weights.sum()
    
    # Calculate parameter ranges for each year
    yearly_ranges = []
    for year in range(analysis_timeframe):
        # Determine if rent resets this year
        rx_resets = (year % rx_lease_years == 0)
        ry_resets = (year % ry_lease_years == 0)
        
        # Calculate total years of hikes applied based on lease structure
        # For 1-year lease: Year 1=0 hikes, Year 2=1 hike, Year 3=2 hikes
        # For 3-year lease: Year 1-3=0 hikes, Year 4-6=1 hike, Year 7-9=2 hikes
        rx_hike_years = year // rx_lease_years
        ry_hike_years = year // ry_lease_years
        
        # Calculate ranges with compound hikes
        rx_range = rx_base_values * ((1 + rx_hike_rate) ** rx_hike_years)
        ry_range = ry_base_values * ((1 + ry_hike_rate) ** ry_hike_years)
        
        yearly_ranges.append({
            'year': year + 1,
            'rx_range': rx_range.astype(int),
            'ry_range': ry_range.astype(int),
            'rx_resets': rx_resets,
            'ry_resets': ry_resets
        })
    
    # Analyze all combinations for each year
    all_results = []
    yearly_summaries = []
    
    np.random.seed(42)
    
    for year_data in yearly_ranges:
        year = year_data['year']
        rx_range = year_data['rx_range']
        ry_range = year_data['ry_range']
        
        year_combinations = []
        
        # Generate all R combinations for this year
        for rx in rx_range:
            for ry in ry_range:
                # Sample H values (using existing weights)
                move_count = 0
                total_savings = 0
                
                for _ in range(50):  # Reduced samples for performance
                    hx = np.random.choice(hx_values, p=hx_weights)
                    hy = np.random.choice(hy_values, p=hy_weights)
                    
                    s = (12*rx + hx) - (12*ry + hy)
                    
                    if s > 0:
                        move_count += 1
                    total_savings += s
                
                move_prob = move_count / 50
                avg_outcome = total_savings / 50
                decision = "MOVE" if move_prob >= 0.5 else "STAY"
                
                combination_result = {
                    'year': year,
                    'rx': rx,
                    'ry': ry,
                    'move_probability': move_prob,
                    'decision': decision,
                    'avg_outcome': avg_outcome,
                    'rx_resets': year_data['rx_resets'],
                    'ry_resets': year_data['ry_resets']
                }
                
                year_combinations.append(combination_result)
                all_results.append(combination_result)
        
        # Calculate year summary
        year_move_count = sum(1 for r in year_combinations if r['decision'] == 'MOVE')
        year_total = len(year_combinations)
        year_move_rate = year_move_count / year_total * 100
        
        move_results = [r for r in year_combinations if r['decision'] == 'MOVE']
        stay_results = [r for r in year_combinations if r['decision'] == 'STAY']
        
        total_move_savings = sum(r['avg_outcome'] for r in move_results)
        total_stay_damages = sum(r['avg_outcome'] for r in stay_results)
        avg_move_savings = total_move_savings / len(move_results) if move_results else 0
        avg_stay_damages = total_stay_damages / len(stay_results) if stay_results else 0
        
        yearly_summaries.append({
            'year': year,
            'total_combinations': year_total,
            'move_count': year_move_count,
            'stay_count': year_total - year_move_count,
            'move_rate': year_move_rate,
            'rx_range': f"{rx_range[0]}-{rx_range[-1]}",
            'ry_range': f"{ry_range[0]}-{ry_range[-1]}",
            'total_move_savings': total_move_savings,
            'total_stay_damages': total_stay_damages,
            'avg_move_savings': avg_move_savings,
            'avg_stay_damages': avg_stay_damages,
            'rx_resets': year_data['rx_resets'],
            'ry_resets': year_data['ry_resets']
        })
    
    return all_results, yearly_summaries, yearly_ranges


def create_lease_structure_dashboard(all_results, yearly_summaries, yearly_ranges, rx_hike_rate, ry_hike_rate, rx_lease_years, ry_lease_years, analysis_timeframe):
    """Create comprehensive dashboard for lease structure analysis"""
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'MOVE Rate Over Time',
            'Parameter Ranges Evolution', 
            'Annual Savings/Damages',
            'Decision Summary by Year'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Extract data
    years = [s['year'] for s in yearly_summaries]
    move_rates = [s['move_rate'] for s in yearly_summaries]
    move_savings = [s['avg_move_savings'] for s in yearly_summaries]
    stay_damages = [s['avg_stay_damages'] for s in yearly_summaries]
    
    # Plot 1: MOVE Rate Timeline
    fig.add_trace(
        go.Scatter(
            x=years,
            y=move_rates,
            mode='lines+markers',
            name='MOVE Rate (%)',
            line=dict(color='blue', width=3),
            marker=dict(size=10),
            hovertemplate='Year %{x}<br>MOVE Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add lease reset indicators
    for i, year_data in enumerate(yearly_ranges):
        if year_data['rx_resets'] or year_data['ry_resets']:
            reset_label = []
            if year_data['rx_resets']: reset_label.append('Rx')
            if year_data['ry_resets']: reset_label.append('Ry')
            
            fig.add_vline(
                x=year_data['year'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"{'+'.join(reset_label)} Reset",
                row=1, col=1
            )
    
    # Plot 2: Parameter Ranges (simplified as midpoint tracking)
    rx_midpoints = []
    ry_midpoints = []
    for yr_data in yearly_ranges:
        rx_mid = (yr_data['rx_range'][0] + yr_data['rx_range'][-1]) / 2
        ry_mid = (yr_data['ry_range'][0] + yr_data['ry_range'][-1]) / 2
        rx_midpoints.append(rx_mid)
        ry_midpoints.append(ry_mid)
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=rx_midpoints,
            mode='lines+markers',
            name=f'Rx Mid ({rx_hike_rate*100:.2f}%/yr, {rx_lease_years}yr lease)',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=ry_midpoints,
            mode='lines+markers', 
            name=f'Ry Mid ({ry_hike_rate*100:.2f}%/yr, {ry_lease_years}yr lease)',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    # Plot 3: Savings/Damages
    fig.add_trace(
        go.Bar(
            x=years,
            y=move_savings,
            name='Avg MOVE Savings',
            marker_color='green',
            hovertemplate='Year %{x}<br>Avg MOVE Savings: %{y:+.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=years,
            y=stay_damages,
            name='Avg STAY Damages',
            marker_color='red',
            hovertemplate='Year %{x}<br>Avg STAY Damages: %{y:+.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: Decision Counts
    move_counts = [s['move_count'] for s in yearly_summaries]
    stay_counts = [s['stay_count'] for s in yearly_summaries]
    
    fig.add_trace(
        go.Bar(
            x=years,
            y=move_counts,
            name='MOVE Count',
            marker_color='lightblue'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=years,
            y=stay_counts,
            name='STAY Count', 
            marker_color='lightcoral'
        ),
        row=2, col=2
    )
    
    # Calculate overall statistics
    all_move_count = sum(1 for r in all_results if r['decision'] == 'MOVE')
    total_combinations = len(all_results)
    overall_move_rate = all_move_count / total_combinations * 100
    
    all_move_savings = sum(r['avg_outcome'] for r in all_results if r['decision'] == 'MOVE')
    all_stay_damages = sum(r['avg_outcome'] for r in all_results if r['decision'] == 'STAY')
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Lease Structure Analysis: {analysis_timeframe}-Year Timeline<br>' +
                 f'Overall MOVE Rate: {overall_move_rate:.1f}% | ' +
                 f'Total MOVE Savings: {all_move_savings:+,.0f} | ' +
                 f'Total STAY Damages: {all_stay_damages:+,.0f}',
            x=0.5,
            font=dict(size=14)
        ),
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="MOVE Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Rent Amount", row=1, col=2)
    fig.update_yaxes(title_text="Annual Outcome", row=2, col=1)
    fig.update_yaxes(title_text="Combinations Count", row=2, col=2)
    
    return fig


def get_lease_structure_inputs():
    """Get inputs for lease structure analysis"""
    print("\n" + "="*60)
    print("LEASE STRUCTURE ANALYSIS INPUT")
    print("="*60)
    
    try:
        # Starting range conditions
        rx_min = int(input("Enter Rx starting minimum (e.g., 480): "))
        ry_min = int(input("Enter Ry starting minimum (e.g., 530): "))
        
        # Hike rates
        rx_hike_input = input("Enter annual Rx hike percentage (e.g., 6.125): ")
        rx_hike_rate = float(rx_hike_input) / 100.0
        
        ry_hike_input = input("Enter annual Ry hike percentage (e.g., 0): ")
        ry_hike_rate = float(ry_hike_input) / 100.0
        
        # Lease structures
        rx_lease_years = int(input("Enter Rx lease duration in years (e.g., 1 for yearly renewal): "))
        ry_lease_years = int(input("Enter Ry lease duration in years (e.g., 3 for 3-year fixed): "))
        
        analysis_timeframe = int(input("Enter analysis timeframe in years (e.g., 5): "))
        
        return rx_min, ry_min, rx_hike_rate, ry_hike_rate, rx_lease_years, ry_lease_years, analysis_timeframe
        
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return None, None, None, None, None, None, None


def analyze_all_lease_structures(rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe):
    """
    Analyze ALL possible lease structure combinations
    
    Parameters:
    - rx_min: Starting minimum Rx value
    - ry_min: Starting minimum Ry value
    - rx_hike_rate: Annual Rx hike rate (decimal)
    - ry_hike_rate: Annual Ry hike rate (decimal)
    - analysis_timeframe: Total years to analyze
    
    Returns:
    - Comprehensive analysis across all lease structure combinations
    """
    lease_lengths = [1, 2, 3, 4, 5]  # Lease duration options
    all_lease_scenarios = []
    
    total_scenarios = len(lease_lengths) ** 2
    current_scenario = 0
    
    print(f"Analyzing {total_scenarios} lease structure combinations...")
    print("Progress: ", end="", flush=True)
    
    for rx_lease_years in lease_lengths:
        for ry_lease_years in lease_lengths:
            current_scenario += 1
            
            # Progress indicator
            if current_scenario % 5 == 0:
                print(f"{current_scenario}/{total_scenarios} ", end="", flush=True)
            
            # Run analysis for this lease structure combination
            all_results, yearly_summaries, yearly_ranges = analyze_lease_structure_scenarios(
                rx_min, ry_min, rx_hike_rate, ry_hike_rate, 
                rx_lease_years, ry_lease_years, analysis_timeframe
            )
            
            # Calculate overall metrics for this lease combination
            all_move_count = sum(1 for r in all_results if r['decision'] == 'MOVE')
            total_combinations = len(all_results)
            overall_move_rate = all_move_count / total_combinations * 100
            
            all_move_savings = sum(r['avg_outcome'] for r in all_results if r['decision'] == 'MOVE')
            all_stay_damages = sum(r['avg_outcome'] for r in all_results if r['decision'] == 'STAY')
            net_outcome = all_move_savings + all_stay_damages
            
            # Calculate year-end ranges
            final_year_data = yearly_ranges[-1]
            final_rx_range = f"{final_year_data['rx_range'][0]}-{final_year_data['rx_range'][-1]}"
            final_ry_range = f"{final_year_data['ry_range'][0]}-{final_year_data['ry_range'][-1]}"
            
            lease_scenario = {
                'rx_lease_years': rx_lease_years,
                'ry_lease_years': ry_lease_years,
                'overall_move_rate': overall_move_rate,
                'total_combinations': total_combinations,
                'move_count': all_move_count,
                'stay_count': total_combinations - all_move_count,
                'total_move_savings': all_move_savings,
                'total_stay_damages': all_stay_damages,
                'net_outcome': net_outcome,
                'final_rx_range': final_rx_range,
                'final_ry_range': final_ry_range,
                'yearly_summaries': yearly_summaries
            }
            
            all_lease_scenarios.append(lease_scenario)
    
    print("\nCompleted!")
    return all_lease_scenarios


def create_lease_structure_heatmap(lease_scenarios, rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe):
    """Create heatmap showing MOVE rates for all lease structure combinations"""
    
    lease_lengths = [1, 2, 3, 4, 5]
    
    # Create matrix of MOVE rates
    move_rate_matrix = []
    net_outcome_matrix = []
    
    for rx_lease in lease_lengths:
        move_row = []
        outcome_row = []
        for ry_lease in lease_lengths:
            # Find the scenario for this combination
            scenario = next(s for s in lease_scenarios 
                          if s['rx_lease_years'] == rx_lease and s['ry_lease_years'] == ry_lease)
            move_row.append(scenario['overall_move_rate'])
            outcome_row.append(scenario['net_outcome'])
        move_rate_matrix.append(move_row)
        net_outcome_matrix.append(outcome_row)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('MOVE Rate by Lease Structure (%)', 'Net Outcome by Lease Structure'),
        horizontal_spacing=0.15
    )
    
    # Heatmap 1: MOVE Rates
    fig.add_trace(
        go.Heatmap(
            z=move_rate_matrix,
            x=[f"{l}yr" for l in lease_lengths],
            y=[f"{l}yr" for l in lease_lengths],
            colorscale='RdYlGn',
            colorbar=dict(title="MOVE Rate (%)", x=0.45),
            hovertemplate='Rx: %{y} lease<br>Ry: %{x} lease<br>MOVE Rate: %{z:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Heatmap 2: Net Outcomes
    fig.add_trace(
        go.Heatmap(
            z=net_outcome_matrix,
            x=[f"{l}yr" for l in lease_lengths],
            y=[f"{l}yr" for l in lease_lengths],
            colorscale='RdYlGn',
            colorbar=dict(title="Net Outcome", x=1.02),
            hovertemplate='Rx: %{y} lease<br>Ry: %{x} lease<br>Net Outcome: %{z:+,.0f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Comprehensive Lease Structure Analysis<br>' +
                 f'Rx: starts {rx_min} (+{rx_hike_rate*100:.3f}%/yr) | ' +
                 f'Ry: starts {ry_min} (+{ry_hike_rate*100:.3f}%/yr) | ' +
                 f'{analysis_timeframe} years',
            x=0.5,
            font=dict(size=14)
        ),
        height=500,
        width=1000
    )
    
    # Update axes
    fig.update_xaxes(title_text="Ry Lease Duration", row=1, col=1)
    fig.update_yaxes(title_text="Rx Lease Duration", row=1, col=1)
    fig.update_xaxes(title_text="Ry Lease Duration", row=1, col=2)
    fig.update_yaxes(title_text="Rx Lease Duration", row=1, col=2)
    
    return fig


def print_comprehensive_lease_analysis(lease_scenarios, rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe):
    """Print comprehensive analysis of all lease structure combinations"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE LEASE STRUCTURE ANALYSIS")
    print("="*100)
    
    # Sort scenarios by net outcome (best to worst)
    sorted_scenarios = sorted(lease_scenarios, key=lambda x: x['net_outcome'], reverse=True)
    
    print(f"{'Rank':>4} {'Rx Lease':>8} {'Ry Lease':>8} {'Move%':>7} {'Net Outcome':>12} {'Total Combos':>12}")
    print("-" * 100)
    
    for i, scenario in enumerate(sorted_scenarios, 1):
        print(f"{i:>4d} {scenario['rx_lease_years']:>7}yr {scenario['ry_lease_years']:>7}yr "
              f"{scenario['overall_move_rate']:>6.1f}% {scenario['net_outcome']:>+11,.0f} "
              f"{scenario['total_combinations']:>11,d}")
    
    # Best and worst scenarios
    best_scenario = sorted_scenarios[0]
    worst_scenario = sorted_scenarios[-1]
    
    print("\n" + "="*100)
    print("KEY INSIGHTS:")
    print("="*100)
    
    print(f"BEST LEASE STRUCTURE:")
    print(f"  Rx: {best_scenario['rx_lease_years']}-year lease, Ry: {best_scenario['ry_lease_years']}-year lease")
    print(f"  MOVE Rate: {best_scenario['overall_move_rate']:.1f}%")
    print(f"  Net Outcome: {best_scenario['net_outcome']:+,.0f}")
    print(f"  Final Ranges: Rx {best_scenario['final_rx_range']}, Ry {best_scenario['final_ry_range']}")
    
    print(f"\nWORST LEASE STRUCTURE:")
    print(f"  Rx: {worst_scenario['rx_lease_years']}-year lease, Ry: {worst_scenario['ry_lease_years']}-year lease")
    print(f"  MOVE Rate: {worst_scenario['overall_move_rate']:.1f}%")
    print(f"  Net Outcome: {worst_scenario['net_outcome']:+,.0f}")
    print(f"  Final Ranges: Rx {worst_scenario['final_rx_range']}, Ry {worst_scenario['final_ry_range']}")
    
    # Overall statistics across ALL lease structures
    total_move_decisions = sum(s['move_count'] for s in lease_scenarios)
    total_all_decisions = sum(s['total_combinations'] for s in lease_scenarios)
    overall_move_rate = total_move_decisions / total_all_decisions * 100
    
    total_all_savings = sum(s['total_move_savings'] for s in lease_scenarios)
    total_all_damages = sum(s['total_stay_damages'] for s in lease_scenarios)
    grand_net_outcome = total_all_savings + total_all_damages
    
    print(f"\nGRAND TOTALS ACROSS ALL LEASE STRUCTURES:")
    print(f"  Total Decisions Analyzed: {total_all_decisions:,}")
    print(f"  Overall MOVE Rate: {overall_move_rate:.1f}% ({total_move_decisions:,} decisions)")
    print(f"  Grand Net Outcome: {grand_net_outcome:+,.0f}")
    
    if overall_move_rate > 50:
        print(f"  → UNIVERSAL RECOMMENDATION: MOVE (favorable across {overall_move_rate:.1f}% of all scenarios)")
    else:
        print(f"  → UNIVERSAL RECOMMENDATION: STAY (favorable across {100-overall_move_rate:.1f}% of all scenarios)")
    
    # Show scenarios that favor MOVE vs STAY
    move_favorable = [s for s in lease_scenarios if s['overall_move_rate'] > 50]
    stay_favorable = [s for s in lease_scenarios if s['overall_move_rate'] <= 50]
    
    print(f"\nLEASE STRUCTURES FAVORING MOVE: {len(move_favorable)}/25")
    if move_favorable:
        for scenario in sorted(move_favorable, key=lambda x: x['overall_move_rate'], reverse=True)[:3]:
            print(f"  Rx {scenario['rx_lease_years']}yr, Ry {scenario['ry_lease_years']}yr: {scenario['overall_move_rate']:.1f}% MOVE")
    
    print(f"\nLEASE STRUCTURES FAVORING STAY: {len(stay_favorable)}/25")
    if stay_favorable:
        for scenario in sorted(stay_favorable, key=lambda x: x['overall_move_rate'])[:3]:
            print(f"  Rx {scenario['rx_lease_years']}yr, Ry {scenario['ry_lease_years']}yr: {scenario['overall_move_rate']:.1f}% MOVE")


def get_comprehensive_inputs():
    """Get inputs for comprehensive lease structure analysis"""
    print("\n" + "="*60)
    print("COMPREHENSIVE LEASE STRUCTURE ANALYSIS")
    print("="*60)
    print("This will analyze ALL combinations of lease structures (1-5 years each)")
    
    try:
        rx_min = int(input("Enter Rx starting minimum (e.g., 510): "))
        ry_min = int(input("Enter Ry starting minimum (e.g., 550): "))
        
        rx_hike_input = input("Enter annual Rx hike percentage (e.g., 6.125): ")
        rx_hike_rate = float(rx_hike_input) / 100.0
        
        ry_hike_input = input("Enter annual Ry hike percentage (e.g., 6.125): ")
        ry_hike_rate = float(ry_hike_input) / 100.0
        
        analysis_timeframe = int(input("Enter analysis timeframe in years (e.g., 5): "))
        
        return rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe
        
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return None, None, None, None, None


def print_lease_structure_summary(yearly_summaries, all_results):
    """Print detailed lease structure analysis summary"""
    print("\n" + "="*80)
    print("LEASE STRUCTURE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"{'Year':>4} {'Move%':>7} {'Rx Range':>12} {'Ry Range':>12} {'MoveAvg':>9} {'StayAvg':>9} {'Resets':>8}")
    print("-" * 80)
    
    for s in yearly_summaries:
        resets = []
        if s['rx_resets']: resets.append('Rx')
        if s['ry_resets']: resets.append('Ry')
        reset_str = '+'.join(resets) if resets else '-'
        
        print(f"{s['year']:>4d} {s['move_rate']:>6.1f}% {s['rx_range']:>12} {s['ry_range']:>12} "
              f"{s['avg_move_savings']:>9.0f} {s['avg_stay_damages']:>9.0f} {reset_str:>8}")
    
    # Overall summary
    all_move_count = sum(1 for r in all_results if r['decision'] == 'MOVE')
    total_combinations = len(all_results)
    overall_move_rate = all_move_count / total_combinations * 100
    
    all_move_savings = sum(r['avg_outcome'] for r in all_results if r['decision'] == 'MOVE')
    all_stay_damages = sum(r['avg_outcome'] for r in all_results if r['decision'] == 'STAY')
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY:")
    print("="*80)
    print(f"Total Combinations Analyzed: {total_combinations:,}")
    print(f"Overall MOVE Rate: {overall_move_rate:.1f}% ({all_move_count:,} combinations)")
    print(f"Overall STAY Rate: {100-overall_move_rate:.1f}% ({total_combinations-all_move_count:,} combinations)")
    print(f"Total MOVE Expected Savings: {all_move_savings:+,.0f}")
    print(f"Total STAY Expected Damages: {all_stay_damages:+,.0f}")
    print(f"Net Expected Outcome: {all_move_savings + all_stay_damages:+,.0f}")
    
    if overall_move_rate > 50:
        print(f"→ RECOMMENDATION: MOVE (favorable in {overall_move_rate:.1f}% of scenarios)")
    else:
        print(f"→ RECOMMENDATION: STAY (favorable in {100-overall_move_rate:.1f}% of scenarios)")


def analyze_lease_scenario(lease_years, rx_values, ry_values, n_samples=1000):
    """
    Analyze lease scenario with user-provided Rx/Ry values and sampled Hx/Hy values
    
    Parameters:
    - lease_years: int, number of years in lease
    - rx_values: list of Rx values for each year
    - ry_values: list of Ry values for each year  
    - n_samples: number of Hx/Hy combinations to sample
    
    Returns:
    - Decision analysis results for each year
    """
    if len(rx_values) != lease_years or len(ry_values) != lease_years:
        raise ValueError("Number of Rx and Ry values must match lease years")
    
    # Set up H value sampling (same weights as main analysis)
    np.random.seed(42)
    hx_values = np.array(range(560, 1001, 5))
    hy_values = np.array(range(500, 651, 5))
    
    # Bell curve weights for H values
    hx_center = len(hx_values) // 2
    hx_weights = np.exp(-0.5 * ((np.arange(len(hx_values)) - hx_center) / (len(hx_values) / 4)) ** 2)
    hx_weights = hx_weights / hx_weights.sum()
    
    hy_center = len(hy_values) // 2
    hy_weights = np.exp(-0.5 * ((np.arange(len(hy_values)) - hy_center) / (len(hy_values) / 4)) ** 2)
    hy_weights = hy_weights / hy_weights.sum()
    
    yearly_results = []
    
    for year in range(lease_years):
        rx = rx_values[year]
        ry = ry_values[year]
        
        # Sample H values for this year
        move_count = 0
        stay_count = 0
        total_savings = 0
        s_values = []
        
        for _ in range(n_samples):
            hx = np.random.choice(hx_values, p=hx_weights)
            hy = np.random.choice(hy_values, p=hy_weights)
            
            # Calculate decision score with 12x multiplier for R values
            s = (12*rx + hx) - (12*ry + hy)
            s_values.append(s)
            
            if s > 0:  # MOVE
                move_count += 1
                # Savings = what you save by moving (positive s means current is more expensive)
                total_savings += s
            else:  # STAY
                stay_count += 1
                # Damage = what you lose by staying (negative s means potential savings missed)
                total_savings += s  # This will be negative for damages
        
        # Calculate average outcome
        move_probability = move_count / n_samples
        avg_outcome = total_savings / n_samples
        
        # Determine predominant decision
        decision = "MOVE" if move_probability >= 0.5 else "STAY"
        
        yearly_results.append({
            'year': year + 1,
            'rx': rx,
            'ry': ry,
            'move_probability': move_probability,
            'decision': decision,
            'avg_annual_outcome': avg_outcome,
            's_mean': np.mean(s_values),
            's_std': np.std(s_values),
            'move_count': move_count,
            'stay_count': stay_count
        })
    
    return yearly_results


def print_lease_analysis(results):
    """Print formatted lease analysis results"""
    print("\n" + "="*80)
    print("LEASE SCENARIO ANALYSIS")
    print("="*80)
    
    print(f"{'Year':>4} {'Rx':>6} {'Ry':>6} {'Decision':>8} {'Move%':>6} {'Avg Outcome':>12} {'Mean s':>8}")
    print("-" * 80)
    
    total_outcome = 0
    for result in results:
        outcome_str = f"{result['avg_annual_outcome']:+.0f}"
        print(f"{result['year']:>4d} {result['rx']:>6} {result['ry']:>6} "
              f"{result['decision']:>8} {result['move_probability']*100:>5.1f}% "
              f"{outcome_str:>12} {result['s_mean']:>8.1f}")
        total_outcome += result['avg_annual_outcome']
    
    print("-" * 80)
    print(f"{'TOTAL':>26} {total_outcome:+12.0f}")
    print()
    
    # Summary
    move_years = sum(1 for r in results if r['decision'] == 'MOVE')
    stay_years = len(results) - move_years
    
    print(f"SUMMARY:")
    print(f"  Years recommending MOVE: {move_years}")
    print(f"  Years recommending STAY: {stay_years}")
    print(f"  Total expected outcome over {len(results)} years: {total_outcome:+.0f}")
    print(f"  Average annual outcome: {total_outcome/len(results):+.1f}")
    
    if total_outcome > 0:
        print(f"  → Overall recommendation: MOVE (expected savings)")
    else:
        print(f"  → Overall recommendation: STAY (expected damages if moving)")


def get_user_lease_inputs():
    """Interactive function to get lease parameters from user"""
    print("\n" + "="*60)
    print("LEASE SCENARIO INPUT")
    print("="*60)
    
    try:
        lease_years = int(input("Enter number of lease years: "))
        
        # Ask for rent hike options
        rx_hike_choice = input("Do you want to apply annual rent hike to Rx? (y/n): ").lower().strip()
        rx_hike_rate = 0.0
        
        if rx_hike_choice in ['y', 'yes']:
            rx_hike_input = input("Enter annual Rx rent hike percentage (e.g., 6.125 for 6.125%): ")
            rx_hike_rate = float(rx_hike_input) / 100.0
            print(f"Annual Rx rent hike set to: {rx_hike_rate*100:.3f}%")
        
        ry_hike_choice = input("Do you want to apply annual rent hike to Ry? (y/n): ").lower().strip()
        ry_hike_rate = 0.0
        
        if ry_hike_choice in ['y', 'yes']:
            ry_hike_input = input("Enter annual Ry rent hike percentage (e.g., 4.5 for 4.5%): ")
            ry_hike_rate = float(ry_hike_input) / 100.0
            print(f"Annual Ry rent hike set to: {ry_hike_rate*100:.3f}%")
        
        rx_values = []
        ry_values = []
        
        if rx_hike_rate > 0 or ry_hike_rate > 0:
            # Get initial values and calculate subsequent years with compound hikes
            print(f"\nEnter initial rent values:")
            if rx_hike_rate > 0 and ry_hike_rate > 0:
                print("Note: Both Rx and Ry will be calculated with compound rent hikes.")
            elif rx_hike_rate > 0:
                print("Note: Rx will be calculated with compound rent hike, Ry values are independent.")
            else:
                print("Note: Ry will be calculated with compound rent hike, Rx values are independent.")
            
            # Get Year 1 values
            print(f"Year 1:")
            rx_initial = int(input(f"  Rx (initial current rent): "))
            ry_initial = int(input(f"  Ry (initial alternative rent): "))
            rx_values.append(rx_initial)
            ry_values.append(ry_initial)
            
            # Calculate subsequent years with compound hikes
            current_rx = rx_initial
            current_ry = ry_initial
            
            for year in range(1, lease_years):
                print(f"Year {year + 1}:")
                
                # Calculate Rx
                if rx_hike_rate > 0:
                    current_rx = current_rx * (1 + rx_hike_rate)
                    print(f"  Rx (with {rx_hike_rate*100:.3f}% hike): {current_rx:.0f}")
                    rx_values.append(int(current_rx))
                else:
                    # No hike, use same value as previous year
                    print(f"  Rx (no hike): {current_rx}")
                    rx_values.append(current_rx)
                
                # Calculate Ry
                if ry_hike_rate > 0:
                    current_ry = current_ry * (1 + ry_hike_rate)
                    print(f"  Ry (with {ry_hike_rate*100:.3f}% hike): {current_ry:.0f}")
                    ry_values.append(int(current_ry))
                else:
                    # No hike, use same value as previous year
                    print(f"  Ry (no hike): {current_ry}")
                    ry_values.append(current_ry)
                
        else:
            # Manual entry for each year (original behavior)
            print(f"\nEnter Rx and Ry values for each year:")
            for year in range(lease_years):
                print(f"Year {year + 1}:")
                rx = int(input(f"  Rx (current rent): "))
                ry = int(input(f"  Ry (alternative rent): "))
                rx_values.append(rx)
                ry_values.append(ry)
        
        # Show summary
        print(f"\nLEASE SUMMARY:")
        print(f"{'Year':>4} {'Rx':>8} {'Ry':>8} {'Rx-Ry':>8}")
        print("-" * 32)
        for i, (rx, ry) in enumerate(zip(rx_values, ry_values)):
            print(f"{i+1:>4d} {rx:>8d} {ry:>8d} {rx-ry:>8d}")
        
        return lease_years, rx_values, ry_values
        
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return None, None, None


def main():
    """Main execution function"""
    
    # Generate weighted combinations
    print("Generating weighted parameter combinations...")
    df = generate_weighted_combinations(n_samples=5000)
    
    # Print statistics
    print_statistics(df)
    
    # Create and show plot
    print("\nCreating parallel coordinates plot...")
    fig = create_parallel_coordinates_plot(df)
    fig.show()
    
    # Save data
    print("\nSaving data to 'decision_combinations.csv'...")
    df.to_csv('decision_combinations.csv', index=False)
    
    # Run threshold analyses
    analyze_d_threshold_effects()
    analyze_f_threshold_effects()
    
    # Ask user what type of analysis they want
    print("\n" + "="*60)
    print("ANALYSIS OPTIONS:")
    print("1. Specific lease scenario (manual rent input)")
    print("2. Lease structure analysis (single lease structure with hikes)")
    print("3. Comprehensive analysis (ALL lease structures with hikes)")
    print("="*60)
    
    user_choice = input("Choose analysis type (1, 2, or 3), or 'n' to skip: ").lower().strip()
    
    if user_choice == '1':
        lease_years, rx_values, ry_values = get_user_lease_inputs()
        if lease_years is not None:
            print(f"\nAnalyzing {lease_years}-year lease scenario...")
            lease_results = analyze_lease_scenario(lease_years, rx_values, ry_values)
            print_lease_analysis(lease_results)
            
    elif user_choice == '2':
        lease_inputs = get_lease_structure_inputs()
        if lease_inputs[0] is not None:
            rx_min, ry_min, rx_hike_rate, ry_hike_rate, rx_lease_years, ry_lease_years, analysis_timeframe = lease_inputs
            
            print(f"\nAnalyzing lease structure scenarios...")
            print(f"Rx: starts {rx_min}, {rx_hike_rate*100:.3f}%/yr hike, {rx_lease_years}-year lease")
            print(f"Ry: starts {ry_min}, {ry_hike_rate*100:.3f}%/yr hike, {ry_lease_years}-year lease") 
            print(f"Analysis timeframe: {analysis_timeframe} years")
            print("This may take a moment...")
            
            all_results, yearly_summaries, yearly_ranges = analyze_lease_structure_scenarios(
                rx_min, ry_min, rx_hike_rate, ry_hike_rate, rx_lease_years, ry_lease_years, analysis_timeframe
            )
            
            # Print summary
            print_lease_structure_summary(yearly_summaries, all_results)
            
            # Create and show graph
            print("\nCreating lease structure dashboard...")
            dashboard_fig = create_lease_structure_dashboard(
                all_results, yearly_summaries, yearly_ranges, 
                rx_hike_rate, ry_hike_rate, rx_lease_years, ry_lease_years, analysis_timeframe
            )
            dashboard_fig.show()
            
    elif user_choice == '3':
        comp_inputs = get_comprehensive_inputs()
        if comp_inputs[0] is not None:
            rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe = comp_inputs
            
            print(f"\nRunning comprehensive lease structure analysis...")
            print(f"Parameters: Rx starts {rx_min} (+{rx_hike_rate*100:.3f}%/yr), Ry starts {ry_min} (+{ry_hike_rate*100:.3f}%/yr)")
            print(f"Analyzing 25 lease structure combinations over {analysis_timeframe} years")
            print("This will take several moments...")
            
            all_lease_scenarios = analyze_all_lease_structures(
                rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe
            )
            
            # Print comprehensive summary
            print_comprehensive_lease_analysis(
                all_lease_scenarios, rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe
            )
            
            # Create and show heatmap
            print("\nCreating lease structure heatmap...")
            heatmap_fig = create_lease_structure_heatmap(
                all_lease_scenarios, rx_min, ry_min, rx_hike_rate, ry_hike_rate, analysis_timeframe
            )
            heatmap_fig.show()
    
    return df

if __name__ == "__main__":
    df = main()
    
    print("\nAnalysis completed! Check 'decision_combinations.csv' for detailed data.")
