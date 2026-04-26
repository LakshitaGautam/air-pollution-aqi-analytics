import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("AQI.csv")

# Parse dates and extract features
df['last_update'] = pd.to_datetime(df['last_update'])
df['year'] = df['last_update'].dt.year
df['month'] = df['last_update'].dt.month
df['month_name'] = df['last_update'].dt.strftime('%B')
df['quarter'] = df['last_update'].dt.quarter

# ============================================================
# BASIC ANALYSIS
# ============================================================

year_trend = df.groupby('year')['pollutant_avg'].mean().reset_index()
pollutant_avg = df.groupby('pollutant_id')['pollutant_avg'].mean().reset_index().sort_values('pollutant_avg', ascending=True)

# Monthly trends
monthly_avg = df.groupby(['year', 'month', 'month_name'])['pollutant_avg'].mean().reset_index()
monthly_pivot = monthly_avg.pivot(index='month_name', columns='year', values='pollutant_avg').fillna(0)

# ============================================================
# STATE-WISE ANALYSIS
# ============================================================

state_avg = df.groupby('state')['pollutant_avg'].mean().reset_index()
top_states = state_avg.sort_values(by='pollutant_avg', ascending=False).head(15)
bottom_states = state_avg.sort_values(by='pollutant_avg', ascending=True).head(10)

# ============================================================
# REGION-WISE ANALYSIS
# ============================================================

# Region mapping
north = ["Delhi","Uttar Pradesh","Haryana","Punjab","Rajasthan",
         "Uttarakhand","Himachal Pradesh","Jammu and Kashmir",
         "Ladakh","Chandigarh"]

south = ["Tamil Nadu","Karnataka","Kerala","Andhra Pradesh",
         "Telangana","Puducherry"]

east = ["Bihar","West Bengal","Jharkhand","Odisha"]

west = ["Maharashtra","Gujarat","Goa",
        "Dadra and Nagar Haveli and Daman and Diu"]

central = ["Madhya Pradesh","Chhattisgarh"]

northeast = ["Assam","Arunachal Pradesh","Manipur",
             "Meghalaya","Mizoram","Nagaland",
             "Tripura","Sikkim"]             

islands = ["Andaman and Nicobar Islands","Lakshadweep"]

# Create region mapping dictionary
region_map = {}
for state in north: region_map[state] = "North"
for state in south: region_map[state] = "South"
for state in east: region_map[state] = "East"
for state in west: region_map[state] = "West"
for state in central: region_map[state] = "Central"
for state in northeast: region_map[state] = "Northeast"
for state in islands: region_map[state] = "Islands"

# Add region column
df['region'] = df['state'].map(region_map)

# Calculate region averages
region_avg = df.groupby('region')['pollutant_avg'].mean().reset_index().sort_values('pollutant_avg', ascending=False)

# ============================================================
# CREATE BASIC PLOTLY FIGURES (FIXED - ADDED MISSING FIGURES)
# ============================================================

CHART_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e2e8f0', size=13, family='Georgia, Times New Roman, serif'),
    title_font=dict(size=18, color='#f1f5f9', family='Georgia, Times New Roman, serif'),
    margin=dict(l=40, r=20, t=50, b=60),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
)

AXIS_STYLE = dict(gridcolor='rgba(148,163,184,0.1)', color='#94a3b8')
AXIS_LAYOUT = dict(xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)

# 1. Overall Trend with gradient
fig_basic_trend = px.line(year_trend, x='year', y='pollutant_avg',
                          markers=True,
                          title="📈 National Pollution Trend (2022-2025)",
                          color_discrete_sequence=["#FF6B6B"])
fig_basic_trend.update_traces(line=dict(width=4), marker=dict(size=12))
fig_basic_trend.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# 2. Pollutant Contribution - Horizontal bar (FIXED - was missing)
fig_pollutant = px.bar(pollutant_avg,
                       x='pollutant_avg',
                       y='pollutant_id',
                       orientation='h',
                       title="🧪 Pollutant Concentration Levels (All India)",
                       color='pollutant_avg',
                       color_continuous_scale=["#4ECDC4", "#45B7D1", "#96CEB4"])
fig_pollutant.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# 3. State Analysis - Top polluted
fig_state = px.bar(top_states,
                   x='state',
                   y='pollutant_avg',
                   title="🔥 Top 15 Most Polluted States (All India)",
                   color='pollutant_avg',
                   color_continuous_scale=["#FFE66D", "#FF6B6B", "#C44569"])
fig_state.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# 4. Region Analysis
colors_regions = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D', '#A8E6CF', '#FF8E53']
fig_region = px.bar(region_avg,
                    x='region',
                    y='pollutant_avg',
                    title="🗺️ Regional Pollution Comparison (7 Regions)",
                    color='region',
                    color_discrete_sequence=colors_regions)
fig_region.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT, showlegend=False)

# 5. Cleanest States (FIXED - was missing)
fig_clean = px.bar(bottom_states,
                   x='state',
                   y='pollutant_avg',
                   title="🍃 Top 10 Cleanest States (All India)",
                   color='pollutant_avg',
                   color_continuous_scale=["#96CEB4", "#4ECDC4", "#45B7D1"])
fig_clean.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# 6. Monthly Heatmap
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pivot = monthly_pivot.reindex(month_order)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=monthly_pivot.values,
    x=monthly_pivot.columns,
    y=monthly_pivot.index,
    colorscale='Viridis',
    text=np.round(monthly_pivot.values, 1),
    texttemplate='%{text}',
    textfont={"size": 10, "color": "white"},
    hoverongaps=False))

fig_heatmap.update_layout(
    title="📅 Monthly Pollution Heatmap (2022-2025)",
    **CHART_LAYOUT,
    **AXIS_LAYOUT
)

# ============================================================
# PREDICTION MODELS FOR 2026
# ============================================================

print("🔄 Training prediction models for 2026...")

# Prepare yearly data for prediction
yearly_state_data = df.groupby(['state', 'year'])['pollutant_avg'].mean().reset_index()
yearly_region_data = df.groupby(['region', 'year'])['pollutant_avg'].mean().reset_index()
yearly_national_data = df.groupby('year')['pollutant_avg'].mean().reset_index()

# Function to predict using linear regression
def predict_2026(yearly_data, group_col=None, group_value=None):
    predictions = []
    
    if group_col:
        # Filter for specific group (state or region)
        group_data = yearly_data[yearly_data[group_col] == group_value]
    else:
        # National level
        group_data = yearly_data
    
    if len(group_data) < 2:  # Need at least 2 years for prediction
        return None
    
    X = group_data['year'].values.reshape(-1, 1)
    y = group_data['pollutant_avg'].values
    
    # Try polynomial regression for better accuracy
    try:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        X_2026 = poly.transform([[2026]])
        pred = model.predict(X_2026)[0]
    except:
        # Fallback to linear regression
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[2026]])[0]
    
    # Ensure prediction is non-negative
    return max(0, pred)

# ============================================================
# STATE-WISE PREDICTIONS FOR 2026
# ============================================================

state_predictions = []
states = df['state'].unique()

for state in states:
    pred_value = predict_2026(yearly_state_data, 'state', state)
    if pred_value is not None:
        # Get historical avg
        hist_avg = yearly_state_data[yearly_state_data['state'] == state]['pollutant_avg'].mean()
        state_predictions.append({
            'state': state,
            'region': region_map.get(state, 'Unknown'),
            'historical_avg': round(hist_avg, 1),
            'predicted_2026': round(pred_value, 1),
            'change': round(((pred_value - hist_avg) / hist_avg) * 100, 1) if hist_avg > 0 else 0,
            'trend': 'Increasing' if pred_value > hist_avg else 'Decreasing' if pred_value < hist_avg else 'Stable'
        })

state_pred_df = pd.DataFrame(state_predictions).sort_values('predicted_2026', ascending=False)

# ============================================================
# REGION-WISE PREDICTIONS FOR 2026
# ============================================================

region_predictions = []
regions = df['region'].unique()

for region in regions:
    if pd.notna(region):
        pred_value = predict_2026(yearly_region_data, 'region', region)
        if pred_value is not None:
            hist_avg = yearly_region_data[yearly_region_data['region'] == region]['pollutant_avg'].mean()
            region_predictions.append({
                'region': region,
                'historical_avg': round(hist_avg, 1),
                'predicted_2026': round(pred_value, 1),
                'change': round(((pred_value - hist_avg) / hist_avg) * 100, 1) if hist_avg > 0 else 0,
                'trend': 'Increasing' if pred_value > hist_avg else 'Decreasing' if pred_value < hist_avg else 'Stable'
            })

region_pred_df = pd.DataFrame(region_predictions).sort_values('predicted_2026', ascending=False)

# ============================================================
# NATIONAL PREDICTION FOR 2026
# ============================================================

national_pred = predict_2026(yearly_national_data)
national_hist = yearly_national_data['pollutant_avg'].mean()
national_change = ((national_pred - national_hist) / national_hist) * 100 if national_hist > 0 else 0

# ============================================================
# CREATE PREDICTION VISUALIZATIONS
# ============================================================

# 1. National Trend with Prediction
years_extended = list(range(2022, 2027))
values_extended = list(yearly_national_data['pollutant_avg']) + [national_pred]

fig_national_pred = go.Figure()
fig_national_pred.add_trace(go.Scatter(
    x=years_extended[:-1], y=values_extended[:-1],
    mode='lines+markers', name='Historical',
    line=dict(color='#4ECDC4', width=4),
    marker=dict(size=10)
))
fig_national_pred.add_trace(go.Scatter(
    x=[2025, 2026], y=[values_extended[-2], values_extended[-1]],
    mode='lines+markers', name='Prediction',
    line=dict(color='#FF6B6B', width=4, dash='dash'),
    marker=dict(size=12, symbol='diamond')
))
fig_national_pred.update_layout(
    title="📈 National Pollution Trend with 2026 Prediction",
    **CHART_LAYOUT,
    xaxis=dict(gridcolor='rgba(148,163,184,0.1)', color='#94a3b8', tickmode='linear', tick0=2022, dtick=1),
    yaxis=dict(gridcolor='rgba(148,163,184,0.1)', color='#94a3b8', title='AQI'),
)

# 2. State Predictions Top 20
top_pred_states = state_pred_df.head(20)
fig_state_pred = px.bar(top_pred_states,
                        x='state', y='predicted_2026',
                        color='trend',
                        title="🔮 Top 20 States by Predicted AQI (2026)",
                        text='predicted_2026',
                        color_discrete_map={'Increasing': '#FF6B6B', 'Decreasing': '#96CEB4', 'Stable': '#FFE66D'})
fig_state_pred.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# 3. Region Predictions
fig_region_pred = px.bar(region_pred_df,
                         x='region', y='predicted_2026',
                         color='trend',
                         title="🌍 Region-wise AQI Predictions for 2026",
                         text='predicted_2026',
                         color_discrete_map={'Increasing': '#FF6B6B', 'Decreasing': '#96CEB4', 'Stable': '#FFE66D'})
fig_region_pred.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# 4. State Prediction vs Historical Comparison
fig_state_comparison = go.Figure()
fig_state_comparison.add_trace(go.Bar(
    name='Historical Avg (2022-2025)',
    x=top_pred_states['state'].head(10),
    y=top_pred_states['historical_avg'].head(10),
    marker_color='#4ECDC4'
))
fig_state_comparison.add_trace(go.Bar(
    name='Predicted 2026',
    x=top_pred_states['state'].head(10),
    y=top_pred_states['predicted_2026'].head(10),
    marker_color='#FF6B6B'
))
fig_state_comparison.update_layout(
    title="📊 Top 10 States: Historical vs Predicted (2026)",
    barmode='group',
    **CHART_LAYOUT,
    **AXIS_LAYOUT
)

# ============================================================
# REGION-WISE TOP/BOTTOM STATES ANALYSIS
# ============================================================

# Create dataframes for top polluted and cleanest states in each region
region_top_states = []
region_bottom_states = []

for region in regions:
    if pd.notna(region):
        region_data = df[df['region'] == region]
        state_avg_region = region_data.groupby('state')['pollutant_avg'].mean().reset_index()
        
        # Get top 3 most polluted in region
        top_3 = state_avg_region.nlargest(3, 'pollutant_avg')
        top_3['region'] = region
        top_3['category'] = 'Most Polluted'
        region_top_states.append(top_3)
        
        # Get top 3 cleanest in region
        bottom_3 = state_avg_region.nsmallest(3, 'pollutant_avg')
        bottom_3['region'] = region
        bottom_3['category'] = 'Cleanest'
        region_bottom_states.append(bottom_3)

region_top_df = pd.concat(region_top_states, ignore_index=True)
region_bottom_df = pd.concat(region_bottom_states, ignore_index=True)

# ============================================================
# POLLUTANT ANALYSIS BY STATE AND REGION
# ============================================================

def categorize_pollutant_level(value):
    if value > 100:
        return "High 🔴"
    elif value > 50:
        return "Medium 🟡"
    else:
        return "Low 🟢"

# State-wise pollutant analysis
state_pollutant_analysis = df.groupby(['state', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
state_pollutant_analysis['pollutant_level'] = state_pollutant_analysis['pollutant_avg'].apply(categorize_pollutant_level)

# Find highest pollutant for each state
state_highest_pollutant = state_pollutant_analysis.loc[state_pollutant_analysis.groupby('state')['pollutant_avg'].idxmax()]
state_highest_pollutant = state_highest_pollutant.sort_values('pollutant_avg', ascending=False)

# Region-wise pollutant analysis
region_pollutant_analysis = df.groupby(['region', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
region_pollutant_analysis['pollutant_level'] = region_pollutant_analysis['pollutant_avg'].apply(categorize_pollutant_level)

# Find highest pollutant for each region
region_highest_pollutant = region_pollutant_analysis.loc[region_pollutant_analysis.groupby('region')['pollutant_avg'].idxmax()]
region_highest_pollutant = region_highest_pollutant.sort_values('pollutant_avg', ascending=False)

# Create pivot table for state-pollutant matrix
state_pollutant_pivot = state_pollutant_analysis.pivot(index='state', columns='pollutant_id', values='pollutant_avg').fillna(0)

# ============================================================
# CREATE ADDITIONAL VISUALIZATIONS
# ============================================================

# Region-wise Top Polluted States
fig_region_top = px.bar(region_top_df, 
                        x='state', 
                        y='pollutant_avg',
                        color='region',
                        facet_col='region', 
                        facet_col_wrap=3,
                        title="🔥 Top 3 Most Polluted States by Region",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        height=600)
fig_region_top.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT, showlegend=False)
fig_region_top.update_xaxes(tickangle=45)

# Region-wise Cleanest States
fig_region_clean = px.bar(region_bottom_df, 
                         x='state', 
                         y='pollutant_avg',
                         color='region',
                         facet_col='region', 
                         facet_col_wrap=3,
                         title="🍃 Top 3 Cleanest States by Region",
                         color_discrete_sequence=px.colors.qualitative.Set3,
                         height=600)
fig_region_clean.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT, showlegend=False)
fig_region_clean.update_xaxes(tickangle=45)

# State-wise Highest Pollutant Chart
fig_state_pollutant = px.bar(state_highest_pollutant.head(20),
                             x='state',
                             y='pollutant_avg',
                             color='pollutant_id',
                             title="🔬 Top 20 States by Their Highest Pollutant",
                             text='pollutant_level',
                             color_discrete_sequence=px.colors.qualitative.Set3)
fig_state_pollutant.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# Pollutant Heatmap by State
fig_pollutant_heatmap = go.Figure(data=go.Heatmap(
    z=state_pollutant_pivot.values,
    x=state_pollutant_pivot.columns,
    y=state_pollutant_pivot.index,
    colorscale='RdYlGn_r',
    text=np.round(state_pollutant_pivot.values, 1),
    texttemplate='%{text}',
    textfont={"size": 8, "color": "white"},
    hoverongaps=False))

fig_pollutant_heatmap.update_layout(
    title="🔥 State-wise Pollutant Concentration Heatmap",
    **CHART_LAYOUT,
    **AXIS_LAYOUT,
    height=800
)

# ============================================================
# ADDITIONAL GRAPHS (10+ NEW CHARTS)
# ============================================================

# NEW 1: Quarterly trend per year
quarterly_data = df.groupby(['year', 'quarter'])['pollutant_avg'].mean().reset_index()
quarterly_data['period'] = quarterly_data['year'].astype(str) + ' Q' + quarterly_data['quarter'].astype(str)
fig_quarterly = px.line(quarterly_data, x='period', y='pollutant_avg', color='year',
                        markers=True, title="📆 Quarterly Pollution Trends (2022-2025)",
                        color_discrete_sequence=['#FF6B6B','#4ECDC4','#FFE66D','#A8E6CF'])
fig_quarterly.update_traces(line=dict(width=3), marker=dict(size=8))
fig_quarterly.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 2: Pollutant distribution by region (grouped bar)
region_pollutant_grouped = df.groupby(['region', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
fig_region_pollutant = px.bar(region_pollutant_grouped, x='region', y='pollutant_avg',
                              color='pollutant_id', barmode='group',
                              title="🌍 Pollutant Breakdown by Region",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
fig_region_pollutant.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 3: Box plot of pollutant distribution per region
fig_box_region = px.box(df, x='region', y='pollutant_avg', color='region',
                        title="📦 Pollution Variability by Region (Box Plot)",
                        color_discrete_sequence=colors_regions)
fig_box_region.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT, showlegend=False)

# NEW 4: Violin plot by pollutant type
fig_violin = px.violin(df, x='pollutant_id', y='pollutant_avg', color='pollutant_id',
                       box=True, points='outliers',
                       title="🎻 Pollutant Value Distribution (Violin Plot)",
                       color_discrete_sequence=px.colors.qualitative.Set2)
fig_violin.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT, showlegend=False)

# NEW 5: Scatter – city count vs avg pollution per state
city_count = df.groupby('state')['city'].nunique().reset_index()
city_count.columns = ['state', 'city_count']
state_scatter = state_avg.merge(city_count, on='state')
state_scatter['region'] = state_scatter['state'].map(region_map)
fig_scatter = px.scatter(state_scatter, x='city_count', y='pollutant_avg',
                         color='region', size='pollutant_avg', text='state',
                         title="🏙️ Cities Count vs Avg Pollution by State",
                         color_discrete_sequence=colors_regions)
fig_scatter.update_traces(textposition='top center', textfont=dict(size=8))
fig_scatter.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 6: Year-over-year change by region
region_yoy = df.groupby(['region', 'year'])['pollutant_avg'].mean().reset_index()
fig_region_yearly = px.line(region_yoy, x='year', y='pollutant_avg', color='region',
                            markers=True, title="📊 Year-over-Year Regional Trends",
                            color_discrete_sequence=colors_regions)
fig_region_yearly.update_traces(line=dict(width=3), marker=dict(size=9))
fig_region_yearly.update_layout(**CHART_LAYOUT,
    xaxis=dict(gridcolor='rgba(148,163,184,0.1)', color='#94a3b8', tickmode='linear', tick0=2022, dtick=1),
    yaxis=AXIS_STYLE)

# NEW 7: Radar/Spider chart for region pollutants
from plotly.subplots import make_subplots
region_radar_data = df.groupby(['region', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
pollutant_types = region_radar_data['pollutant_id'].unique().tolist()
fig_radar = go.Figure()
radar_colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFE66D','#A8E6CF','#FF8E53']
for i, reg in enumerate(region_avg['region']):
    vals = []
    for p in pollutant_types:
        v = region_radar_data[(region_radar_data['region']==reg) & (region_radar_data['pollutant_id']==p)]['pollutant_avg']
        vals.append(float(v.values[0]) if len(v) > 0 else 0)
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=pollutant_types + [pollutant_types[0]],
        fill='toself', name=reg,
        line=dict(color=radar_colors[i % len(radar_colors)]),
        opacity=0.7
    ))
fig_radar.update_layout(
    title="🕸️ Region Pollutant Profile (Radar Chart)",
    polar=dict(
        bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(visible=True, gridcolor='rgba(148,163,184,0.2)', color='#94a3b8'),
        angularaxis=dict(gridcolor='rgba(148,163,184,0.2)', color='#94a3b8')
    ),
    **CHART_LAYOUT
)

# NEW 8: Pie chart — region share of total pollution load
region_total = df.groupby('region')['pollutant_avg'].sum().reset_index()
fig_pie_region = px.pie(region_total, names='region', values='pollutant_avg',
                        title="🥧 Region Share of Total Pollution Load",
                        color_discrete_sequence=colors_regions, hole=0.4)
fig_pie_region.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 9: Sunburst: region → state pollution
sunburst_data = df.groupby(['region', 'state'])['pollutant_avg'].mean().reset_index()
sunburst_data['region'] = sunburst_data['region'].fillna('Unknown')
fig_sunburst = px.sunburst(sunburst_data, path=['region', 'state'], values='pollutant_avg',
                           title="☀️ Sunburst: Region → State Pollution Hierarchy",
                           color='pollutant_avg', color_continuous_scale='RdYlGn_r')
fig_sunburst.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 10: Treemap of states by pollution
treemap_data = df.groupby(['region','state'])['pollutant_avg'].mean().reset_index().dropna()
fig_treemap = px.treemap(treemap_data, path=['region','state'], values='pollutant_avg',
                         title="🗂️ Treemap: State-level Pollution by Region",
                         color='pollutant_avg', color_continuous_scale='RdYlGn_r')
fig_treemap.update_layout(**CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 11: Waterfall chart — national AQI change year to year
wf_years = yearly_national_data['year'].tolist() + [2026]
wf_values = yearly_national_data['pollutant_avg'].tolist() + [national_pred]
wf_measure = ['absolute'] + ['relative'] * (len(wf_years) - 1)
wf_delta = [wf_values[0]] + [wf_values[i]-wf_values[i-1] for i in range(1, len(wf_values))]
fig_waterfall = go.Figure(go.Waterfall(
    name="AQI Change", orientation="v",
    measure=wf_measure,
    x=[str(y) for y in wf_years],
    y=wf_delta,
    connector={"line":{"color":"rgba(148,163,184,0.3)"}},
    increasing={"marker":{"color":"#FF6B6B"}},
    decreasing={"marker":{"color":"#96CEB4"}},
    totals={"marker":{"color":"#4ECDC4"}}
))
fig_waterfall.update_layout(title="💧 Waterfall: Year-on-Year AQI Change", **CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 12: Funnel chart — states ranked by AQI (top 10)
funnel_data = state_avg.sort_values('pollutant_avg', ascending=False).head(10)
fig_funnel = go.Figure(go.Funnel(
    y=funnel_data['state'],
    x=funnel_data['pollutant_avg'],
    textposition="inside",
    textinfo="value+percent initial",
    marker=dict(color=['#FF6B6B','#FF7B7B','#FF8E8E','#FFA0A0','#FFB3B3',
                       '#FFC6C6','#FFD9D9','#FFECEC','#4ECDC4','#45B7D1'])
))
fig_funnel.update_layout(title="🔻 Funnel: Top 10 Most Polluted States", **CHART_LAYOUT, **AXIS_LAYOUT)

# NEW 13: Area chart — seasonal pollution by year
monthly_line = df.groupby(['year', 'month'])['pollutant_avg'].mean().reset_index()
fig_area = px.area(monthly_line, x='month', y='pollutant_avg', color='year',
                   title="🌊 Seasonal Pollution Patterns (Area Chart)",
                   color_discrete_sequence=['#FF6B6B','#4ECDC4','#FFE66D','#A8E6CF'],
                   labels={'month': 'Month', 'pollutant_avg': 'Avg AQI'})
fig_area.update_layout(**CHART_LAYOUT,
    xaxis=dict(gridcolor='rgba(148,163,184,0.1)', color='#94a3b8', tickmode='linear', tick0=1, dtick=1),
    yaxis=AXIS_STYLE)

# ============================================================
# STATISTICS CARDS
# ============================================================

avg_aqi = df['pollutant_avg'].mean()
max_aqi = df['pollutant_avg'].max()
max_state = (df.groupby('state')['pollutant_avg'].mean()).idxmax()
total_records = len(df)
total_cities = df['city'].nunique()
total_states = df['state'].nunique()

# Most common high pollutant
high_pollutant_counts = state_pollutant_analysis[state_pollutant_analysis['pollutant_level'].str.contains('High')]['pollutant_id'].value_counts()
most_common_high_pollutant = high_pollutant_counts.index[0] if not high_pollutant_counts.empty else "N/A"

min_state = (df.groupby('state')['pollutant_avg'].mean()).idxmin()
cleanest_region = region_avg.iloc[-1]['region'] if len(region_avg) > 0 else 'N/A'
total_pollutants = df['pollutant_id'].nunique()

# ============================================================
# BUILD HTML DASHBOARD
# ============================================================

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AQI Analytics | India Air Quality Intelligence 2022–2026</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --bg: #050a14;
            --surface: #0d1526;
            --surface2: #111e35;
            --border: rgba(99,179,237,0.12);
            --accent: #38bdf8;
            --accent2: #f472b6;
            --accent3: #34d399;
            --danger: #f87171;
            --warn: #fbbf24;
            --text: #e2e8f0;
            --muted: #64748b;
            --glow: 0 0 40px rgba(56,189,248,0.15);
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html {{ scroll-behavior: smooth; }}

        body {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* ── ANIMATED BACKGROUND ── */
        body::before {{
            content: '';
            position: fixed; inset: 0;
            background:
                radial-gradient(ellipse 80% 50% at 20% 10%, rgba(56,189,248,0.07) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(244,114,182,0.06) 0%, transparent 60%),
                radial-gradient(ellipse 50% 60% at 50% 50%, rgba(52,211,153,0.04) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
            animation: bgShift 20s ease-in-out infinite alternate;
        }}
        @keyframes bgShift {{
            0% {{ opacity: 0.6; }}
            100% {{ opacity: 1; }}
        }}

        /* ── PARTICLES ── */
        #particles {{ position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }}
        .particle {{
            position:absolute; border-radius:50%;
            animation: floatUp linear infinite;
            opacity:0;
        }}
        @keyframes floatUp {{
            0%  {{ transform: translateY(100vh) scale(0); opacity:0; }}
            10% {{ opacity: 0.6; }}
            90% {{ opacity: 0.3; }}
            100%{{ transform: translateY(-10vh) scale(1.5); opacity:0; }}
        }}

        /* ── NAVBAR ── */
        nav {{
            position: sticky; top:0; z-index:100;
            display:flex; align-items:center; justify-content:space-between;
            padding: 1rem 2.5rem;
            background: rgba(5,10,20,0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
        }}
        .brand {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-weight: 800;
            font-size: 1.35rem;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display:flex; align-items:center; gap:10px;
            letter-spacing: -0.5px;
        }}
        .brand-icon {{
            width:36px; height:36px; border-radius:10px;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            display:flex; align-items:center; justify-content:center;
            color: white; font-size:1rem;
            -webkit-text-fill-color: white;
            box-shadow: 0 0 20px rgba(56,189,248,0.4);
            animation: pulse-glow 3s ease-in-out infinite;
        }}
        @keyframes pulse-glow {{
            0%,100% {{ box-shadow: 0 0 15px rgba(56,189,248,0.4); }}
            50% {{ box-shadow: 0 0 35px rgba(56,189,248,0.8); }}
        }}
        .nav-badge {{
            background: rgba(56,189,248,0.1);
            border: 1px solid rgba(56,189,248,0.3);
            border-radius: 50px;
            padding: 4px 14px;
            font-size:0.78rem;
            color: var(--accent);
            font-weight: 500;
        }}

        /* ── HERO ── */
        .hero {{
            position:relative; z-index:1;
            text-align:center;
            padding: 5rem 2rem 3rem;
        }}
        .hero-eyebrow {{
            display:inline-flex; align-items:center; gap:8px;
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.2);
            border-radius:50px;
            padding: 6px 18px;
            font-size: 0.82rem;
            color: var(--accent);
            letter-spacing: 2px;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 1.5rem;
            animation: fadeDown 0.8s ease both;
        }}
        .hero h1 {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size: clamp(2.2rem, 5vw, 4rem);
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -2px;
            margin-bottom: 1rem;
            animation: fadeDown 0.9s ease both;
        }}
        .hero h1 span {{
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 50%, var(--accent3) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-size: 200% 200%;
            animation: gradShift 4s ease infinite;
        }}
        @keyframes gradShift {{
            0%,100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}
        .hero p {{
            color: var(--muted);
            font-size:1.05rem;
            max-width: 560px;
            margin: 0 auto 2.5rem;
            line-height: 1.7;
            animation: fadeDown 1s ease both;
        }}

        /* ── KPI GRID ── */
        .kpi-grid {{
            position:relative; z-index:1;
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(190px,1fr));
            gap:16px;
            max-width: 1400px;
            margin: 0 auto 3rem;
            padding: 0 2rem;
        }}
        .kpi {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 22px 20px;
            position:relative;
            overflow:hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            animation: fadeUp 0.7s ease both;
            cursor:default;
        }}
        .kpi::before {{
            content:'';
            position:absolute; inset:0;
            background: linear-gradient(135deg, transparent 60%, rgba(56,189,248,0.04));
            border-radius:18px;
        }}
        .kpi:hover {{
            transform: translateY(-6px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(56,189,248,0.2);
        }}
        .kpi-icon {{
            width:42px; height:42px; border-radius:12px;
            display:flex; align-items:center; justify-content:center;
            font-size:1.1rem; margin-bottom:14px;
        }}
        .kpi-val {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size:1.9rem; font-weight:800;
            letter-spacing:-1px;
            line-height:1;
            margin-bottom:6px;
        }}
        .kpi-lbl {{
            font-size:0.76rem;
            color: var(--muted);
            text-transform:uppercase;
            letter-spacing:1.5px;
            font-weight:500;
        }}
        .kpi-badge {{
            position:absolute; top:16px; right:16px;
            font-size:0.72rem;
            padding: 3px 10px; border-radius:20px;
            font-weight:600;
        }}
        .badge-danger {{ background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); }}
        .badge-success {{ background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3); }}
        .badge-info {{ background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); }}
        .badge-warn {{ background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }}

        /* ── PREDICTION HERO CARD ── */
        .pred-hero {{
            position:relative; z-index:1;
            max-width:1400px; margin:0 auto 3rem;
            padding: 0 2rem;
        }}
        .pred-hero-card {{
            background: linear-gradient(135deg, rgba(248,113,113,0.1) 0%, rgba(244,114,182,0.1) 50%, rgba(56,189,248,0.1) 100%);
            border: 1px solid rgba(248,113,113,0.25);
            border-radius:24px;
            padding: 2.5rem;
            display:flex; align-items:center; justify-content:space-between;
            gap:2rem; flex-wrap:wrap;
            position:relative; overflow:hidden;
        }}
        .pred-hero-card::before {{
            content:'';
            position:absolute; top:-50%; right:-10%;
            width:400px; height:400px; border-radius:50%;
            background: radial-gradient(circle, rgba(248,113,113,0.08), transparent 70%);
            pointer-events:none;
        }}
        .pred-main-val {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size:3.5rem; font-weight:800;
            background:linear-gradient(135deg,#f87171,#f472b6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            line-height:1; letter-spacing:-2px;
        }}
        .pred-label {{ color:var(--muted); font-size:0.85rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px; }}
        .pred-change {{
            font-size:1.1rem; font-weight:600;
            display:flex; align-items:center; gap:8px; margin-top:12px;
        }}
        .pred-meta {{ display:flex; gap:2rem; flex-wrap:wrap; }}
        .pred-meta-item {{ text-align:center; }}
        .pred-meta-val {{ font-family: Georgia, 'Times New Roman', Times, serif; font-size:1.6rem; font-weight:700; }}
        .pred-meta-lbl {{ color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; }}

        /* ── TAB SYSTEM ── */
        .tabs-wrapper {{
            position:relative; z-index:1;
            max-width:1400px; margin:0 auto;
            padding: 0 2rem 4rem;
        }}

        .tab-home {{
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap:20px;
            margin-bottom:3rem;
        }}
        .tab-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            min-height: 160px;
        }}
        .tab-box::before {{
            content: '';
            position: absolute; inset: 0;
            background: linear-gradient(135deg, transparent 60%, rgba(56,189,248,0.04));
            border-radius: 16px;
        }}
        .tab-box:hover {{
            transform: translateY(-6px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.4), 0 0 0 1px rgba(56,189,248,0.25);
            border-color: rgba(56,189,248,0.3);
        }}
        .tab-box.active-box {{
            background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(244,114,182,0.1));
            border-color: rgba(56,189,248,0.4);
            box-shadow: 0 0 40px rgba(56,189,248,0.15);
        }}
        .tab-box-icon {{
            width: 52px; height: 52px;
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem;
            margin: 0 auto;
        }}
        .tab-box-title {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size: 1rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.3px;
        }}
        .tab-box-desc {{
            font-size: 0.75rem;
            color: var(--muted);
            line-height: 1.4;
        }}
        .tab-box-count {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .tab-pane {{ display:none; animation: fadeUp 0.4s ease both; }}
        .tab-pane.active {{ display:block; }}

        /* ── CHART GRID ── */
        .chart-grid {{
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(580px,1fr));
            gap:20px; margin-bottom:20px;
        }}
        .chart-grid-3 {{
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(380px,1fr));
            gap:20px; margin-bottom:20px;
        }}
        .chart-full {{ grid-column:1/-1; }}

        .chart-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow:hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            position:relative;
        }}
        .chart-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 24px 60px rgba(0,0,0,0.35);
            border-color: rgba(56,189,248,0.2);
        }}
        .chart-card::before {{
            content:'';
            position:absolute; top:0; left:0; right:0; height:3px;
            background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
            opacity:0;
            transition: opacity 0.3s;
        }}
        .chart-card:hover::before {{ opacity:1; }}

        .chart-header {{
            padding: 18px 20px 12px;
            display:flex; align-items:center; gap:10px;
            border-bottom: 1px solid var(--border);
        }}
        .chart-header-icon {{
            width:32px; height:32px; border-radius:9px;
            display:flex; align-items:center; justify-content:center;
            font-size:0.9rem;
        }}
        .chart-header h3 {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size:0.95rem; font-weight:700;
            flex:1; letter-spacing:-0.3px;
        }}
        .chart-tag {{
            font-size:0.7rem;
            padding:2px 8px; border-radius:10px;
            font-weight:600; text-transform:uppercase; letter-spacing:1px;
        }}

        /* ── DATA TABLE ── */
        .data-table-wrap {{
            background:var(--surface); border:1px solid var(--border);
            border-radius:20px; overflow:hidden; margin-bottom:20px;
        }}
        .data-table-header {{
            padding:18px 24px; border-bottom:1px solid var(--border);
            display:flex; align-items:center; gap:10px;
        }}
        .data-table-header h3 {{ font-family: Georgia, 'Times New Roman', Times, serif; font-size:1rem; font-weight:700; }}
        .tbl-scroll {{ overflow-x:auto; max-height:440px; overflow-y:auto; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.84rem; }}
        thead tr {{
            background:var(--surface2);
            position:sticky; top:0; z-index:2;
        }}
        th {{
            padding:12px 16px; text-align:left;
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size:0.75rem; font-weight:700;
            text-transform:uppercase; letter-spacing:1.5px;
            color:var(--muted); white-space:nowrap;
        }}
        tbody tr {{ border-bottom:1px solid rgba(99,179,237,0.06); transition:background 0.2s; }}
        tbody tr:hover {{ background:rgba(56,189,248,0.04); }}
        td {{ padding:10px 16px; }}
        .pill {{
            display:inline-block; padding:3px 10px; border-radius:20px;
            font-size:0.75rem; font-weight:600;
        }}
        .pill-red {{ background:rgba(248,113,113,0.15); color:#f87171; }}
        .pill-green {{ background:rgba(52,211,153,0.15); color:#34d399; }}
        .pill-yellow {{ background:rgba(251,191,36,0.15); color:#fbbf24; }}
        .pill-blue {{ background:rgba(56,189,248,0.15); color:#38bdf8; }}
        .up {{ color:#f87171; }}
        .down {{ color:#34d399; }}
        .stable {{ color:#fbbf24; }}

        /* ── SECTION DIVIDER ── */
        .section-label {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size:0.72rem; font-weight:700;
            text-transform:uppercase; letter-spacing:3px;
            color:var(--accent); margin-bottom:12px;
            display:flex; align-items:center; gap:8px;
        }}
        .section-label::before {{
            content:''; display:block;
            width:20px; height:2px;
            background:var(--accent);
            border-radius:2px;
        }}

        /* ── FOOTER ── */
        footer {{
            position:relative; z-index:1;
            border-top: 1px solid var(--border);
            padding: 2.5rem;
            text-align:center;
            background: var(--surface);
        }}
        footer p {{ color:var(--muted); font-size:0.83rem; line-height:1.8; }}
        footer .footer-brand {{
            font-family: Georgia, 'Times New Roman', Times, serif;
            font-size:1rem; font-weight:700; color:var(--text);
            margin-bottom:8px;
        }}
        .legend-dots {{
            display:flex; align-items:center; justify-content:center;
            gap:16px; flex-wrap:wrap; margin-top:12px;
        }}
        .legend-dot {{
            display:flex; align-items:center; gap:6px;
            font-size:0.78rem; color:var(--muted);
        }}
        .dot {{ width:8px; height:8px; border-radius:50%; }}

        /* ── SCROLL PROGRESS ── */
        #scroll-progress {{
            position:fixed; top:0; left:0; height:3px; z-index:999;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            width:0%; transition: width 0.1s;
            box-shadow: 0 0 10px var(--accent);
        }}

        /* ── TOOLTIP ── */
        .tooltip-icon {{
            width:18px; height:18px; border-radius:50%;
            background:var(--surface2); border:1px solid var(--border);
            display:inline-flex; align-items:center; justify-content:center;
            font-size:0.65rem; color:var(--muted); cursor:help;
        }}

        /* ── ANIMATIONS ── */
        @keyframes fadeDown {{
            from {{ opacity:0; transform:translateY(-16px); }}
            to {{ opacity:1; transform:translateY(0); }}
        }}
        @keyframes fadeUp {{
            from {{ opacity:0; transform:translateY(20px); }}
            to {{ opacity:1; transform:translateY(0); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}

        /* ── RESPONSIVE ── */
        @media(max-width:768px) {{
            .chart-grid, .chart-grid-3 {{ grid-template-columns:1fr; }}
            .hero h1 {{ font-size:2rem; letter-spacing:-1px; }}
            .kpi-grid {{ grid-template-columns: repeat(2,1fr); }}
            .pred-hero-card {{ flex-direction:column; }}
            .tab-home {{ grid-template-columns: repeat(2,1fr); }}
        }}
    </style>
</head>
<body>

<div id="scroll-progress"></div>
<div id="particles"></div>

<!-- NAVBAR -->
<nav>
    <div class="brand">
        <div class="brand-icon"><i class="fas fa-wind"></i></div>
        AQI Intelligence
    </div>
    <div class="nav-badge">🇮🇳 India • 2022–2026</div>
</nav>

<!-- HERO -->
<section class="hero">
    <div class="hero-eyebrow"><span>●</span> Live Analytics Platform</div>
    <h1>India Air Quality<br><span>Intelligence Dashboard</span></h1>
    <p>Comprehensive analysis of air pollution across {total_states} states and {total_cities} cities — with machine learning predictions for 2026.</p>
</section>

<!-- KPI CARDS -->
<div class="kpi-grid">
    <div class="kpi" style="animation-delay:0.1s">
        <div class="kpi-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-chart-line"></i></div>
        <div class="kpi-val" style="color:#f87171;">{avg_aqi:.1f}</div>
        <div class="kpi-lbl">National Avg AQI</div>
        <span class="kpi-badge badge-danger">2022–25</span>
    </div>
    <div class="kpi" style="animation-delay:0.15s">
        <div class="kpi-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-exclamation-triangle"></i></div>
        <div class="kpi-val" style="color:#fbbf24;">{max_aqi:.0f}</div>
        <div class="kpi-lbl">Peak AQI Recorded</div>
        <span class="kpi-badge badge-warn">Peak</span>
    </div>
    <div class="kpi" style="animation-delay:0.2s">
        <div class="kpi-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-map-marker-alt"></i></div>
        <div class="kpi-val" style="color:#f472b6; font-size:1.2rem; padding-top:4px;">{max_state[:12]}</div>
        <div class="kpi-lbl">Most Polluted State</div>
        <span class="kpi-badge badge-danger">Worst</span>
    </div>
    <div class="kpi" style="animation-delay:0.25s">
        <div class="kpi-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-leaf"></i></div>
        <div class="kpi-val" style="color:#34d399; font-size:1.2rem; padding-top:4px;">{min_state[:12]}</div>
        <div class="kpi-lbl">Cleanest State & UTS</div>
        <span class="kpi-badge badge-success">Best</span>
    </div>
    <div class="kpi" style="animation-delay:0.3s">
        <div class="kpi-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-city"></i></div>
        <div class="kpi-val" style="color:#38bdf8;">{total_cities}</div>
        <div class="kpi-lbl">Cities Monitored</div>
        <span class="kpi-badge badge-info">Coverage</span>
    </div>
    <div class="kpi" style="animation-delay:0.35s">
        <div class="kpi-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-flag"></i></div>
        <div class="kpi-val" style="color:#38bdf8;">{total_states}</div>
        <div class="kpi-lbl">States & UTs</div>
        <span class="kpi-badge badge-info">All India</span>
    </div>
    <div class="kpi" style="animation-delay:0.4s">
        <div class="kpi-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-flask"></i></div>
        <div class="kpi-val" style="color:#fbbf24;">{total_pollutants}</div>
        <div class="kpi-lbl">Pollutant Types</div>
        <span class="kpi-badge badge-warn">Tracked</span>
    </div>
    <div class="kpi" style="animation-delay:0.45s">
        <div class="kpi-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-biohazard"></i></div>
        <div class="kpi-val" style="color:#f472b6;">{most_common_high_pollutant}</div>
        <div class="kpi-lbl">Top High Pollutant</div>
        <span class="kpi-badge badge-danger">Alert</span>
    </div>
</div>

<!-- PREDICTION HERO CARD -->
<div class="pred-hero">
    <div class="pred-hero-card">
        <div>
            <div class="pred-label"><i class="fas fa-robot"></i> &nbsp; ML Prediction 2026</div>
            <div class="pred-main-val">{national_pred:.1f}</div>
            <div style="color:var(--muted); margin-top:6px; font-size:0.88rem;">Predicted National Average AQI</div>
            <div class="pred-change {'up' if national_change > 0 else 'down'}">
                <i class="fas fa-{'arrow-up' if national_change > 0 else 'arrow-down'}"></i>
                <span style="color:{'#f87171' if national_change > 0 else '#34d399'};">{national_change:+.1f}% vs historical average</span>
            </div>
        </div>
        <div class="pred-meta">
            <div class="pred-meta-item">
                <div class="pred-meta-val" style="color:#38bdf8;">{len(state_predictions)}</div>
                <div class="pred-meta-lbl">States Predicted</div>
            </div>
            <div class="pred-meta-item">
                <div class="pred-meta-val" style="color:#f472b6;">{len(region_predictions)}</div>
                <div class="pred-meta-lbl">Regions Predicted</div>
            </div>
            <div class="pred-meta-item">
                <div class="pred-meta-val" style="color:#34d399;">Poly</div>
                <div class="pred-meta-lbl">Regression Model</div>
            </div>
            <div class="pred-meta-item">
                <div class="pred-meta-val" style="color:#fbbf24;">4yr</div>
                <div class="pred-meta-lbl">Training Data</div>
            </div>
        </div>
    </div>
</div>

<!-- TABS -->
<div class="tabs-wrapper">
    <div class="tab-home">
        <div class="tab-box active-box" onclick="switchTab('overview',this)">
            <div class="tab-box-icon" style="background:rgba(248,113,113,0.15);color:#f87171;"><i class="fas fa-home"></i></div>
            <div class="tab-box-title">Overview</div>
            <div class="tab-box-desc">National trends, seasonal patterns and summary</div>
            <span class="tab-box-count" style="background:rgba(248,113,113,0.12);color:#f87171;">6 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('predictions',this)">
            <div class="tab-box-icon" style="background:rgba(244,114,182,0.15);color:#f472b6;"><i class="fas fa-brain"></i></div>
            <div class="tab-box-title">Predictions</div>
            <div class="tab-box-desc">ML forecasts for 2026 at national, state and region level</div>
            <span class="tab-box-count" style="background:rgba(244,114,182,0.12);color:#f472b6;">5 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('states',this)">
            <div class="tab-box-icon" style="background:rgba(251,191,36,0.15);color:#fbbf24;"><i class="fas fa-map"></i></div>
            <div class="tab-box-title">States</div>
            <div class="tab-box-desc">Most polluted and cleanest states with pollutant breakdown</div>
            <span class="tab-box-count" style="background:rgba(251,191,36,0.12);color:#fbbf24;">5 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('regions',this)">
            <div class="tab-box-icon" style="background:rgba(52,211,153,0.15);color:#34d399;"><i class="fas fa-globe-asia"></i></div>
            <div class="tab-box-title">Regions</div>
            <div class="tab-box-desc">7-region comparison with year-over-year pollutant profiles</div>
            <span class="tab-box-count" style="background:rgba(52,211,153,0.12);color:#34d399;">8 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('pollutants',this)">
            <div class="tab-box-icon" style="background:rgba(56,189,248,0.15);color:#38bdf8;"><i class="fas fa-flask"></i></div>
            <div class="tab-box-title">Pollutants</div>
            <div class="tab-box-desc">Deep dive into pollutant types, violin plots and state matrix</div>
            <span class="tab-box-count" style="background:rgba(56,189,248,0.12);color:#38bdf8;">3 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('advanced',this)">
            <div class="tab-box-icon" style="background:rgba(168,110,255,0.15);color:#a87bff;"><i class="fas fa-chart-pie"></i></div>
            <div class="tab-box-title">Advanced</div>
            <div class="tab-box-desc">Sunburst, treemap, radar and advanced visualizations</div>
            <span class="tab-box-count" style="background:rgba(168,110,255,0.12);color:#a87bff;">6 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('timeseries',this)">
            <div class="tab-box-icon" style="background:rgba(99,179,237,0.15);color:#63b3ed;"><i class="fas fa-calendar-alt"></i></div>
            <div class="tab-box-title">Time Series</div>
            <div class="tab-box-desc">Monthly heatmap, quarterly breakdown and seasonal patterns</div>
            <span class="tab-box-count" style="background:rgba(99,179,237,0.12);color:#63b3ed;">3 Charts</span>
        </div>
        <div class="tab-box" onclick="switchTab('rankings',this)">
            <div class="tab-box-icon" style="background:rgba(251,113,133,0.15);color:#fb7185;"><i class="fas fa-trophy"></i></div>
            <div class="tab-box-title">Rankings</div>
            <div class="tab-box-desc">Funnel, scatter and ranked comparisons across all states</div>
            <span class="tab-box-count" style="background:rgba(251,113,133,0.12);color:#fb7185;">3 Charts</span>
        </div>
    </div>

    <!-- ══════════════════ OVERVIEW TAB ══════════════════ -->
    <div id="overview" class="tab-pane active">
        <div class="section-label">Core Metrics</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-chart-line"></i></div>
                    <h3>National Trend + 2026 Forecast</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Trend</span>
                </div>
                {fig_national_pred.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-flask"></i></div>
                    <h3>Pollutant Distribution</h3>
                    <span class="chart-tag" style="background:rgba(251,191,36,0.1); color:#fbbf24;">Breakdown</span>
                </div>
                {fig_pollutant.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-map"></i></div>
                    <h3>Regional Pollution Comparison</h3>
                    <span class="chart-tag" style="background:rgba(56,189,248,0.1); color:#38bdf8;">Regions</span>
                </div>
                {fig_region.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-calendar-alt"></i></div>
                    <h3>Monthly Pollution Heatmap</h3>
                    <span class="chart-tag" style="background:rgba(244,114,182,0.1); color:#f472b6;">Seasonal</span>
                </div>
                {fig_heatmap.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-chart-area"></i></div>
                    <h3>Seasonal Pollution Patterns</h3>
                    <span class="chart-tag" style="background:rgba(52,211,153,0.1); color:#34d399;">Area</span>
                </div>
                {fig_area.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-chart-line"></i></div>
                    <h3>Quarterly Trends (2022–2025)</h3>
                    <span class="chart-tag" style="background:rgba(56,189,248,0.1); color:#38bdf8;">Quarterly</span>
                </div>
                {fig_quarterly.to_html(full_html=False)}
            </div>
        </div>
    </div>

    <!-- ══════════════════ PREDICTIONS TAB ══════════════════ -->
    <div id="predictions" class="tab-pane">
        <div class="section-label">2026 ML Forecasts</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-chart-line"></i></div>
                    <h3>National Trend + Prediction</h3>
                </div>
                {fig_national_pred.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-globe"></i></div>
                    <h3>Region Predictions 2026</h3>
                </div>
                {fig_region_pred.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-map-marker-alt"></i></div>
                    <h3>Top 20 State Predictions</h3>
                </div>
                {fig_state_pred.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-chart-bar"></i></div>
                    <h3>Historical vs Predicted (Top 10)</h3>
                </div>
                {fig_state_comparison.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-water"></i></div>
                    <h3>Year-on-Year AQI Waterfall</h3>
                    <span class="chart-tag" style="background:rgba(251,191,36,0.1); color:#fbbf24;">Δ Change</span>
                </div>
                {fig_waterfall.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-filter"></i></div>
                    <h3>Pollution Funnel — Top 10 States</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Ranked</span>
                </div>
                {fig_funnel.to_html(full_html=False)}
            </div>
        </div>

        <!-- State Predictions Table -->
        <div class="data-table-wrap">
            <div class="data-table-header">
                <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-table"></i></div>
                <h3>State-wise 2026 Predictions — Top 30</h3>
                <span class="kpi-badge badge-warn" style="margin-left:auto;">ML Forecast</span>
            </div>
            <div class="tbl-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>#</th><th>State</th><th>Region</th>
                            <th>Historical (2022–25)</th><th>Predicted 2026</th>
                            <th>Change</th><th>Trend</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for i, (_, row) in enumerate(state_pred_df.head(30).iterrows(), 1):
    arrow = "▲" if row['change'] > 0 else "▼" if row['change'] < 0 else "◆"
    cls = "up" if row['change'] > 0 else "down" if row['change'] < 0 else "stable"
    trend_pill = "pill-red" if row['trend']=='Increasing' else "pill-green" if row['trend']=='Decreasing' else "pill-yellow"
    html_template += f"""
                        <tr>
                            <td style="color:var(--muted);">{i}</td>
                            <td><strong>{row['state']}</strong></td>
                            <td><span class="pill pill-blue">{row['region']}</span></td>
                            <td>{row['historical_avg']}</td>
                            <td><strong style="color:{'#f87171' if row['predicted_2026']>100 else '#fbbf24' if row['predicted_2026']>50 else '#34d399'};">{row['predicted_2026']}</strong></td>
                            <td class="{cls}">{arrow} {abs(row['change']):.1f}%</td>
                            <td><span class="pill {trend_pill}">{row['trend']}</span></td>
                        </tr>"""

html_template += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Region Predictions Table -->
        <div class="data-table-wrap">
            <div class="data-table-header">
                <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-globe-asia"></i></div>
                <h3>Region-wise 2026 Predictions</h3>
            </div>
            <div class="tbl-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Region</th><th>Historical</th><th>Predicted 2026</th>
                            <th>Change</th><th>Trend</th>
                        </tr>
                    </thead>
                    <tbody>
"""
for _, row in region_pred_df.iterrows():
    arrow = "▲" if row['change'] > 0 else "▼" if row['change'] < 0 else "◆"
    cls = "up" if row['change'] > 0 else "down" if row['change'] < 0 else "stable"
    trend_pill = "pill-red" if row['trend']=='Increasing' else "pill-green" if row['trend']=='Decreasing' else "pill-yellow"
    html_template += f"""
                        <tr>
                            <td><strong>{row['region']}</strong></td>
                            <td>{row['historical_avg']}</td>
                            <td><strong style="color:{'#f87171' if row['predicted_2026']>100 else '#fbbf24' if row['predicted_2026']>50 else '#34d399'};">{row['predicted_2026']}</strong></td>
                            <td class="{cls}">{arrow} {abs(row['change']):.1f}%</td>
                            <td><span class="pill {trend_pill}">{row['trend']}</span></td>
                        </tr>"""

html_template += f"""
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ══════════════════ STATES TAB ══════════════════ -->
    <div id="states" class="tab-pane">
        <div class="section-label">State Analysis</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-fire"></i></div>
                    <h3>Top 15 Most Polluted States</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Worst</span>
                </div>
                {fig_state.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-tree"></i></div>
                    <h3>Top 10 Cleanest States</h3>
                    <span class="chart-tag" style="background:rgba(52,211,153,0.1); color:#34d399;">Best</span>
                </div>
                {fig_clean.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-flask"></i></div>
                    <h3>States by Dominant Pollutant</h3>
                </div>
                {fig_state_pollutant.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-bullseye"></i></div>
                    <h3>Cities vs Pollution Scatter</h3>
                    <span class="chart-tag" style="background:rgba(56,189,248,0.1); color:#38bdf8;">Scatter</span>
                </div>
                {fig_scatter.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-filter"></i></div>
                    <h3>Top 10 States Pollution Funnel</h3>
                </div>
                {fig_funnel.to_html(full_html=False)}
            </div>
            <div class="chart-card chart-full">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-th"></i></div>
                    <h3>State × Pollutant Heatmap</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Matrix</span>
                </div>
                {fig_pollutant_heatmap.to_html(full_html=False)}
            </div>
        </div>
    </div>

    <!-- ══════════════════ REGIONS TAB ══════════════════ -->
    <div id="regions" class="tab-pane">
        <div class="section-label">Regional Intelligence</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-map"></i></div>
                    <h3>Regional Pollution Overview</h3>
                </div>
                {fig_region.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-chart-line"></i></div>
                    <h3>Year-over-Year by Region</h3>
                    <span class="chart-tag" style="background:rgba(244,114,182,0.1); color:#f472b6;">Multi-line</span>
                </div>
                {fig_region_yearly.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid-3">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-chart-pie"></i></div>
                    <h3>Region Pollution Share</h3>
                </div>
                {fig_pie_region.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-spider"></i></div>
                    <h3>Pollutant Profile Radar</h3>
                    <span class="chart-tag" style="background:rgba(251,191,36,0.1); color:#fbbf24;">Radar</span>
                </div>
                {fig_radar.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-box"></i></div>
                    <h3>Pollution Variability (Box)</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Distribution</span>
                </div>
                {fig_box_region.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-fire"></i></div>
                    <h3>Top 3 Polluted States per Region</h3>
                </div>
                {fig_region_top.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-tree"></i></div>
                    <h3>Top 3 Cleanest States per Region</h3>
                </div>
                {fig_region_clean.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-chart-bar"></i></div>
                    <h3>Pollutant Breakdown by Region</h3>
                </div>
                {fig_region_pollutant.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-chart-line"></i></div>
                    <h3>Region Predictions 2026</h3>
                </div>
                {fig_region_pred.to_html(full_html=False)}
            </div>
        </div>
    </div>

    <!-- ══════════════════ POLLUTANTS TAB ══════════════════ -->
    <div id="pollutants" class="tab-pane">
        <div class="section-label">Pollutant Deep Dive</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-flask"></i></div>
                    <h3>Pollutant Concentration Levels</h3>
                </div>
                {fig_pollutant.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-music"></i></div>
                    <h3>Pollutant Distribution (Violin)</h3>
                    <span class="chart-tag" style="background:rgba(244,114,182,0.1); color:#f472b6;">Violin</span>
                </div>
                {fig_violin.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-card" style="margin-bottom:20px;">
            <div class="chart-header">
                <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-th"></i></div>
                <h3>State × Pollutant Concentration Heatmap</h3>
                <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Full Matrix</span>
            </div>
            {fig_pollutant_heatmap.to_html(full_html=False)}
        </div>

        <!-- Pollutant Summary Table -->
        <div class="data-table-wrap">
            <div class="data-table-header">
                <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-table"></i></div>
                <h3>State-wise Dominant Pollutant Analysis</h3>
            </div>
            <div class="tbl-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>#</th><th>State</th><th>Region</th>
                            <th>Dominant Pollutant</th><th>Value</th><th>Level</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for i, (_, row) in enumerate(state_highest_pollutant.head(50).iterrows(), 1):
    region = region_map.get(row['state'], 'Unknown')
    level_pill = "pill-red" if "High" in row['pollutant_level'] else "pill-yellow" if "Medium" in row['pollutant_level'] else "pill-green"
    html_template += f"""
                        <tr>
                            <td style="color:var(--muted);">{i}</td>
                            <td><strong>{row['state']}</strong></td>
                            <td><span class="pill pill-blue">{region}</span></td>
                            <td><code style="background:rgba(56,189,248,0.1); color:#38bdf8; padding:2px 8px; border-radius:6px; font-size:0.8rem;">{row['pollutant_id']}</code></td>
                            <td><strong>{row['pollutant_avg']:.1f}</strong></td>
                            <td><span class="pill {level_pill}">{row['pollutant_level']}</span></td>
                        </tr>"""

html_template += f"""
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ══════════════════ ADVANCED TAB ══════════════════ -->
    <div id="advanced" class="tab-pane">
        <div class="section-label">Advanced Visualizations</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-sun"></i></div>
                    <h3>Sunburst: Region → State Hierarchy</h3>
                    <span class="chart-tag" style="background:rgba(52,211,153,0.1); color:#34d399;">Sunburst</span>
                </div>
                {fig_sunburst.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-th-large"></i></div>
                    <h3>Treemap: State Pollution by Region</h3>
                    <span class="chart-tag" style="background:rgba(251,191,36,0.1); color:#fbbf24;">Treemap</span>
                </div>
                {fig_treemap.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-music"></i></div>
                    <h3>Pollutant Violin Distribution</h3>
                </div>
                {fig_violin.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-box"></i></div>
                    <h3>Regional Variability Box Plot</h3>
                </div>
                {fig_box_region.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-bullseye"></i></div>
                    <h3>City Count vs Pollution Scatter</h3>
                </div>
                {fig_scatter.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-spider"></i></div>
                    <h3>Radar: Region Pollutant Profile</h3>
                </div>
                {fig_radar.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-water"></i></div>
                    <h3>Waterfall: Annual AQI Change</h3>
                </div>
                {fig_waterfall.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-chart-area"></i></div>
                    <h3>Seasonal Area Chart by Year</h3>
                </div>
                {fig_area.to_html(full_html=False)}
            </div>
        </div>
    </div>

    <!-- TIME SERIES TAB -->
    <div id="timeseries" class="tab-pane">
        <div class="section-label">Time Series Analysis</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(244,114,182,0.15); color:#f472b6;"><i class="fas fa-calendar-alt"></i></div>
                    <h3>Monthly Pollution Heatmap (2022–2025)</h3>
                    <span class="chart-tag" style="background:rgba(244,114,182,0.1); color:#f472b6;">Heatmap</span>
                </div>
                {fig_heatmap.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-chart-line"></i></div>
                    <h3>Quarterly Pollution Trends (2022–2025)</h3>
                    <span class="chart-tag" style="background:rgba(56,189,248,0.1); color:#38bdf8;">Quarterly</span>
                </div>
                {fig_quarterly.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-chart-area"></i></div>
                    <h3>Seasonal Pollution Patterns by Year</h3>
                    <span class="chart-tag" style="background:rgba(52,211,153,0.1); color:#34d399;">Area</span>
                </div>
                {fig_area.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(251,191,36,0.15); color:#fbbf24;"><i class="fas fa-water"></i></div>
                    <h3>Year-on-Year AQI Change (Waterfall)</h3>
                    <span class="chart-tag" style="background:rgba(251,191,36,0.1); color:#fbbf24;">Waterfall</span>
                </div>
                {fig_waterfall.to_html(full_html=False)}
            </div>
        </div>
    </div>

    <!-- RANKINGS TAB -->
    <div id="rankings" class="tab-pane">
        <div class="section-label">State Rankings</div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-filter"></i></div>
                    <h3>Top 10 Most Polluted States — Funnel View</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Funnel</span>
                </div>
                {fig_funnel.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(56,189,248,0.15); color:#38bdf8;"><i class="fas fa-bullseye"></i></div>
                    <h3>Cities Count vs Average Pollution by State</h3>
                    <span class="chart-tag" style="background:rgba(56,189,248,0.1); color:#38bdf8;">Scatter</span>
                </div>
                {fig_scatter.to_html(full_html=False)}
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(52,211,153,0.15); color:#34d399;"><i class="fas fa-tree"></i></div>
                    <h3>Top 10 Cleanest States</h3>
                    <span class="chart-tag" style="background:rgba(52,211,153,0.1); color:#34d399;">Best</span>
                </div>
                {fig_clean.to_html(full_html=False)}
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-header-icon" style="background:rgba(248,113,113,0.15); color:#f87171;"><i class="fas fa-fire"></i></div>
                    <h3>Top 15 Most Polluted States</h3>
                    <span class="chart-tag" style="background:rgba(248,113,113,0.1); color:#f87171;">Worst</span>
                </div>
                {fig_state.to_html(full_html=False)}
            </div>
        </div>
    </div>

</div><!-- /tabs-wrapper -->

<!-- FOOTER -->
<footer>
    <div class="footer-brand">AQI Intelligence Platform</div>
    <p>
        Data Source: India AQI Monitoring Network (2022–2025) &nbsp;·&nbsp;
        {total_states} States &amp; UTs &nbsp;·&nbsp; {total_cities} Cities &nbsp;·&nbsp;
        {total_pollutants} Pollutants Tracked
    </p>
    <p>Predictions powered by Polynomial Regression (degree 2) with Linear fallback &nbsp;·&nbsp;
       Last updated {pd.Timestamp.now().strftime('%B %d, %Y')}</p>
    <div class="legend-dots">
        <span class="legend-dot"><span class="dot" style="background:#f87171;"></span> High AQI &gt; 100</span>
        <span class="legend-dot"><span class="dot" style="background:#fbbf24;"></span> Medium 50–100</span>
        <span class="legend-dot"><span class="dot" style="background:#34d399;"></span> Low &lt; 50</span>
        <span class="legend-dot"><span class="dot" style="background:#38bdf8;"></span> Prediction 2026</span>
    </div>
</footer>

<script>
    // ── TAB SWITCHING ──
    function switchTab(id, el) {{
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tab-box').forEach(b => b.classList.remove('active-box'));
        document.getElementById(id).classList.add('active');
        if(el) el.classList.add('active-box');
        window.scrollTo({{ top: document.querySelector('.tabs-wrapper').offsetTop - 20, behavior:'smooth' }});
    }}

    // ── SCROLL PROGRESS ──
    window.addEventListener('scroll', () => {{
        const s = document.documentElement;
        const pct = (s.scrollTop / (s.scrollHeight - s.clientHeight)) * 100;
        document.getElementById('scroll-progress').style.width = pct + '%';
    }});

    // ── PARTICLES ──
    (function() {{
        const container = document.getElementById('particles');
        const colors = ['#38bdf8','#f472b6','#34d399','#fbbf24','#818cf8'];
        for(let i = 0; i < 40; i++) {{
            const p = document.createElement('div');
            p.className = 'particle';
            const size = Math.random() * 4 + 2;
            p.style.cssText = `
                width: ${{size}}px; height: ${{size}}px;
                left: ${{Math.random() * 100}}%;
                background: ${{colors[Math.floor(Math.random()*colors.length)]}};
                animation-duration: ${{Math.random() * 20 + 15}}s;
                animation-delay: -${{Math.random() * 20}}s;
                filter: blur(${{Math.random() * 1}}px);
            `;
            container.appendChild(p);
        }}
    }})();

    // ── COUNTER ANIMATION ──
    function animateCounter(el, target, decimals=0) {{
        const duration = 1800;
        const start = performance.now();
        const update = (time) => {{
            const p = Math.min((time - start) / duration, 1);
            const ease = 1 - Math.pow(1 - p, 4);
            el.textContent = (target * ease).toFixed(decimals);
            if(p < 1) requestAnimationFrame(update);
        }};
        requestAnimationFrame(update);
    }}

    // Animate KPI counters on load
    window.addEventListener('load', () => {{
        setTimeout(() => {{
            document.querySelectorAll('.kpi-val').forEach(el => {{
                const val = parseFloat(el.textContent);
                if(!isNaN(val)) {{
                    const dec = el.textContent.includes('.') ? 1 : 0;
                    animateCounter(el, val, dec);
                }}
            }});
        }}, 300);
    }});

    // ── INTERSECTION OBSERVER for chart cards ──
    const observer = new IntersectionObserver((entries) => {{
        entries.forEach(e => {{
            if(e.isIntersecting) {{
                e.target.style.opacity = '1';
                e.target.style.transform = 'translateY(0)';
            }}
        }});
    }}, {{ threshold: 0.1 }});

    document.querySelectorAll('.chart-card, .kpi, .data-table-wrap').forEach(card => {{
        card.style.opacity = '0';
        card.style.transform = 'translateY(24px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    }});
</script>
</body>
</html>"""

with open("AQIHTML.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("\n" + "="*60)
print("✅ MIND-BLOWING DASHBOARD CREATED SUCCESSFULLY!")
print("="*60)
print("\n📊 FILE: AQIHTML.html")
print("\n🎨 NEW DESIGN FEATURES:")
print("   • Dark deep-space aesthetic (#050a14 background)")
print("   • Animated floating particles")
print("   • Scroll progress bar")
print("   • Counter animation on KPI cards")
print("   • Intersection Observer scroll animations")
print("   • Glassmorphism chart cards with hover glows")
print("   • Gradient hero headline")
print("   • 6-tab navigation system")
print("   • Prediction hero card")
print("   • Sticky navbar with blur")
print("\n📊 NEW GRAPHS ADDED (13 additional):")
print("   • Quarterly trend lines")
print("   • Pollutant breakdown grouped bar by region")
print("   • Box plot variability by region")
print("   • Violin plot by pollutant type")
print("   • Scatter: city count vs avg pollution")
print("   • Year-over-year multi-line by region")
print("   • Radar/spider chart: region pollutant profiles")
print("   • Pie/donut: region pollution share")
print("   • Sunburst: region → state hierarchy")
print("   • Treemap: state pollution by region")
print("   • Waterfall: year-on-year AQI change")
print("   • Funnel: top 10 polluted states ranked")
print("   • Area chart: seasonal patterns by year")
