# pages/2_EDA_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/dengue_training_60k.csv')
    df['log1p_cases'] = np.log1p(df['total_cases'])
    return df

st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown("Interactive exploration of the dengue training dataset (60,000 rows)")
st.markdown("---")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Place `dengue_training_60k.csv` in the `data/` folder.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    selected_cities = st.multiselect("Select Cities", df['city'].unique().tolist(),
                                      default=df['city'].unique().tolist())
    year_range = st.slider("Year Range", int(df['year'].min()), int(df['year'].max()),
                           (int(df['year'].min()), int(df['year'].max())))

df_f = df[df['city'].isin(selected_cities) &
          df['year'].between(year_range[0], year_range[1])]

# ── Top metrics ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Outbreak Rate",   f"{df_f['outbreak_zone'].mean()*100:.1f}%")
c3.metric("Mean Cases/Week", f"{df_f['total_cases'].mean():.1f}")
c4.metric("Max Cases/Week",  f"{df_f['total_cases'].max()}")
c5.metric("Cities Selected", len(selected_cities))

st.markdown("---")

# ── Row 1: Distribution plots ──────────────────────────────────────────
st.markdown("### Target Variable Distributions")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df_f, x='total_cases', nbins=60,
                       title='total_cases — Raw Distribution (Skew=4.93)',
                       color_discrete_sequence=['#2563EB'])
    fig.update_layout(showlegend=False, xaxis_title='Cases per Week',
                      yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df_f, x='log1p_cases', nbins=50,
                       title='log1p(total_cases) — After Transform (Skew=0.64)',
                       color_discrete_sequence=['#059669'])
    fig.update_layout(showlegend=False, xaxis_title='log1p(Cases)',
                      yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Class balance + seasonal ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    vc = df_f['outbreak_zone'].value_counts().reset_index()
    vc.columns = ['Class', 'Count']
    vc['Label'] = vc['Class'].map({0: 'No Outbreak (0)', 1: 'Outbreak (1)'})
    fig = px.bar(vc, x='Label', y='Count',
                 title='Class Distribution (outbreak_zone)',
                 color='Label',
                 color_discrete_map={'No Outbreak (0)': '#059669', 'Outbreak (1)': '#DC2626'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    weekly = df_f.groupby('weekofyear').agg(
        mean_cases=('total_cases', 'mean'),
        outbreak_rate=('outbreak_zone', 'mean')
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=weekly['weekofyear'], y=weekly['mean_cases'],
                             name='Mean Cases', line=dict(color='#2563EB', width=2),
                             fill='tozeroy', fillcolor='rgba(37,99,235,0.1)'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=weekly['weekofyear'], y=weekly['outbreak_rate']*100,
                             name='Outbreak Rate %', line=dict(color='#DC2626', width=2,
                             dash='dash')), secondary_y=True)
    fig.update_layout(title='Seasonal Pattern: Cases & Outbreak Rate by Week')
    fig.update_xaxes(title_text='Week of Year')
    fig.update_yaxes(title_text='Mean Cases', secondary_y=False)
    fig.update_yaxes(title_text='Outbreak Rate (%)', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: City comparison + correlation ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    city_stats = df_f.groupby('city').agg(
        mean_cases=('total_cases', 'mean'),
        outbreak_pct=('outbreak_zone', lambda x: x.mean()*100)
    ).reset_index().sort_values('mean_cases', ascending=True)

    fig = px.bar(city_stats, x='mean_cases', y='city', orientation='h',
                 title='Mean Weekly Cases by City',
                 color='mean_cases', color_continuous_scale='YlOrRd',
                 labels={'mean_cases': 'Mean Cases', 'city': 'City'})
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    num_cols = ['lag1_cases', 'lag4_cases', 'reanalysis_relative_humidity_percent',
                'precipitation_amt_mm', 'hospital_capacity_index',
                'population_density', 'reanalysis_tdtr_k', 'total_cases', 'outbreak_zone']
    corr_matrix = df_f[num_cols].corr()
    fig = px.imshow(corr_matrix, text_auto='.2f',
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='RdYlGn', zmin=-1, zmax=1,
                    aspect='auto')
    fig.update_layout(font_size=10)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 4: Scatter + box ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    sample = df_f.sample(min(4000, len(df_f)), random_state=42)
    fig = px.scatter(sample, x='reanalysis_relative_humidity_percent',
                     y='log1p_cases', color='outbreak_zone',
                     title='Humidity vs log1p(Cases)',
                     color_discrete_map={0: '#059669', 1: '#DC2626'},
                     labels={'outbreak_zone': 'Outbreak'},
                     opacity=0.4, size_max=5)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df_f, x='city', y='total_cases',
                 title='Cases Distribution by City',
                 color='city',
                 labels={'total_cases': 'Cases', 'city': 'City'})
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    fig.update_yaxes(range=[0, 80])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### Raw Data Sample")
st.dataframe(df_f.sample(min(200, len(df_f)), random_state=42).reset_index(drop=True),
             use_container_width=True)
