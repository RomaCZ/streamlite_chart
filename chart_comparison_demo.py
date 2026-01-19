import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample project metrics data
@st.cache_data
def generate_sample_data():
    """Generate sample project metrics data"""
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='W')
    projects = ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta']

    data = []
    for project in projects:
        np.random.seed(hash(project) % 2**32)
        for date in dates:
            data.append({
                'Date': date,
                'Project': project,
                'Tasks Completed': np.random.randint(10, 50),
                'Automated Tests': np.random.randint(50, 200),
                'Manual Tests': np.random.randint(5, 30),
                'Bugs Found': np.random.randint(0, 15),
                'Code Coverage': np.random.uniform(70, 95),
                'Sprint Progress': np.random.uniform(60, 100)
            })

    return pd.DataFrame(data)

df = generate_sample_data()

# App title and intro
st.set_page_config(page_title="Chart Library Comparison", layout="wide")
st.title("📊 Streamlit Charting Libraries Comparison")
st.markdown("Compare different visualization libraries for your project metrics dashboard")

# Sidebar for filtering
st.sidebar.header("Data Filters")
selected_projects = st.sidebar.multiselect(
    "Select Projects",
    options=df['Project'].unique(),
    default=df['Project'].unique()
)

filtered_df = df[df['Project'].isin(selected_projects)]

# Show sample data
with st.expander("📋 View Sample Data"):
    st.dataframe(filtered_df.head(20))

st.divider()

# ============================================================================
# 1. PLOTLY
# ============================================================================
st.header("1. 📈 Plotly")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Example Charts")

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Line chart
        fig1 = px.line(filtered_df, x='Date', y='Tasks Completed', color='Project',
                      title='Tasks Completed Over Time',
                      markers=True)
        fig1.update_layout(hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)

        # Bar chart
        monthly_data = filtered_df.groupby(['Project']).agg({
            'Tasks Completed': 'sum',
            'Automated Tests': 'sum',
            'Bugs Found': 'sum'
        }).reset_index()

        fig2 = px.bar(monthly_data, x='Project', y=['Tasks Completed', 'Automated Tests', 'Bugs Found'],
                     title='Metrics Summary by Project',
                     barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    except ImportError:
        st.error("Plotly not installed. Run: `pip install plotly`")

with col2:
    st.subheader("Pros & Cons")
    st.markdown("""
    **✅ Pros:**
    - Highly interactive (zoom, pan, hover)
    - Professional look out-of-the-box
    - Excellent for dashboards
    - Wide variety of chart types
    - Easy to customize colors/themes
    - Good documentation

    **❌ Cons:**
    - Larger library size
    - Can be slower with huge datasets
    - Some advanced customization requires learning

    **🎨 Customization:**
    - Themes & templates
    - Full control over colors, fonts, layouts
    - Custom hover tooltips
    - Animations

    **📦 Install:** `pip install plotly`
    """)

st.divider()

# ============================================================================
# 2. ALTAIR
# ============================================================================
st.header("2. 🎨 Altair")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Example Charts")

    try:
        import altair as alt

        # Line chart with selection
        chart1 = alt.Chart(filtered_df).mark_line(point=True).encode(
            x='Date:T',
            y='Tasks Completed:Q',
            color='Project:N',
            tooltip=['Date:T', 'Project:N', 'Tasks Completed:Q']
        ).properties(
            title='Tasks Completed Over Time',
            width=600,
            height=300
        ).interactive()

        st.altair_chart(chart1, use_container_width=True)

        # Multi-metric scatter plot
        chart2 = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x='Automated Tests:Q',
            y='Code Coverage:Q',
            color='Project:N',
            tooltip=['Project:N', 'Automated Tests:Q', 'Code Coverage:Q', 'Date:T']
        ).properties(
            title='Test Automation vs Code Coverage',
            width=600,
            height=300
        ).interactive()

        st.altair_chart(chart2, use_container_width=True)

    except ImportError:
        st.error("Altair not installed. Run: `pip install altair`")

with col2:
    st.subheader("Pros & Cons")
    st.markdown("""
    **✅ Pros:**
    - Clean, declarative syntax
    - Great for statistical visualizations
    - Built-in interactivity
    - Works well with Streamlit
    - Vega-Lite based (web standard)
    - Good for exploratory analysis

    **❌ Cons:**
    - 5000 row limit (configurable)
    - Less chart variety than Plotly
    - Steeper learning curve for complex charts

    **🎨 Customization:**
    - Theme configuration
    - Custom color schemes
    - Composable charts (layering, concat)
    - Conditional encoding

    **📦 Install:** `pip install altair`
    """)

st.divider()

# ============================================================================
# 3. MATPLOTLIB / SEABORN
# ============================================================================
st.header("3. 📊 Matplotlib + Seaborn")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Example Charts")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        sns.set_style("whitegrid")

        # Line plot
        fig, ax = plt.subplots(figsize=(10, 4))
        for project in filtered_df['Project'].unique():
            project_data = filtered_df[filtered_df['Project'] == project]
            ax.plot(project_data['Date'], project_data['Tasks Completed'],
                   marker='o', label=project, linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Tasks Completed')
        ax.set_title('Tasks Completed Over Time')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Heatmap of correlations
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_cols = ['Tasks Completed', 'Automated Tests', 'Manual Tests',
                       'Bugs Found', 'Code Coverage']
        correlation_matrix = filtered_df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax)
        ax.set_title('Metrics Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig)

    except ImportError:
        st.error("Matplotlib/Seaborn not installed. Run: `pip install matplotlib seaborn`")

with col2:
    st.subheader("Pros & Cons")
    st.markdown("""
    **✅ Pros:**
    - Industry standard
    - Maximum flexibility
    - Huge community & examples
    - Works with any Python environment
    - Seaborn adds statistical plots
    - Publication-quality output

    **❌ Cons:**
    - Static images (not interactive)
    - More verbose syntax
    - Requires more code for styling
    - Not as "modern" looking

    **🎨 Customization:**
    - Complete control over every element
    - Custom styles & themes
    - Matplotlib rcParams
    - Seaborn themes & palettes

    **📦 Install:** `pip install matplotlib seaborn`
    """)

st.divider()

# ============================================================================
# 4. STREAMLIT NATIVE CHARTS
# ============================================================================
st.header("4. ⚡ Streamlit Native Charts")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Example Charts")

    # Prepare data for native charts
    pivot_tasks = filtered_df.pivot_table(
        values='Tasks Completed',
        index='Date',
        columns='Project'
    ).fillna(0)

    # Line chart
    st.line_chart(pivot_tasks)

    # Bar chart - aggregate by project
    project_totals = filtered_df.groupby('Project')['Tasks Completed'].sum()
    st.bar_chart(project_totals)

    # Area chart
    st.area_chart(pivot_tasks)

with col2:
    st.subheader("Pros & Cons")
    st.markdown("""
    **✅ Pros:**
    - Zero configuration needed
    - Fastest to implement
    - Native Streamlit integration
    - Lightweight
    - Consistent with app design
    - Good for simple dashboards

    **❌ Cons:**
    - Limited chart types
    - Minimal customization
    - Basic interactivity only
    - No complex visualizations

    **🎨 Customization:**
    - Very limited
    - Colors controlled by theme
    - Cannot customize labels/titles much

    **📦 Install:** Built-in (no install needed)
    """)

st.divider()

# ============================================================================
# 5. ECHARTS (via streamlit-echarts)
# ============================================================================
st.header("5. 🚀 ECharts")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Example Charts")

    try:
        from streamlit_echarts import st_echarts

        # Prepare data
        project_summary = filtered_df.groupby('Project').agg({
            'Tasks Completed': 'sum',
            'Automated Tests': 'sum',
            'Bugs Found': 'sum'
        }).reset_index()

        # Bar chart with ECharts
        options = {
            "title": {"text": "Project Metrics Summary"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"data": ["Tasks Completed", "Automated Tests", "Bugs Found"]},
            "xAxis": {"type": "category", "data": project_summary['Project'].tolist()},
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": "Tasks Completed",
                    "type": "bar",
                    "data": project_summary['Tasks Completed'].tolist(),
                },
                {
                    "name": "Automated Tests",
                    "type": "bar",
                    "data": project_summary['Automated Tests'].tolist(),
                },
                {
                    "name": "Bugs Found",
                    "type": "bar",
                    "data": project_summary['Bugs Found'].tolist(),
                }
            ]
        }
        st_echarts(options=options, height="400px")

        # Gauge chart for average progress
        avg_progress = filtered_df['Sprint Progress'].mean()
        gauge_option = {
            "title": {"text": "Average Sprint Progress"},
            "series": [
                {
                    "type": "gauge",
                    "progress": {"show": True},
                    "detail": {"valueAnimation": True, "formatter": "{value}%"},
                    "data": [{"value": round(avg_progress, 1), "name": "Progress"}]
                }
            ]
        }
        st_echarts(options=gauge_option, height="300px")

    except ImportError:
        st.error("streamlit-echarts not installed. Run: `pip install streamlit-echarts`")

with col2:
    st.subheader("Pros & Cons")
    st.markdown("""
    **✅ Pros:**
    - Extremely powerful & flexible
    - Beautiful animations
    - Huge chart variety
    - Great performance
    - Mobile-friendly
    - Rich interactivity

    **❌ Cons:**
    - Requires JSON configuration
    - Steeper learning curve
    - Less Python-like syntax
    - Documentation is mixed (Chinese/English)

    **🎨 Customization:**
    - Almost unlimited customization
    - Themes & custom styling
    - Animations & transitions
    - Complex interactions

    **📦 Install:** `pip install streamlit-echarts`
    """)

st.divider()

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
st.header("📋 Summary & Recommendations")

comparison_data = pd.DataFrame({
    'Library': ['Plotly', 'Altair', 'Matplotlib/Seaborn', 'Streamlit Native', 'ECharts'],
    'Ease of Use': ['⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐'],
    'Interactivity': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐', '⭐⭐', '⭐⭐⭐⭐⭐'],
    'Customization': ['⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐', '⭐⭐⭐⭐⭐'],
    'Performance': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
    'Chart Variety': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐⭐'],
    'Best For': [
        'Dashboards, interactive reports',
        'Statistical analysis, clean design',
        'Publications, maximum control',
        'Quick prototypes, simple charts',
        'Complex visualizations, animations'
    ]
})

st.dataframe(comparison_data, use_container_width=True, hide_index=True)

st.subheader("💡 Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🏆 For Your Use Case:**

    **Plotly** is recommended if:
    - You want professional dashboards
    - Interactivity is important
    - You need variety of charts
    - Customization matters
    """)

with col2:
    st.markdown("""
    **🎯 Quick Start:**

    **Streamlit Native** is good if:
    - Simple metrics display
    - Minimal setup time
    - Basic charts suffice
    - Prototyping quickly
    """)

with col3:
    st.markdown("""
    **⚡ Maximum Power:**

    **ECharts** is best if:
    - You need stunning visuals
    - Complex interactions required
    - Performance is critical
    - You have time to learn
    """)

st.divider()

st.subheader("🛠️ Installation Commands")
st.code("""
# Install all libraries to try them out
pip install plotly altair matplotlib seaborn streamlit-echarts

# Or install individually:
pip install plotly              # Most recommended for your use case
pip install altair              # Clean, declarative approach
pip install matplotlib seaborn  # Traditional, maximum flexibility
pip install streamlit-echarts   # Most powerful, steeper learning curve
""", language="bash")

st.success("💡 **Tip:** Start with Plotly for the best balance of ease-of-use, interactivity, and customization!")
