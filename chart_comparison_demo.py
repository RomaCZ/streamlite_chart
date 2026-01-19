import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample project metrics data
@st.cache_data
def generate_sample_data():
    """Generate sample project metrics data with realistic trends"""
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='W')
    projects = ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta']

    data = []
    for idx, project in enumerate(projects):
        np.random.seed(hash(project) % 2**32)
        base_tasks = 20 + idx * 5
        base_tests = 100 + idx * 20
        base_coverage = 75 + idx * 3

        for i, date in enumerate(dates):
            # Add realistic trends over time
            trend_factor = 1 + (i / len(dates)) * 0.3  # 30% growth over the year
            seasonal = 1 + 0.1 * np.sin(i / 4)  # Small seasonal variation

            data.append({
                'Date': date,
                'Project': project,
                'Tasks Completed': int(base_tasks * trend_factor * seasonal + np.random.randint(-5, 10)),
                'Automated Tests': int(base_tests * trend_factor + np.random.randint(-20, 30)),
                'Manual Tests': int(15 * (1.1 - trend_factor * 0.1) + np.random.randint(-3, 5)),  # Manual tests decrease as automation increases
                'Bugs Found': max(0, int(10 * (1.2 - trend_factor * 0.2) + np.random.randint(-3, 5))),  # Bugs decrease over time
                'Code Coverage': min(98, base_coverage + (i / len(dates)) * 15 + np.random.uniform(-2, 3)),
                'Sprint Progress': min(100, 70 + trend_factor * 10 + np.random.uniform(-5, 10)),
                'Team Size': 5 + idx
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

# Date range selector
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered_df = df[df['Project'].isin(selected_projects)].copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_date) &
                              (filtered_df['Date'].dt.date <= end_date)]

# Show code snippets toggle
st.sidebar.divider()
show_code = st.sidebar.checkbox("Show Code Snippets", value=False)

# Key metrics
if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", f"{filtered_df['Tasks Completed'].sum():,.0f}",
                 delta=f"{filtered_df['Tasks Completed'].mean():.1f} avg/week")
    with col2:
        st.metric("Automated Tests", f"{filtered_df['Automated Tests'].sum():,.0f}",
                 delta=f"{filtered_df['Automated Tests'].mean():.0f} avg/week")
    with col3:
        avg_coverage = filtered_df['Code Coverage'].mean()
        st.metric("Avg Coverage", f"{avg_coverage:.1f}%",
                 delta=f"{filtered_df['Code Coverage'].std():.1f}% std dev")
    with col4:
        total_bugs = filtered_df['Bugs Found'].sum()
        st.metric("Bugs Found", f"{total_bugs:,.0f}",
                 delta=f"-{filtered_df['Bugs Found'].mean():.1f} avg/week" if total_bugs > 0 else "0",
                 delta_color="inverse")

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
        fig1.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig1, width='stretch')

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import plotly.express as px

fig = px.line(df, x='Date', y='Tasks Completed', color='Project',
              title='Tasks Completed Over Time', markers=True)
fig.update_layout(hovermode='x unified')
st.plotly_chart(fig, width='stretch')
                """, language="python")

        # Bar chart
        monthly_data = filtered_df.groupby(['Project']).agg({
            'Tasks Completed': 'sum',
            'Automated Tests': 'sum',
            'Bugs Found': 'sum'
        }).reset_index()

        fig2 = px.bar(monthly_data, x='Project', y=['Tasks Completed', 'Automated Tests', 'Bugs Found'],
                     title='Metrics Summary by Project',
                     barmode='group')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, width='stretch')

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import plotly.express as px

monthly_data = df.groupby(['Project']).agg({
    'Tasks Completed': 'sum',
    'Automated Tests': 'sum',
    'Bugs Found': 'sum'
}).reset_index()

fig = px.bar(monthly_data, x='Project',
             y=['Tasks Completed', 'Automated Tests', 'Bugs Found'],
             title='Metrics Summary by Project', barmode='group')
st.plotly_chart(fig, width='stretch')
                """, language="python")

        # Pie chart for project distribution
        project_tasks = filtered_df.groupby('Project')['Tasks Completed'].sum().reset_index()
        fig3 = px.pie(project_tasks, values='Tasks Completed', names='Project',
                     title='Task Distribution by Project',
                     hole=0.4)  # Donut chart
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, width='stretch')

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import plotly.express as px

project_tasks = df.groupby('Project')['Tasks Completed'].sum()
fig = px.pie(project_tasks, values='Tasks Completed', names='Project',
             title='Task Distribution by Project', hole=0.4)
st.plotly_chart(fig, width='stretch')
                """, language="python")

        # Box plot for distribution analysis
        fig4 = px.box(filtered_df, x='Project', y='Code Coverage',
                     title='Code Coverage Distribution by Project',
                     color='Project',
                     points='all')  # Show all points
        fig4.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig4, width='stretch')

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import plotly.express as px

fig = px.box(df, x='Project', y='Code Coverage',
             title='Code Coverage Distribution by Project',
             color='Project', points='all')
st.plotly_chart(fig, width='stretch')
                """, language="python")

        # Bubble chart - 3 dimensions
        fig5 = px.scatter(filtered_df, x='Automated Tests', y='Code Coverage',
                         size='Tasks Completed', color='Project',
                         title='Tests vs Coverage (bubble size = tasks completed)',
                         hover_data=['Date', 'Bugs Found'],
                         size_max=40)
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, width='stretch')

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import plotly.express as px

fig = px.scatter(df, x='Automated Tests', y='Code Coverage',
                size='Tasks Completed', color='Project',
                title='Tests vs Coverage (bubble size = tasks)',
                hover_data=['Date', 'Bugs Found'], size_max=40)
st.plotly_chart(fig, width='stretch')
                """, language="python")

        # Stacked area chart
        pivot_coverage = filtered_df.pivot_table(
            values='Code Coverage',
            index='Date',
            columns='Project',
            aggfunc='mean'
        ).fillna(0)

        fig6 = go.Figure()
        for project in pivot_coverage.columns:
            fig6.add_trace(go.Scatter(
                x=pivot_coverage.index,
                y=pivot_coverage[project],
                name=project,
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(0,0,0,0.1)'
            ))
        fig6.update_layout(
            title='Code Coverage Trends (Stacked Area)',
            xaxis_title='Date',
            yaxis_title='Code Coverage %',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig6, width='stretch')

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import plotly.graph_objects as go

pivot_data = df.pivot_table(values='Code Coverage',
                            index='Date', columns='Project')
fig = go.Figure()
for project in pivot_data.columns:
    fig.add_trace(go.Scatter(x=pivot_data.index, y=pivot_data[project],
                            name=project, mode='lines', stackgroup='one'))
fig.update_layout(title='Code Coverage Trends (Stacked Area)')
st.plotly_chart(fig, width='stretch')
                """, language="python")

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

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import altair as alt

chart = alt.Chart(df).mark_line(point=True).encode(
    x='Date:T',
    y='Tasks Completed:Q',
    color='Project:N',
    tooltip=['Date:T', 'Project:N', 'Tasks Completed:Q']
).properties(title='Tasks Completed Over Time').interactive()

st.altair_chart(chart, use_container_width=True)
                """, language="python")

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

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import altair as alt

chart = alt.Chart(df).mark_circle(size=60).encode(
    x='Automated Tests:Q',
    y='Code Coverage:Q',
    color='Project:N',
    tooltip=['Project:N', 'Automated Tests:Q', 'Code Coverage:Q']
).properties(title='Test Automation vs Code Coverage').interactive()

st.altair_chart(chart, use_container_width=True)
                """, language="python")

        # Stacked bar chart
        chart3 = alt.Chart(filtered_df.groupby(['Project', 'Date']).agg({
            'Automated Tests': 'sum',
            'Manual Tests': 'sum'
        }).reset_index().melt(id_vars=['Project', 'Date'],
                             var_name='Test Type',
                             value_name='Count')).mark_bar().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Count:Q', title='Number of Tests'),
            color='Test Type:N',
            tooltip=['Date:T', 'Project:N', 'Test Type:N', 'Count:Q']
        ).properties(
            title='Automated vs Manual Tests Over Time',
            width=600,
            height=300
        ).interactive()

        st.altair_chart(chart3, use_container_width=True)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import altair as alt

melted = df.melt(id_vars=['Project', 'Date'],
                value_vars=['Automated Tests', 'Manual Tests'],
                var_name='Test Type', value_name='Count')
chart = alt.Chart(melted).mark_bar().encode(
    x='Date:T', y='Count:Q', color='Test Type:N',
    tooltip=['Date:T', 'Test Type:N', 'Count:Q']
).properties(title='Automated vs Manual Tests').interactive()

st.altair_chart(chart, use_container_width=True)
                """, language="python")

        # Histogram
        chart4 = alt.Chart(filtered_df).mark_bar(opacity=0.7).encode(
            x=alt.X('Code Coverage:Q', bin=alt.Bin(maxbins=20), title='Code Coverage %'),
            y=alt.Y('count()', title='Frequency'),
            color='Project:N',
            tooltip=['Project:N', 'count()']
        ).properties(
            title='Code Coverage Distribution (Histogram)',
            width=600,
            height=300
        )

        st.altair_chart(chart4, use_container_width=True)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import altair as alt

chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
    x=alt.X('Code Coverage:Q', bin=alt.Bin(maxbins=20)),
    y=alt.Y('count()', title='Frequency'),
    color='Project:N',
    tooltip=['Project:N', 'count()']
).properties(title='Code Coverage Distribution')

st.altair_chart(chart, use_container_width=True)
                """, language="python")

        # Area chart with selection
        brush = alt.selection_interval(encodings=['x'])

        chart5_base = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(
            x='Date:T',
            y='Bugs Found:Q',
            color='Project:N',
            tooltip=['Date:T', 'Project:N', 'Bugs Found:Q']
        ).properties(
            title='Bugs Found Over Time (Brush to zoom)',
            width=600,
            height=250
        )

        chart5 = chart5_base.add_params(brush)

        st.altair_chart(chart5, use_container_width=True)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import altair as alt

brush = alt.selection_interval(encodings=['x'])
chart = alt.Chart(df).mark_area(opacity=0.6).encode(
    x='Date:T', y='Bugs Found:Q', color='Project:N',
    tooltip=['Date:T', 'Project:N', 'Bugs Found:Q']
).properties(title='Bugs Found Over Time').add_params(brush)

st.altair_chart(chart, use_container_width=True)
                """, language="python")

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

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
for project in df['Project'].unique():
    project_data = df[df['Project'] == project]
    ax.plot(project_data['Date'], project_data['Tasks Completed'],
           marker='o', label=project, linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Tasks Completed')
ax.set_title('Tasks Completed Over Time')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
                """, language="python")

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

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
numeric_cols = ['Tasks Completed', 'Automated Tests', 'Manual Tests',
               'Bugs Found', 'Code Coverage']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
           cmap='coolwarm', center=0, ax=ax)
ax.set_title('Metrics Correlation Heatmap')
st.pyplot(fig)
                """, language="python")

        # Violin plot - distribution with kernel density
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=filtered_df, x='Project', y='Code Coverage',
                      palette='Set2', ax=ax)
        ax.set_title('Code Coverage Distribution by Project (Violin Plot)')
        ax.set_ylabel('Code Coverage %')
        plt.tight_layout()
        st.pyplot(fig)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df, x='Project', y='Code Coverage',
              palette='Set2', ax=ax)
ax.set_title('Code Coverage Distribution (Violin Plot)')
st.pyplot(fig)
                """, language="python")

        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        project_summary = filtered_df.groupby('Project').agg({
            'Automated Tests': 'sum',
            'Manual Tests': 'sum'
        })
        project_summary.plot(kind='bar', stacked=True, ax=ax,
                            color=['#2E86AB', '#A23B72'])
        ax.set_title('Test Types by Project (Stacked Bar)')
        ax.set_xlabel('Project')
        ax.set_ylabel('Number of Tests')
        ax.legend(title='Test Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
project_summary = df.groupby('Project').agg({
    'Automated Tests': 'sum',
    'Manual Tests': 'sum'
})
project_summary.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Test Types by Project (Stacked Bar)')
plt.xticks(rotation=45)
st.pyplot(fig)
                """, language="python")

        # Histogram with KDE
        fig, ax = plt.subplots(figsize=(10, 5))
        for project in filtered_df['Project'].unique():
            project_data = filtered_df[filtered_df['Project'] == project]
            sns.histplot(data=project_data, x='Tasks Completed',
                        kde=True, label=project, alpha=0.5, ax=ax)
        ax.set_title('Tasks Completed Distribution (Histogram with KDE)')
        ax.set_xlabel('Tasks Completed')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 5))
for project in df['Project'].unique():
    project_data = df[df['Project'] == project]
    sns.histplot(data=project_data, x='Tasks Completed',
                kde=True, label=project, alpha=0.5, ax=ax)
ax.set_title('Tasks Completed Distribution')
ax.legend()
st.pyplot(fig)
                """, language="python")

        # Pair plot preview (using subset)
        fig = sns.pairplot(filtered_df[['Project', 'Tasks Completed', 'Automated Tests',
                                        'Code Coverage']].sample(min(100, len(filtered_df))),
                          hue='Project', diag_kind='kde', height=2)
        fig.fig.suptitle('Metrics Relationships (Pair Plot)', y=1.02)
        st.pyplot(fig)

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
import seaborn as sns

fig = sns.pairplot(df[['Project', 'Tasks Completed',
                       'Automated Tests', 'Code Coverage']],
                  hue='Project', diag_kind='kde')
st.pyplot(fig)
                """, language="python")

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

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
from streamlit_echarts import st_echarts

options = {
    "title": {"text": "Project Metrics Summary"},
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["Tasks", "Tests", "Bugs"]},
    "xAxis": {"type": "category", "data": projects},
    "yAxis": {"type": "value"},
    "series": [
        {"name": "Tasks", "type": "bar", "data": task_data},
        {"name": "Tests", "type": "bar", "data": test_data}
    ]
}
st_echarts(options=options, height="400px")
                """, language="python")

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

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
from streamlit_echarts import st_echarts

gauge_option = {
    "series": [{
        "type": "gauge",
        "progress": {"show": True},
        "detail": {"formatter": "{value}%"},
        "data": [{"value": 85, "name": "Progress"}]
    }]
}
st_echarts(options=gauge_option, height="300px")
                """, language="python")

        # Radar chart for project comparison
        project_avg = filtered_df.groupby('Project').agg({
            'Tasks Completed': 'mean',
            'Automated Tests': 'mean',
            'Code Coverage': 'mean',
            'Sprint Progress': 'mean'
        }).reset_index()

        # Normalize to 0-100 scale for radar
        max_tasks = project_avg['Tasks Completed'].max()
        max_tests = project_avg['Automated Tests'].max()

        radar_data = []
        for _, row in project_avg.iterrows():
            radar_data.append({
                "value": [
                    round((row['Tasks Completed'] / max_tasks) * 100, 1),
                    round((row['Automated Tests'] / max_tests) * 100, 1),
                    round(row['Code Coverage'], 1),
                    round(row['Sprint Progress'], 1)
                ],
                "name": row['Project']
            })

        radar_option = {
            "title": {"text": "Project Performance Comparison (Radar)"},
            "legend": {"data": project_avg['Project'].tolist()},
            "radar": {
                "indicator": [
                    {"name": "Tasks", "max": 100},
                    {"name": "Tests", "max": 100},
                    {"name": "Coverage", "max": 100},
                    {"name": "Progress", "max": 100}
                ]
            },
            "series": [{
                "type": "radar",
                "data": radar_data
            }]
        }
        st_echarts(options=radar_option, height="400px")

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
from streamlit_echarts import st_echarts

radar_option = {
    "title": {"text": "Project Performance (Radar)"},
    "radar": {
        "indicator": [
            {"name": "Tasks", "max": 100},
            {"name": "Tests", "max": 100},
            {"name": "Coverage", "max": 100}
        ]
    },
    "series": [{
        "type": "radar",
        "data": [{"value": [80, 90, 85], "name": "Project A"}]
    }]
}
st_echarts(options=radar_option, height="400px")
                """, language="python")

        # Pie chart with animation
        pie_data = []
        for _, row in project_summary.iterrows():
            pie_data.append({
                "value": int(row['Tasks Completed']),
                "name": row['Project']
            })

        pie_option = {
            "title": {"text": "Task Distribution", "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "type": "pie",
                    "radius": "50%",
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)"
                        }
                    }
                }
            ]
        }
        st_echarts(options=pie_option, height="400px")

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
from streamlit_echarts import st_echarts

pie_option = {
    "title": {"text": "Task Distribution"},
    "tooltip": {"trigger": "item"},
    "series": [{
        "type": "pie",
        "radius": "50%",
        "data": [
            {"value": 1048, "name": "Project A"},
            {"value": 735, "name": "Project B"}
        ]
    }]
}
st_echarts(options=pie_option, height="400px")
                """, language="python")

        # Line chart with smooth curves and area
        dates_list = sorted(filtered_df['Date'].unique())
        line_series = []

        for project in filtered_df['Project'].unique():
            project_data = filtered_df[filtered_df['Project'] == project].sort_values('Date')
            line_series.append({
                "name": project,
                "type": "line",
                "smooth": True,
                "areaStyle": {"opacity": 0.3},
                "data": project_data['Tasks Completed'].tolist()
            })

        line_option = {
            "title": {"text": "Tasks Completed Trend"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": filtered_df['Project'].unique().tolist()},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": [str(d)[:10] for d in dates_list]
            },
            "yAxis": {"type": "value"},
            "series": line_series
        }
        st_echarts(options=line_option, height="400px")

        if show_code:
            with st.expander("📝 View Code"):
                st.code("""
from streamlit_echarts import st_echarts

line_option = {
    "title": {"text": "Tasks Trend"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": dates},
    "yAxis": {"type": "value"},
    "series": [{
        "type": "line",
        "smooth": True,
        "areaStyle": {"opacity": 0.3},
        "data": task_values
    }]
}
st_echarts(options=line_option, height="400px")
                """, language="python")

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

st.dataframe(comparison_data, width='stretch', hide_index=True)

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
