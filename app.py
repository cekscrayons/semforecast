import streamlit as st
import pandas as pd
import io
import plotly.graph_objs as go
import plotly.express as px

# Import the forecast model
from sem_forecast_model import SEMForecastModel, load_data_from_csv

# Page configuration
st.set_page_config(
    page_title="Annual SEM Budget Forecasting Tool", 
    layout="wide", 
    page_icon="📊"
)

# Custom CSS for improved dark theme and white controls
st.markdown("""
<style>
.stApp {
    background-color: #212121;
}
.stSidebar {
    background-color: #1a1a1a;
}
/* White slider and button styling */
div[data-baseweb="slider"] {
    -webkit-appearance: none;
    background: transparent;
}
div[data-baseweb="slider"] [role="slider"] {
    background-color: white !important;
}
.stButton>button {
    color: white;
    background-color: white;
    border: none;
}
.stButton>button:hover {
    color: white;
    background-color: #1b79ff !important;
}
/* Metric styling for negative changes */
.negative {
    color: red;
}
</style>
""", unsafe_allow_html=True)

# Mapping function
def map_slider_to_value(slider_value, parameter_type):
    mappings = {
        'impression_share_growth': {
            'Low': 1.1,
            'Medium': 1.5,
            'High': 2.0
        },
        'conversion_rate_sensitivity': {
            'Reducing': 0.95,
            'Stable': 0.85,
            'Increasing': 0.75
        },
        'diminishing_returns': {
            'Low': 0.95,
            'Medium': 0.85,
            'High': 0.75
        }
    }
    return mappings[parameter_type][slider_value]

# (Rest of the existing chart creation function remains the same)

# Main Streamlit app
st.title("Annual SEM Budget Forecasting Tool")

# Add a brief explanation
st.markdown("""
This tool provides a comprehensive annual forecast for your SEM budget, 
leveraging advanced machine learning techniques to project performance 
based on historical data and configurable parameters.
""")

# Sidebar for model parameters
with st.sidebar:
    st.header("Forecast Configuration")
    
    # Core parameters with sliders and tooltips
    min_roas = st.slider(
        "Minimum ROAS Threshold ($)", 
        10, 30, 20, 1, 
        help="The minimum acceptable Return on Ad Spend. Lower values allow more aggressive spending."
    )
    growth_factor = st.slider(
        "YOY Growth Factor (%)", 
        0, 30, 10, 1, 
        help="Percentage increase in year-over-year performance. Higher values indicate more ambitious growth targets."
    ) / 100 + 1.0
    aov_growth = st.slider(
        "YOY AOV Growth (%)", 
        0, 20, 5, 1, 
        help="Average Order Value growth percentage. Reflects expected increase in customer spending."
    ) / 100 + 1.0
    cpc_inflation = st.slider(
        "YOY CPC Inflation (%)", 
        0, 15, 2, 1, 
        help="Cost Per Click inflation rate. Accounts for increasing digital advertising costs."
    ) / 100 + 1.0
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        impression_share_growth = st.select_slider(
            "Impression Share Competition", 
            options=['Low', 'Medium', 'High'],
            value='Medium',
            help="Describes the level of competitive landscape for market share."
        )
        conversion_rate_sensitivity = st.select_slider(
            "Conv. Rate Stability", 
            options=['Reducing', 'Stable', 'Increasing'],
            value='Stable',
            help="Indicates the expected trend in conversion rates."
        )
        diminishing_returns = st.select_slider(
            "Diminishing Returns", 
            options=['Low', 'Medium', 'High'],
            value='Medium',
            help="Indicates how quickly additional spending yields smaller returns."
        )

# (Existing file upload and forecast generation code remains the same)

# Modify the metrics comparison section
if 'forecast' in locals():
    st.subheader("Annual Metrics Comparison")
    
    # Calculate annual totals
    metrics_to_compare = {
        'Annual Budget': ('Cost', 'Weekly_Budget'),
        'Annual Revenue': ('Revenue', 'Projected_Revenue'),
        'Annual Transactions': ('Transactions', 'Projected_Transactions'),
        'Annual Clicks': ('Clicks', 'Projected_Clicks'),
        'Annual ROAS': ('ROAS', 'Projected_ROAS')
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for i, (display_name, (historical_col, forecast_col)) in enumerate(metrics_to_compare.items()):
        # Calculate annual totals
        historical_value = data[historical_col].sum()
        forecast_value = forecast[forecast_col].sum()
        pct_change = ((forecast_value - historical_value) / historical_value) * 100
        
        with [col1, col2, col3, col4, col5][i]:
            delta_class = 'negative' if pct_change < 0 else ''
            st.metric(
                display_name, 
                f"{forecast_value:,.2f}", 
                delta=f"{pct_change:.2f}%",
                delta_color='inverse'
            )
    
    # Forecast preview with scrollable table
    st.subheader("Forecast Preview")
    st.dataframe(forecast.head(10), height=300)

# (Rest of the existing code remains the same)
