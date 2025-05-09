import streamlit as st
import pandas as pd
import io

# Import the forecast model
from sem_forecast_model import SEMForecastModel, load_data_from_csv

# Page configuration
st.set_page_config(
    page_title="Annual SEM Budget Forecasting Tool", 
    layout="wide", 
    page_icon="📊"
)

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

# File uploader
uploaded_file = st.file_uploader("Upload CSV with historical SEM data", type=['csv'])

# Run forecast button
if st.button("Generate Forecast"):
    if uploaded_file is not None:
        try:
            # Load data using the model's data loading function
            data = load_data_from_csv(uploaded_file)
            
            # Initialize and run forecast model
            model = SEMForecastModel(
                data, 
                min_roas_threshold=min_roas, 
                yoy_growth=growth_factor, 
                yoy_cpc_inflation=cpc_inflation, 
                yoy_aov_growth=aov_growth
            )
            
            # Add new parameter mappings
            try:
                # Directly map and assign advanced controls
                model.impression_share_multiplier = map_slider_to_value(
                    impression_share_growth, 'impression_share_growth'
                )
                model.conversion_rate_sensitivity = map_slider_to_value(
                    conversion_rate_sensitivity, 'conversion_rate_sensitivity'
                )
                model.diminishing_returns_factor = map_slider_to_value(
                    diminishing_returns, 'diminishing_returns'
                )
            except Exception as mapping_error:
                st.warning(f"Could not apply advanced parameters: {mapping_error}")
            
            # Run forecast
            forecast = model.run_forecast()
            
            # Annual Metrics Comparison
            st.subheader("Annual Metrics Comparison")
            
            # Prepare comparison metrics
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
                historical_value = data[historical_col].sum() if historical_col in data.columns else data[historical_col].mean() * 52
                forecast_value = forecast[forecast_col].sum()
                pct_change = ((forecast_value - historical_value) / historical_value) * 100
                
                with [col1, col2, col3, col4, col5][i]:
                    # Special handling for ROAS
                    if display_name == 'Annual ROAS':
                        forecast_value = forecast['Projected_Revenue'].sum() / forecast['Weekly_Budget'].sum()
                        historical_value = data['Revenue'].sum() / data['Cost'].sum()
                        pct_change = ((forecast_value - historical_value) / historical_value) * 100
                        st.metric(
                            display_name, 
                            f"${forecast_value:.2f}", 
                            delta=f"{pct_change:.2f}%",
                            delta_color='off'
                        )
                    # Special handling for Clicks and Transactions (0 decimal places)
                    elif display_name in ['Annual Clicks', 'Annual Transactions']:
                        st.metric(
                            display_name, 
                            f"{forecast_value:,.0f}", 
                            delta=f"{pct_change:.2f}%",
                            delta_color='off'
                        )
                    # Default handling for other metrics
                    else:
                        st.metric(
                            display_name, 
                            f"{forecast_value:,.2f}", 
                            delta=f"{pct_change:.2f}%",
                            delta_color='off'
                        )
            
            # Forecast preview with scrollable table
            st.subheader("Forecast Preview")
            st.dataframe(forecast, height=400)
            
            # Create download button for forecast
            csv_buffer = io.StringIO()
            forecast.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            st.download_button(
                label="Download Forecast CSV",
                data=csv_str,
                file_name="sem_forecast.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
    else:
        st.warning("Please upload a CSV file first")
