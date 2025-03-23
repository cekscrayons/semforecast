import streamlit as st
import pandas as pd
import io
import plotly.graph_objs as go
import plotly.express as px

# Import the forecast model
from sem_forecast_model import SEMForecastModel, load_data_from_csv

# Custom styling
st.set_page_config(
    page_title="SEM Forecast Model", 
    layout="wide", 
    page_icon="📊"
)

# Custom CSS for dark theme
st.markdown("""
<style>
.stApp {
    background-color: #212121;
}
.stSidebar {
    background-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# Mapping function for slider values
def map_slider_to_value(slider_value, parameter_type):
    mappings = {
        'impression_share_growth': {
            'Conservative': 1.1,
            'Moderate': 1.5,
            'Aggressive': 2.0
        },
        'conversion_rate_sensitivity': {
            'Conservative Market': 0.95,
            'Balanced Market': 0.85,
            'Competitive Market': 0.75
        },
        'diminishing_returns': {
            'Low Impact': 0.95,
            'Moderate Impact': 0.85,
            'High Impact': 0.75
        }
    }
    return mappings[parameter_type][slider_value]

# Visualization function
def create_spend_comparison_chart(historical_data, forecast_data):
    # Prepare data for plotting
    historical_spend = historical_data[['Week', 'Cost']].rename(columns={'Cost': 'Spend'})
    historical_spend['Type'] = 'Historical'
    
    forecast_spend = forecast_data[['Week', 'Weekly_Budget']].rename(columns={'Weekly_Budget': 'Spend'})
    forecast_spend['Type'] = 'Forecast'
    
    # Combine datasets
    combined_spend = pd.concat([historical_spend, forecast_spend])
    
    # Create Plotly figure
    fig = px.line(combined_spend, x='Week', y='Spend', color='Type', 
                  title='Historical vs Forecasted Weekly Spend',
                  labels={'Spend': 'Weekly Budget ($)'},
                  color_discrete_map={
                      'Historical': '#4287f5',  # Blue for historical
                      'Forecast': '#42f554'     # Green for forecast
                  })
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    return fig

# Main Streamlit app
st.title("SEM Budget Forecasting Tool")

# Sidebar for model parameters
with st.sidebar:
    st.header("Forecast Configuration")
    
    # Default parameters matching original PyCharm model
    min_roas = st.slider("Minimum ROAS Threshold ($)", 10, 30, 20, 1)
    growth_factor = st.slider("YOY Growth Factor (%)", 0, 30, 10, 1) / 100 + 1.0
    aov_growth = st.slider("YOY AOV Growth (%)", 0, 20, 5, 1) / 100 + 1.0
    cpc_inflation = st.slider("YOY CPC Inflation (%)", 0, 15, 2, 1) / 100 + 1.0
    
    # New categorical sliders with default values matching original model assumptions
    impression_share_growth = st.select_slider(
        "Impression Share Growth", 
        options=['Conservative', 'Moderate', 'Aggressive'],
        value='Moderate',
        help="Determines how quickly you can expand market share:\n"
             "- Conservative: Minimal, careful growth\n"
             "- Moderate: Balanced, steady expansion\n"
             "- Aggressive: Rapid market share capture"
    )

    conversion_rate_sensitivity = st.select_slider(
        "Conversion Rate Sensitivity", 
        options=['Conservative Market', 'Balanced Market', 'Competitive Market'],
        value='Balanced Market',
        help="Describes market conditions affecting conversion rates:\n"
             "- Conservative Market: Stable customer base, predictable conversions\n"
             "- Balanced Market: Moderate competition, typical conversion fluctuations\n"
             "- Competitive Market: High competition, more significant conversion challenges"
    )

    diminishing_returns = st.select_slider(
        "Return on Incremental Spend", 
        options=['Low Impact', 'Moderate Impact', 'High Impact'],
        value='Moderate Impact',
        help="Indicates how quickly additional spending yields smaller returns:\n"
             "- Low Impact: Spending remains consistently effective\n"
             "- Moderate Impact: Gradual decrease in returns\n"
             "- High Impact: Rapid decline in spending efficiency"
    )

# File uploader
uploaded_file = st.file_uploader("Upload CSV with historical SEM data", type=['csv'])

# Run forecast button
if st.button("Generate Forecast"):
    if uploaded_file is not None:
        try:
            # Load data using the model's data loading function
            data = load_data_from_csv(uploaded_file)
            
            # Display input data sample
            st.subheader("Input Data Sample")
            st.dataframe(data.head())
            
            # Initialize and run forecast model
            model = SEMForecastModel(
                data, 
                min_roas_threshold=min_roas, 
                yoy_growth=growth_factor, 
                yoy_cpc_inflation=cpc_inflation, 
                yoy_aov_growth=aov_growth
            )
            
            # Add new parameter mappings
            model.impression_share_multiplier = map_slider_to_value(
                impression_share_growth, 'impression_share_growth'
            )
            model.conversion_rate_sensitivity = map_slider_to_value(
                conversion_rate_sensitivity, 'conversion_rate_sensitivity'
            )
            model.diminishing_returns_factor = map_slider_to_value(
                diminishing_returns, 'diminishing_returns'
            )
            
            # Run forecast
            forecast = model.run_forecast()
            
            # Display basic information about the forecast
            st.success("Forecast Generated Successfully")
            st.write(f"Total Weeks Forecasted: {len(forecast)}")
            
            # Get summary statistics
            summary = model.get_summary_stats()
            st.write("Summary Statistics:")
            st.write(summary)
            
            # Create spend comparison chart
            forecast_chart = create_spend_comparison_chart(data, forecast)
            st.plotly_chart(forecast_chart, use_container_width=True)
            
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
