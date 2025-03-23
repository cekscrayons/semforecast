import streamlit as st
import pandas as pd
import io
from sem_forecast_model import SEMForecastModel, load_data_from_csv

# Set page configuration
st.set_page_config(page_title="SEM Forecast Model", layout="wide")

# Title
st.title("SEM Forecast Model")

# Sidebar for model parameters
with st.sidebar:
    st.header("Forecast Configuration")
    
    # Core model parameters with sliders
    min_roas = st.slider("Minimum ROAS Threshold ($)", 10, 30, 20, 1)
    growth_factor = st.slider("YOY Growth Factor (%)", 0, 30, 10, 1) / 100 + 1.0
    aov_growth = st.slider("YOY AOV Growth (%)", 0, 20, 5, 1) / 100 + 1.0
    cpc_inflation = st.slider("YOY CPC Inflation (%)", 0, 15, 2, 1) / 100 + 1.0

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
            
            # Run forecast
            forecast = model.run_forecast()
            
            # Display basic information about the forecast
            st.write("Forecast Generated Successfully")
            st.write(f"Total Weeks Forecasted: {len(forecast)}")
            
            # Get summary statistics
            summary = model.get_summary_stats()
            st.write("Summary Statistics:")
            st.write(summary)
            
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
