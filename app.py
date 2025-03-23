import streamlit as st
import pandas as pd
import io
import plotly.graph_objs as go
import plotly.express as px

# Import the forecast model
from sem_forecast_model import SEMForecastModel, load_data_from_csv

# Page configuration
st.set_page_config(
    page_title="SEM Forecast Model", 
    layout="wide", 
    page_icon="📊"
)

# Custom CSS for improved dark theme and UX
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

# Mapping function with clear, simple mappings
def map_slider_to_value(slider_value, parameter_type):
    mappings = {
        'impression_share_growth': {
            'Conservative': 1.1,
            'Moderate': 1.5,
            'Aggressive': 2.0
        },
        'conversion_rate_sensitivity': {
            'Conservative': 0.95,
            'Balanced': 0.85,
            'Competitive': 0.75
        },
        'diminishing_returns': {
            'Low': 0.95,
            'Moderate': 0.85,
            'High': 0.75
        }
    }
    return mappings[parameter_type][slider_value]

def create_smooth_comparison_chart(historical_data, forecast_data, metric):
    # Prepare data for plotting
    historical = historical_data[['Week', metric]].copy()
    historical['Type'] = 'Historical'
    
    forecast = forecast_data[['Week', f'Projected_{metric.capitalize()}']].copy()
    forecast['Type'] = 'Forecast'
    
    # Rename columns for consistency
    historical.columns = ['Week', 'Value', 'Type']
    forecast.columns = ['Week', 'Value', 'Type']
    
    # Combine datasets
    combined_data = pd.concat([historical, forecast])
    
    # Create Plotly figure with smooth lines
    fig = go.Figure()
    
    # Historical data (light grey)
    historical_trace = go.Scatter(
        x=historical['Week'], 
        y=historical['Value'], 
        mode='lines', 
        name='Historical', 
        line=dict(color='#666666', width=2),
        hovertemplate='%{y:.2f}<extra></extra>'
    )
    
    # Forecast data (bright color)
    forecast_trace = go.Scatter(
        x=forecast['Week'], 
        y=forecast['Value'], 
        mode='lines', 
        name='Forecast', 
        line=dict(color='#42f554', width=3),
        hovertemplate='%{y:.2f}<extra></extra>'
    )
    
    fig.add_trace(historical_trace)
    fig.add_trace(forecast_trace)
    
    # Customize layout
    fig.update_layout(
        title=f'Historical vs Forecasted Weekly {metric.capitalize()}',
        xaxis_title='Week',
        yaxis_title=metric.capitalize(),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend_title_text='Type'
    )
    
    return fig

# Main Streamlit app
st.title("SEM Budget Forecasting Tool")

# Sidebar with simplified controls
with st.sidebar:
    st.header("Forecast Configuration")
    
    # Core parameters
    min_roas = st.slider("ROAS Threshold", 10, 30, 20, 1)
    growth_factor = st.slider("YOY Growth", 0, 30, 10, 1) / 100 + 1.0
    aov_growth = st.slider("AOV Growth", 0, 20, 5, 1) / 100 + 1.0
    cpc_inflation = st.slider("CPC Inflation", 0, 15, 2, 1) / 100 + 1.0
    
    # Advanced options with categorical sliders
    impression_share_growth = st.select_slider(
        "Impression Share", 
        options=['Conservative', 'Moderate', 'Aggressive'],
        value='Moderate'
    )
    conversion_rate_sensitivity = st.select_slider(
        "Market Sensitivity", 
        options=['Conservative', 'Balanced', 'Competitive'],
        value='Balanced'
    )
    diminishing_returns = st.select_slider(
        "Return Impact", 
        options=['Low', 'Moderate', 'High'],
        value='Moderate'
    )

# File uploader
uploaded_file = st.file_uploader("Upload CSV with historical SEM data", type=['csv'])

# Run forecast button
if st.button("Generate Forecast"):
    if uploaded_file is not None:
        try:
            # Load data
            data = load_data_from_csv(uploaded_file)
            
            # Initialize and run forecast model
            model = SEMForecastModel(
                data, 
                min_roas_threshold=min_roas, 
                yoy_growth=growth_factor, 
                yoy_cpc_inflation=cpc_inflation, 
                yoy_aov_growth=aov_growth
            )
            
            # Add new parameter mappings with error handling
            try:
                is_multiplier = map_slider_to_value(
                    impression_share_growth, 'impression_share_growth'
                )
                conv_sensitivity = map_slider_to_value(
                    conversion_rate_sensitivity, 'conversion_rate_sensitivity'
                )
                dim_returns = map_slider_to_value(
                    diminishing_returns, 'diminishing_returns'
                )
                
                # Dynamically add attributes if the model supports them
                if hasattr(model, 'impression_share_multiplier'):
                    model.impression_share_multiplier = is_multiplier
                if hasattr(model, 'conversion_rate_sensitivity'):
                    model.conversion_rate_sensitivity = conv_sensitivity
                if hasattr(model, 'diminishing_returns_factor'):
                    model.diminishing_returns_factor = dim_returns
            except Exception as mapping_error:
                st.warning(f"Could not apply advanced parameters: {mapping_error}")
            
            # Run forecast
            forecast = model.run_forecast()
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                spend_chart = create_smooth_comparison_chart(data, forecast, 'Cost')
                st.plotly_chart(spend_chart, use_container_width=True)
            
            with col2:
                revenue_chart = create_smooth_comparison_chart(data, forecast, 'Revenue')
                st.plotly_chart(revenue_chart, use_container_width=True)
            
            # Summary statistics
            summary = model.get_summary_stats()
            st.write("Summary Statistics:")
            st.write(summary)
            
            # Download forecast
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
