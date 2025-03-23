import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Set page title and configuration
st.set_page_config(page_title="SEM Budget Forecasting Tool", layout="wide")

# Create header
st.title("BigW SEM Budget Forecasting Tool")
st.markdown("### Optimize your Google Shopping budgets with advanced forecasting")

# Sidebar for inputs
with st.sidebar:
    st.header("Forecast Settings")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file with historical SEM data", type=["csv"])
    
    # Model parameters
    st.subheader("Model Parameters")
    min_roas = st.slider("Minimum ROAS Threshold ($)", 10, 30, 20, 1)
    growth_factor = st.slider("YOY Growth Factor (%)", 0, 30, 10, 1) / 100 + 1.0
    aov_growth = st.slider("YOY AOV Growth (%)", 0, 20, 5, 1) / 100 + 1.0
    cpc_inflation = st.slider("YOY CPC Inflation (%)", 0, 15, 2, 1) / 100 + 1.0
    
    run_button = st.button("Run Forecast", type="primary")

# Main content area
if uploaded_file is None:
    st.info("👈 Upload your SEM data CSV file to get started")
    
    # Sample dashboard preview
    st.subheader("Sample Dashboard Preview")
    
    # Create some sample data for visualization
    sample_dates = pd.date_range(start='2025-07-01', periods=52, freq='W')
    sample_data = pd.DataFrame({
        'Week': sample_dates,
        'Weekly_Budget': np.random.normal(20000, 5000, 52),
        'Projected_ROAS': np.random.normal(25, 3, 52),
        'Projected_CPC': np.random.normal(0.45, 0.1, 52),
        'Projected_Impr_Share': np.random.normal(65, 10, 52),
        'Projected_Conv_Rate': np.random.normal(2.5, 0.3, 52)
    })
    
    # Display sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly budget chart
        fig1 = px.line(sample_data, x='Week', y='Weekly_Budget', 
                      title='Projected Weekly Budget')
        fig1.update_layout(yaxis_title="Budget ($)")
        st.plotly_chart(fig1, use_container_width=True)
        
        # ROAS chart
        fig2 = px.line(sample_data, x='Week', y='Projected_ROAS',
                      title='Projected Weekly ROAS')
        fig2.update_layout(yaxis_title="ROAS ($)")
        fig2.add_hline(y=20, line_dash="dash", line_color="red", 
                      annotation_text="Min ROAS Threshold")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # CPC and Conversion Rate Chart
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sample_data['Week'], y=sample_data['Projected_CPC'],
                                mode='lines', name='CPC ($)'))
        fig3.add_trace(go.Scatter(x=sample_data['Week'], y=sample_data['Projected_Conv_Rate'],
                                mode='lines', name='Conv. Rate (%)', yaxis="y2"))
        fig3.update_layout(
            title='CPC and Conversion Rate Trends',
            yaxis=dict(title="CPC ($)"),
            yaxis2=dict(title="Conv. Rate (%)", overlaying="y", side="right")
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Impression Share chart
        fig4 = px.line(sample_data, x='Week', y='Projected_Impr_Share',
                      title='Projected Impression Share')
        fig4.update_layout(yaxis_title="Impression Share (%)")
        st.plotly_chart(fig4, use_container_width=True)
    
    # Annual summary metrics
    st.subheader("Annual Summary Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Annual Budget", "$1,028,345")
    with metric_col2:
        st.metric("Total Annual Revenue", "$25,708,625")
    with metric_col3:
        st.metric("Overall Annual ROAS", "$25.00")
    with metric_col4:
        st.metric("Avg. CPC", "$0.47", delta="2%")

else:
    # Here you would:
    # 1. Read the uploaded file
    # 2. Run your model
    # 3. Display the results
    
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")
        
        # Display sample of data
        st.subheader("Input Data Sample")
        st.dataframe(data.head())
        
        if run_button:
            with st.spinner("Running forecast..."):
                # Simulate processing time
                import time
                time.sleep(2)
                
                # Normally you would call your model here
                # forecast = run_category_forecast(data, min_roas, growth_factor, aov_growth, cpc_inflation)
                
                # For now, just use the sample data
                sample_dates = pd.date_range(start='2025-07-01', periods=52, freq='W')
                forecast = pd.DataFrame({
                    'Week': sample_dates,
                    'Weekly_Budget': np.random.normal(20000, 5000, 52),
                    'Projected_Revenue': np.random.normal(500000, 100000, 52),
                    'Projected_ROAS': np.random.normal(25, 3, 52),
                    'Projected_CPC': np.random.normal(0.45, 0.1, 52),
                    'Projected_Impr_Share': np.random.normal(65, 10, 52),
                    'Projected_Conv_Rate': np.random.normal(2.5, 0.3, 52)
                })
                
                st.success("Forecast completed!")
                
                # Display results
                st.subheader("Forecast Results")
                
                # Summary metrics
                total_budget = forecast['Weekly_Budget'].sum()
                total_revenue = forecast['Projected_Revenue'].sum()
                overall_roas = total_revenue / total_budget
                avg_cpc = 0.47
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Total Annual Budget", f"${total_budget:,.2f}")
                with metric_col2:
                    st.metric("Total Annual Revenue", f"${total_revenue:,.2f}")
                with metric_col3:
                    st.metric("Overall Annual ROAS", f"${overall_roas:.2f}")
                with metric_col4:
                    st.metric("Avg. CPC", f"${avg_cpc:.2f}", delta="2%")
                
                # Visualizations
                st.subheader("Budget and ROAS Forecast")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Weekly budget chart
                    fig1 = px.line(forecast, x='Week', y='Weekly_Budget', 
                                title='Projected Weekly Budget')
                    fig1.update_layout(yaxis_title="Budget ($)")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # ROAS chart
                    fig2 = px.line(forecast, x='Week', y='Projected_ROAS',
                                title='Projected Weekly ROAS')
                    fig2.update_layout(yaxis_title="ROAS ($)")
                    fig2.add_hline(y=min_roas, line_dash="dash", line_color="red", 
                                annotation_text="Min ROAS Threshold")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Download link for forecast
                csv_buffer = io.StringIO()
                forecast.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv_str,
                    file_name=f"SEM_Forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
