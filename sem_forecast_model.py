import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math


class SEMForecastModel:
    """
    Enhanced SEM Forecast Model that implements the 12 key factors with realistic market dynamics:
    1. Impression Share Opportunity
    2. Non-Linear CPC Scaling
    3. Conversion Rate by Spend Level
    4. Seasonal Performance Patterns
    5. Efficiency Threshold Adjustment
    6. Quality Score Consideration
    7. Diminishing Returns Model
    8. Year-over-Year Growth Factor
    9. CPC Tier Analysis
    10. Average Order Value Stability
    11. Realistic Auction Dynamics
    12. Flexible ROAS Target with Variance
    """

    def __init__(self, data, min_roas_threshold=20, yoy_growth=1.10,
                 yoy_cpc_inflation=1.02, yoy_aov_growth=1.05):
        """
        Initialize the forecast model with historical data

        Parameters:
        -----------
        data : pandas.DataFrame
            Historical data with required columns
        min_roas_threshold : float
            Minimum ROAS threshold for efficiency adjustment (default: $20)
        yoy_growth : float
            Year-over-year growth factor (default: 10%)
        yoy_cpc_inflation : float
            Year-over-year CPC inflation factor (default: 2%)
        yoy_aov_growth : float
            Year-over-year AOV growth factor (default: 5%)
        """
        self.data = data.copy()
        self.min_roas_threshold = min_roas_threshold
        self.yoy_growth = yoy_growth
        self.yoy_cpc_inflation = yoy_cpc_inflation
        self.yoy_aov_growth = yoy_aov_growth
        
        # Initialize the advanced parameters with default values
        self.impression_share_multiplier = 1.5  # Medium (default)
        self.conversion_rate_sensitivity = 0.85  # Stable (default)
        self.diminishing_returns_factor = 0.85  # Medium (default)

        # Ensure Week is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['Week']):
            self.data['Week'] = pd.to_datetime(self.data['Week'])

        # Sort by date
        self.data = self.data.sort_values('Week')

        # Add week number for seasonal mapping
        self.data['Week_Number'] = self.data['Week'].dt.isocalendar().week

        # Identify Black Friday periods (for specific handling)
        self.data['Is_BlackFriday'] = (
                ((self.data['Week'].dt.month == 11) & (self.data['Week'].dt.day >= 20)) |
                ((self.data['Week'].dt.month == 12) & (self.data['Week'].dt.day <= 5))
        )

        # Calculate global reference metrics
        self.avg_cpc = self.data['Avg_CPC'].mean()
        self.min_roas_headroom = 0.15  # 15% above min threshold considered risky

        # Initialize forecast dataframe
        self.forecast = None
        self.final_forecast = None

        # Initialize forecast_dates
        self.forecast_dates = None

    def preprocess_data(self):
        """Calculate derived metrics from raw data"""

        # 10. Average Order Value Stability - Calculate AOV
        self.data['AOV'] = self.data['Revenue'] / self.data['Transactions']

        # 9. CPC Tier Analysis - Create CPC tiers for analysis
        self.data['CPC_Tier'] = pd.qcut(self.data['Avg_CPC'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

        # Calculate ROAS headroom (how far above threshold)
        self.data['ROAS_Headroom'] = (self.data['ROAS'] - self.min_roas_threshold) / self.min_roas_threshold

        # Flag high performing weeks (for protection logic) with tiers
        self.data['High_Performing'] = self.data['ROAS'] > (self.min_roas_threshold * 1.5)
        # Add performance tiers to better identify extremely strong weeks
        self.data['Performance_Tier'] = 'Normal'
        self.data.loc[self.data['ROAS'] > (self.min_roas_threshold * 1.5), 'Performance_Tier'] = 'Strong'
        self.data.loc[self.data['ROAS'] > (self.min_roas_threshold * 2.0), 'Performance_Tier'] = 'Excellent'
        self.data.loc[self.data['ROAS'] > (self.min_roas_threshold * 3.0), 'Performance_Tier'] = 'Exceptional'

        # 1. Impression Share Opportunity - Calculate potential impression share
        self.data['Potential_Impr_Share'] = self.data['Impr_Share'] + self.data['Lost_IS_Budget']

        # Calculate impression share multiplier (how much room for growth exists)
        # Account for already-high CPCs and low ROAS headroom
        def calculate_is_multiplier(row):
            base_multiplier = row['Potential_Impr_Share'] / row['Impr_Share'] if row['Impr_Share'] > 0 else 1.0
            
            # Apply the external impression_share_multiplier to base_multiplier
            # This will boost or reduce the multiplier based on the advanced control setting
            base_multiplier = base_multiplier * self.impression_share_multiplier / 1.5  # Normalized to default of 1.5

            # UPDATED: More conservative caps on impression share growth
            # Extreme Lost IS Budget requires more caution
            if row['Lost_IS_Budget'] > 80:
                base_multiplier = min(base_multiplier, 1.2)  # More conservative (was 1.3)
            elif row['Lost_IS_Budget'] > 50:
                base_multiplier = min(base_multiplier, 1.4)  # More conservative (was 1.8)

            # CPC relative to average affects scaling (higher CPC = more conservative)
            cpc_ratio = row['Avg_CPC'] / self.avg_cpc if self.avg_cpc > 0 else 1.0
            if cpc_ratio > 1.5:  # CPC significantly above average
                base_multiplier = min(base_multiplier, 1.1)  # More conservative (was 1.2)
            elif cpc_ratio > 1.2:
                base_multiplier = min(base_multiplier, 1.3)  # More conservative (was 1.5)

            # Low ROAS headroom means less room to grow spend while maintaining efficiency
            if row['ROAS_Headroom'] < 0.2:  # Within 20% of threshold
                base_multiplier = min(base_multiplier, 1.08)  # More conservative (was 1.13)
            elif row['ROAS_Headroom'] < 0.5:
                base_multiplier = min(base_multiplier, 1.15)  # More conservative (was 1.3)

            # For periods with very high spend, even more conservative
            if row['Cost'] > 80000:
                base_multiplier = min(base_multiplier, 1.08)  # More conservative (was 1.13)

            return base_multiplier

        self.data['IS_Multiplier'] = self.data.apply(calculate_is_multiplier, axis=1)

        # 6. Quality Score Consideration - Use Lost IS Rank to adjust impression share opportunity
        # Lower Lost IS Rank indicates better quality scores and more efficient growth potential
        self.data['Quality_Score_Factor'] = 1 - (self.data['Lost_IS_Rank'] / 100)

        # Adjust IS Multiplier based on quality score
        self.data['Adjusted_IS_Multiplier'] = self.data['IS_Multiplier'] * (1 + self.data['Quality_Score_Factor'])

        # UPDATED: Apply more conservative cap for all periods
        self.data['Adjusted_IS_Multiplier'] = self.data['Adjusted_IS_Multiplier'].clip(
            upper=2.5)  # More conservative (was 3.0)

        return self

    def apply_cpc_scaling(self):
        """
        2. Non-Linear CPC Scaling
        Higher impression share typically leads to higher CPCs due to competition
        """

        def calculate_cpc_scaling(row):
            # Calculate what portion of the potential impression share we're currently capturing
            captured_portion = row['Impr_Share'] / row['Potential_Impr_Share'] if row['Potential_Impr_Share'] > 0 else 1

            # First, check if this is a high-performing week and adjust base scaling
            if 'Performance_Tier' in row:
                if row['Performance_Tier'] == 'Exceptional' or row['Performance_Tier'] == 'Excellent':
                    # Lower CPC scaling for high-performing weeks - they can absorb more traffic efficiently
                    if captured_portion < 0.5:
                        base_scaling = 1.15  # Less aggressive for efficient weeks
                    elif captured_portion < 0.75:
                        base_scaling = 1.25  # Less aggressive for efficient weeks
                    else:
                        base_scaling = 1.35  # Less aggressive for efficient weeks

                    # For high-performing weeks with low impression share, we want to be less aggressive with CPC
                    # to allow for more impression share growth
                    if row['Impr_Share'] < 40 and row['ROAS_Headroom'] > 0.7:
                        base_scaling = base_scaling * 0.9  # Further reduce CPC scaling

                    # Apply specialized scaling for high-performing weeks
                    return base_scaling

            # Standard case - normal weeks
            # UPDATED: More aggressive base scaling based on captured portion
            if captured_portion < 0.5:
                base_scaling = 1.20  # More aggressive (was 1.15)
            elif captured_portion < 0.75:
                base_scaling = 1.35  # More aggressive (was 1.25)
            else:
                base_scaling = 1.50  # More aggressive (was 1.35)

            # Apply additional scaling when:
            # 1. It's a Black Friday period
            # 2. The CPC is already significantly above average
            # 3. Lost IS Budget is very high
            cpc_ratio = row['Avg_CPC'] / self.avg_cpc if self.avg_cpc > 0 else 1.0
            lost_budget_extreme = row['Lost_IS_Budget'] > 70

            # UPDATED: More aggressive special case scaling
            if row['Is_BlackFriday'] and cpc_ratio > 1.5 and lost_budget_extreme:
                # This combination represents our Black Friday week with high CPC
                # and high lost impression share - requires extremely aggressive scaling
                return base_scaling * 2.8  # More aggressive (was 2.5)
            elif row['Is_BlackFriday'] and cpc_ratio > 1.2:
                return base_scaling * 2.0  # More aggressive (was 1.8)
            elif lost_budget_extreme or cpc_ratio > 1.5:
                return base_scaling * 1.7  # More aggressive (was 1.5)
            else:
                return base_scaling

        self.data['CPC_Scaling'] = self.data.apply(calculate_cpc_scaling, axis=1)
        return self

    def apply_conversion_scaling(self):
        """
        3. Conversion Rate by Spend Level
        Higher spend often leads to lower conversion rates as you reach less qualified users
        """

        def calculate_conversion_scaling(row):
            # Use the externally provided conversion_rate_sensitivity
            # The sensitivity factor affects how much conversion rates decline with increasing spend
            sensitivity_factor = self.conversion_rate_sensitivity
            
            # Lower factor (like 0.75) makes conversion rates more sensitive to increased spend
            # Higher factor (like 0.95) makes conversion rates less sensitive to increased spend
            if row['Cost'] <= 7500:
                base_scaling = 1.0  # Baseline conversion rate
            elif row['Cost'] <= 15000:
                base_scaling = 1.0 - (0.05 * (0.85 / sensitivity_factor))  # Adjusted decrease
            elif row['Cost'] <= 30000:
                base_scaling = 1.0 - (0.10 * (0.85 / sensitivity_factor))  # Adjusted decrease
            else:
                base_scaling = 1.0 - (0.15 * (0.85 / sensitivity_factor))  # Adjusted decrease

            # For Black Friday periods, even with higher spend, conversion intent might be stronger
            if row['Is_BlackFriday']:
                # Apply less severe decrease for Black Friday periods
                base_scaling = max(base_scaling, 0.92)  # Floor at 8% decrease for Black Friday

            return base_scaling

        self.data['Conv_Rate_Scaling'] = self.data.apply(calculate_conversion_scaling, axis=1)
        return self

    def apply_diminishing_returns(self):
        """
        7. Diminishing Returns Model
        Each additional dollar of spend yields progressively less return
        """

        def calculate_diminishing_returns_factor(row):
            # Use the class property for diminishing returns
            external_factor = self.diminishing_returns_factor
            
            # Check for high-performing weeks first - these get special treatment
            if 'Performance_Tier' in row:
                # Relaxed diminishing returns for high-performing weeks
                if row['Performance_Tier'] == 'Exceptional':
                    # Almost no diminishing returns for exceptional weeks
                    return 0.98 * (external_factor / 0.85)  # Normalize to default 0.85
                elif row['Performance_Tier'] == 'Excellent':
                    # Very minimal diminishing returns for excellent weeks
                    return 0.95 * (external_factor / 0.85)
                elif row['Performance_Tier'] == 'Strong' and row['ROAS_Headroom'] > 0.5:
                    # Reduced diminishing returns for strong weeks with good headroom
                    return 0.92 * (external_factor / 0.85)

            # Standard diminishing returns factors for normal weeks - adjusted by external factor
            if row['Adjusted_Cost'] <= 20000:
                base_factor = 1.0  # No diminishing returns at lower spend levels
            elif row['Adjusted_Cost'] <= 40000:
                base_factor = 0.92 * (external_factor / 0.85)
            elif row['Adjusted_Cost'] <= 80000:
                base_factor = 0.84 * (external_factor / 0.85)
            else:
                base_factor = 0.76 * (external_factor / 0.85)

            # Apply much stronger diminishing returns for specific scenarios
            # UPDATED: More aggressive special case handling
            # Scenario 1: Black Friday period with already high spend and low ROAS headroom
            if row['Is_BlackFriday'] and row['Cost'] > 90000 and row['ROAS_Headroom'] < 0.2:
                return base_factor * 0.45  # More aggressive (was 0.5)

            # Scenario 2: Black Friday period with high spend
            elif row['Is_BlackFriday'] and row['Cost'] > 50000:
                return base_factor * 0.65  # More aggressive (was 0.7)

            # Scenario 3: Any period with extremely high adjusted cost
            elif row['Adjusted_Cost'] > 200000:
                return base_factor * 0.55  # More aggressive (was 0.6)

            # Scenario 4: Any period with high spend and low ROAS headroom
            elif row['Cost'] > 50000 and row['ROAS_Headroom'] < 0.3:
                # But don't apply this penalty to high-performing weeks
                if row.get('High_Performing', False):
                    return base_factor * 0.9  # Less aggressive for high performers
                return base_factor * 0.75  # More aggressive (was 0.8)

            return base_factor

        self.data['Diminishing_Returns_Factor'] = self.data.apply(calculate_diminishing_returns_factor, axis=1)
        return self

    def calculate_adjusted_metrics(self):
        """Calculate adjusted metrics based on scaling factors"""

        # Calculate adjusted cost for maximum impression share
        self.data['Adjusted_Cost'] = self.data['Cost'] * self.data['Adjusted_IS_Multiplier']

        # Apply diminishing returns to adjusted cost
        self.apply_diminishing_returns()
        self.data['Adjusted_Cost'] = self.data['Adjusted_Cost'] * self.data['Diminishing_Returns_Factor']

        # Calculate adjusted CPC for higher impression share
        self.data['Adjusted_CPC'] = self.data['Avg_CPC'] * self.data['CPC_Scaling']

        # Recalculate clicks based on adjusted cost and CPC
        self.data['Adjusted_Clicks'] = self.data['Adjusted_Cost'] / self.data['Adjusted_CPC']

        # Apply conversion rate scaling to adjusted clicks
        self.data['Adjusted_Conv_Rate'] = self.data['Conv_Rate'] * self.data['Conv_Rate_Scaling']

        # Calculate expected transactions
        self.data['Adjusted_Transactions'] = self.data['Adjusted_Clicks'] * (self.data['Adjusted_Conv_Rate'] / 100)

        # Calculate adjusted revenue using AOV stability principle
        self.data['Adjusted_Revenue'] = self.data['Adjusted_Transactions'] * self.data['AOV']

        # Calculate adjusted ROAS
        self.data['Adjusted_ROAS'] = self.data['Adjusted_Revenue'] / self.data['Adjusted_Cost']

        return self

    def apply_efficiency_threshold(self):
        """
        5. Efficiency Threshold Adjustment
        Ensure all projected weeks maintain minimum ROAS
        """
        # Identify weeks that need ROAS adjustment
        self.data['ROAS_Adjustment_Needed'] = self.data['Adjusted_ROAS'] < self.min_roas_threshold

        # Calculate efficiency factor for periods that need adjustment
        self.data.loc[self.data['ROAS_Adjustment_Needed'], 'Efficiency_Factor'] = \
            self.min_roas_threshold / self.data.loc[self.data['ROAS_Adjustment_Needed'], 'Adjusted_ROAS']
        self.data.loc[~self.data['ROAS_Adjustment_Needed'], 'Efficiency_Factor'] = 1.0

        # Apply efficiency adjustment to cost
        self.data['Final_Cost'] = self.data['Adjusted_Cost'] / self.data['Efficiency_Factor']
        self.data['Final_Revenue'] = self.data['Adjusted_Revenue']
        self.data['Final_ROAS'] = self.data['Final_Revenue'] / self.data['Final_Cost']

        # For periods that need efficiency adjustment, recalculate other metrics
        self.data.loc[self.data['ROAS_Adjustment_Needed'], 'Final_Clicks'] = \
            self.data.loc[self.data['ROAS_Adjustment_Needed'], 'Adjusted_Clicks'] / \
            self.data.loc[self.data['ROAS_Adjustment_Needed'], 'Efficiency_Factor']

        self.data.loc[~self.data['ROAS_Adjustment_Needed'], 'Final_Clicks'] = \
            self.data.loc[~self.data['ROAS_Adjustment_Needed'], 'Adjusted_Clicks']

        # Calculate final transactions
        self.data['Final_Transactions'] = self.data['Final_Clicks'] * (self.data['Adjusted_Conv_Rate'] / 100)

        return self

    def generate_forecast_dates(self, start_date=None, periods=52):
        """
        Generate forecast dates for the upcoming year

        Parameters:
        -----------
        start_date : datetime, optional
            Start date for the forecast, defaults to first Sunday of next July
        periods : int, optional
            Number of periods (weeks) to forecast, defaults to 52 (full year)
        """
        if start_date is None:
            # Default to first Sunday of next July
            current_year = datetime.now().year
            if datetime.now().month >= 7:
                forecast_year = current_year + 1
            else:
                forecast_year = current_year

            # Find the first Sunday in July
            first_day = datetime(forecast_year, 7, 1)
            days_until_sunday = 6 - first_day.weekday()  # 6 is Sunday
            if days_until_sunday < 0:
                days_until_sunday += 7

            start_date = first_day + timedelta(days=days_until_sunday)

        # Generate weekly dates
        self.forecast_dates = [start_date + timedelta(weeks=i) for i in range(periods)]

        return self

    def apply_seasonal_patterns(self):
        """
        4. Seasonal Performance Patterns
        Map historical data to forecast based on week number
        """
        # Create forecast dataframe
        self.forecast = pd.DataFrame({
            'Week': self.forecast_dates,
            'Week_Number': [d.isocalendar().week for d in self.forecast_dates]
        })

        # Add Black Friday flag for special handling
        self.forecast['Is_BlackFriday'] = (
                ((self.forecast['Week'].dt.month == 11) & (self.forecast['Week'].dt.day >= 20)) |
                ((self.forecast['Week'].dt.month == 12) & (self.forecast['Week'].dt.day <= 5))
        )

        # Map historical data to forecast based on week number
        # Take averages for weeks that appear multiple times in the historical data
        weekly_metrics = self.data.groupby('Week_Number').agg({
            'Cost': 'mean',
            'Revenue': 'mean',
            'CTR': 'mean',
            'Conv_Rate': 'mean',
            'Avg_CPC': 'mean',
            'ROAS': 'mean',
            'AOV': 'mean',
            'Impr_Share': 'mean',
            'Lost_IS_Budget': 'mean',  # Added for impression share targeting
            'Final_Cost': 'mean',
            'Final_Revenue': 'mean',
            'Final_ROAS': 'mean',
            'Final_Clicks': 'mean',
            'Final_Transactions': 'mean',
            'Search_Volume_Index': 'mean',
            'ROAS_Headroom': 'mean',  # Include ROAS headroom for growth factor adjustments
            'High_Performing': 'max',  # Use max to ensure high performing weeks are identified
            'Performance_Tier': lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else 'Normal',
            # Add performance tier
            'Adjusted_Conv_Rate': 'mean'  # Add adjusted conversion rate for forecast
        }).reset_index()

        # Merge with forecast
        self.forecast = self.forecast.merge(weekly_metrics, on='Week_Number', how='left')

        # Handle missing values - differently for different column types
        for col in self.forecast.columns:
            if col not in ['Week', 'Week_Number', 'Is_BlackFriday']:
                # Skip categorical/text columns
                if col == 'Performance_Tier':
                    self.forecast[col] = self.forecast[col].fillna('Normal')
                elif col == 'High_Performing':
                    self.forecast[col] = self.forecast[col].fillna(False)
                else:
                    # For numeric columns, use mean
                    col_mean = self.forecast[col].mean()
                    self.forecast[col] = self.forecast[col].fillna(col_mean)

        return self

    def apply_growth_factor(self):
        """
        8. Year-over-Year Growth Factor
        Apply growth adjustment to the forecast
        """

        # Function to apply variable growth based on performance tiers, impression share, and more
        def calculate_growth_factor(row):
            base_growth = self.yoy_growth  # Default 10% growth

            # Set specific growth rates based on performance tiers
            if 'Performance_Tier' in row:
                if row['Performance_Tier'] == 'Exceptional':
                    # Exceptional performers get very aggressive growth
                    performance_based_growth = 1.35  # 35% growth
                elif row['Performance_Tier'] == 'Excellent':
                    # Excellent performers get strong growth
                    performance_based_growth = 1.25  # 25% growth
                elif row['Performance_Tier'] == 'Strong':
                    # Strong performers get above-standard growth
                    performance_based_growth = 1.18  # 18% growth
                else:
                    # Normal weeks get standard growth
                    performance_based_growth = base_growth
            else:
                # Default if performance tier not available
                performance_based_growth = base_growth

                # Still check high performing flag as fallback
                if row['High_Performing']:
                    performance_based_growth = max(base_growth, 1.15)

            # Additional growth for weeks with low impression share but good ROAS
            roas_headroom = row['ROAS_Headroom'] if not pd.isna(row['ROAS_Headroom']) else 0.5
            impr_share = row['Impr_Share'] if not pd.isna(row['Impr_Share']) else 50
            lost_budget_pct = row['Lost_IS_Budget'] if not pd.isna(row['Lost_IS_Budget']) else 0

            # If impression share is low but ROAS is good, we have opportunity
            if impr_share < 50 and roas_headroom > 0.3 and lost_budget_pct > 20:
                # Add additional growth based on impression share opportunity
                impression_opportunity_factor = 1 + ((50 - impr_share) / 100)  # Up to 50% additional growth
                performance_based_growth = performance_based_growth * impression_opportunity_factor

                # Cap at reasonable maximum
                performance_based_growth = min(performance_based_growth, 1.5)

            # Special handling for Black Friday periods
            if row['Is_BlackFriday']:
                cost = row['Cost'] if not pd.isna(row['Cost']) else 0

                # Even for Black Friday, prioritize high-performing weeks
                if 'Performance_Tier' in row and row['Performance_Tier'] in ['Exceptional', 'Excellent']:
                    # Keep strong growth for high performers, but slightly more conservative
                    bf_growth = performance_based_growth * 0.9
                # Combination of high cost and low ROAS headroom requires caution
                elif cost > 90000 and roas_headroom < 0.2:
                    bf_growth = base_growth * 0.6  # Less conservative than before
                elif cost > 50000 or roas_headroom < 0.3:
                    bf_growth = base_growth * 0.8  # Less conservative than before
                else:
                    bf_growth = base_growth

                return max(bf_growth, base_growth)  # Never go below base growth

            # Return the performance-based growth factor
            return max(performance_based_growth, base_growth)  # Never go below base growth

        # Calculate adjusted growth factors for each week
        self.forecast['Growth_Factor'] = self.forecast.apply(calculate_growth_factor, axis=1)

        # Apply adjusted year-over-year growth with separate factors for CPC and AOV
        self.forecast['Forecast_CPC'] = self.forecast['Avg_CPC'] * self.yoy_cpc_inflation
        self.forecast['Forecast_AOV'] = self.forecast['AOV'] * self.yoy_aov_growth

        # Apply growth to click volume based on calculated growth factors
        self.forecast['Forecast_Clicks'] = self.forecast['Final_Clicks'] * self.forecast['Growth_Factor']

        # Spending floor protection: Ensure we never spend less YOY for any week
        # Set a minimum growth factor of 1.05 (5% growth) for all spend regardless of performance
        self.forecast['Cost_Growth_Factor'] = self.forecast['Growth_Factor'].clip(lower=1.05)

        # Set direct spend targets for high-performing weeks to ensure significant YOY growth
        high_performers = self.forecast['Performance_Tier'].isin(['Excellent', 'Exceptional'])
        self.forecast.loc[high_performers, 'Cost_Growth_Factor'] = self.forecast.loc[
            high_performers, 'Cost_Growth_Factor'].clip(lower=1.2)

        # Calculate direct cost growth for weeks with growth factor
        self.forecast['Direct_Cost_Growth'] = self.forecast['Cost'] * self.forecast['Cost_Growth_Factor']

        # Recalculate cost based on projected clicks and CPC inflation
        self.forecast['Forecast_Cost'] = self.forecast['Forecast_Clicks'] * self.forecast['Forecast_CPC']

        # Take the maximum of click-based cost and direct cost growth to ensure we're always growing spend
        self.forecast['Forecast_Cost'] = self.forecast[['Forecast_Cost', 'Direct_Cost_Growth']].max(axis=1)

        # Recalculate clicks based on the potentially adjusted cost
        self.forecast['Forecast_Clicks'] = self.forecast['Forecast_Cost'] / self.forecast['Forecast_CPC']

        # Calculate conversion rate (same as final)
        # In case the Adjusted_Conv_Rate column is missing, fallback to Conv_Rate
        if 'Adjusted_Conv_Rate' in self.forecast.columns:
            self.forecast['Forecast_Conv_Rate'] = self.forecast['Adjusted_Conv_Rate']
        else:
            self.forecast['Forecast_Conv_Rate'] = self.forecast['Conv_Rate']

        # Calculate transactions using the forecast clicks and conversion rate
        self.forecast['Forecast_Transactions'] = self.forecast['Forecast_Clicks'] * (
                    self.forecast['Forecast_Conv_Rate'] / 100)

        # Calculate revenue using transactions and AOV growth
        self.forecast['Forecast_Revenue'] = self.forecast['Forecast_Transactions'] * self.forecast['Forecast_AOV']

        # Calculate final ROAS
        self.forecast['Forecast_ROAS'] = self.forecast['Forecast_Revenue'] / self.forecast['Forecast_Cost']

        # Apply ROAS adjustment for any weeks that fall below threshold
        self.forecast['ROAS_Below_Threshold'] = self.forecast['Forecast_ROAS'] < self.min_roas_threshold
        self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Cost'] = (
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Revenue'] / self.min_roas_threshold
        )

        # Recalculate clicks after ROAS adjustment
        self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Clicks'] = (
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Cost'] /
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_CPC']
        )

        # Recalculate transactions after ROAS adjustment
        self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Transactions'] = (
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Clicks'] *
                (self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Conv_Rate'] / 100)
        )

        # Calculate impression share growth - targeting higher impression share based on performance
        base_impr_share = self.forecast['Impr_Share']

        # Avoid variable shadowing by using a different name
        lost_budget_pct = self.forecast['Lost_IS_Budget']

        # Calculate potential impression share gain
        potential_gain = lost_budget_pct * 0.5  # Target capturing up to 50% of lost impression share

        # Scale the potential gain based on performance tier
        scaling_factor = pd.Series(0.3, index=self.forecast.index)  # Base scaling (30% of potential)
        scaling_factor.loc[self.forecast['Performance_Tier'] == 'Strong'] = 0.5  # 50% of potential
        scaling_factor.loc[self.forecast['Performance_Tier'] == 'Excellent'] = 0.7  # 70% of potential
        scaling_factor.loc[self.forecast['Performance_Tier'] == 'Exceptional'] = 0.9  # 90% of potential

        # Calculate adjusted impression share
        target_impr_share = base_impr_share + (potential_gain * scaling_factor)

        # Cap impression share at 100%
        self.forecast['Forecast_Impr_Share'] = target_impr_share.clip(upper=100)

        return self

    def apply_auction_dynamics(self):
        """
        Apply realistic auction dynamics to forecast:
        1. Progressive CPC scaling based on spend increases
        2. Realistic impression share growth based on industry benchmarks
        3. Integrated metrics relationships
        """
        # First, calculate spend growth percentages
        self.forecast['Historical_Cost'] = self.forecast['Cost']
        self.forecast['Spend_Growth_Pct'] = (self.forecast['Forecast_Cost'] / self.forecast[
            'Historical_Cost'] - 1) * 100

        # Apply progressive CPC scaling based on spend increases
        def calculate_progressive_cpc_adjustment(row):
            spend_growth = row['Spend_Growth_Pct']
            base_cpc = row['Forecast_CPC']  # Already has the base 2% inflation

            # Progressive CPC scaling based on spend increase
            if spend_growth > 100:  # More than doubled spend
                return base_cpc * 1.30  # 30% CPC increase
            elif spend_growth > 75:
                return base_cpc * 1.25  # 25% CPC increase
            elif spend_growth > 50:
                return base_cpc * 1.20  # 20% CPC increase
            elif spend_growth > 30:
                return base_cpc * 1.15  # 15% CPC increase
            elif spend_growth > 15:
                return base_cpc * 1.10  # 10% CPC increase
            else:
                return base_cpc  # Keep base CPC

        # Apply the progressive CPC scaling
        self.forecast['Forecast_CPC_Adjusted'] = self.forecast.apply(calculate_progressive_cpc_adjustment, axis=1)

        # Save original values for reference
        self.forecast['Original_Forecast_Cost'] = self.forecast['Forecast_Cost']
        self.forecast['Original_Forecast_Clicks'] = self.forecast['Forecast_Clicks']
        self.forecast['Original_Forecast_Impr_Share'] = self.forecast['Forecast_Impr_Share']

        # Recalculate clicks based on adjusted CPC (maintaining the same spend)
        self.forecast['Forecast_Clicks_Adjusted'] = self.forecast['Forecast_Cost'] / self.forecast[
            'Forecast_CPC_Adjusted']

        # Apply more realistic impression share scaling based on industry standards
        def calculate_realistic_impression_share(row):
            base_impr_share = row['Impr_Share']
            spend_growth_pct = row['Spend_Growth_Pct']
            lost_is = row['Lost_IS_Budget'] if not pd.isna(row['Lost_IS_Budget']) else 0

            # No growth or negative growth = no change
            if spend_growth_pct <= 0:
                return base_impr_share

            # Impression share growth follows a diminishing returns curve relative to spend growth
            # Based on industry benchmarks:
            if spend_growth_pct < 20:
                # For small spend increases: nearly 1:1 ratio
                is_growth_ratio = 0.9
            elif spend_growth_pct < 50:
                # For medium spend increases: about 0.7x ratio
                is_growth_ratio = 0.7
            elif spend_growth_pct < 100:
                # For large spend increases: about 0.5x ratio
                is_growth_ratio = 0.5
            else:
                # For massive spend increases: about 0.3x ratio
                is_growth_ratio = 0.3

            # Adjust ratio for high-performing weeks
            if 'Performance_Tier' in row:
                if row['Performance_Tier'] == 'Exceptional':
                    is_growth_ratio *= 1.2  # 20% better efficiency
                elif row['Performance_Tier'] == 'Excellent':
                    is_growth_ratio *= 1.15  # 15% better efficiency
                elif row['Performance_Tier'] == 'Strong':
                    is_growth_ratio *= 1.1  # 10% better efficiency

            # Calculate potential impression share gain based on spend growth
            potential_gain = base_impr_share * (spend_growth_pct / 100) * is_growth_ratio

            # Cap by available impression share from budget
            if potential_gain > lost_is:
                potential_gain = lost_is * 0.9  # Can capture up to 90% of lost impression share

            # Calculate and return new impression share with a hard cap at 95%
            # (it's rarely feasible to achieve 100% impression share)
            new_impr_share = base_impr_share + potential_gain
            return min(new_impr_share, 95)  # Cap at 95%

        # Apply the realistic impression share calculation
        self.forecast['Forecast_Impr_Share_Adjusted'] = self.forecast.apply(calculate_realistic_impression_share,
                                                                            axis=1)

        # Update final forecast metrics with adjusted values
        self.forecast['Forecast_CPC'] = self.forecast['Forecast_CPC_Adjusted']
        self.forecast['Forecast_Clicks'] = self.forecast['Forecast_Clicks_Adjusted']
        self.forecast['Forecast_Impr_Share'] = self.forecast['Forecast_Impr_Share_Adjusted']

        # Recalculate transactions based on updated click volume
        self.forecast['Forecast_Transactions'] = self.forecast['Forecast_Clicks'] * (
                    self.forecast['Forecast_Conv_Rate'] / 100)

        # Recalculate revenue based on updated transactions
        self.forecast['Forecast_Revenue'] = self.forecast['Forecast_Transactions'] * self.forecast['Forecast_AOV']

        # Recalculate ROAS based on updated revenue
        self.forecast['Forecast_ROAS'] = self.forecast['Forecast_Revenue'] / self.forecast['Forecast_Cost']

        # Final ROAS check - ensure we don't fall below the threshold
        self.forecast['ROAS_Below_Threshold'] = self.forecast['Forecast_ROAS'] < self.min_roas_threshold

        # Adjust costs back down if ROAS falls below threshold
        self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Cost'] = (
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Revenue'] / self.min_roas_threshold
        )

        # Recalculate clicks for adjusted costs
        self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Clicks'] = (
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Cost'] /
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_CPC']
        )

        # Recalculate transactions for adjusted clicks
        self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Transactions'] = (
                self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Clicks'] *
                (self.forecast.loc[self.forecast['ROAS_Below_Threshold'], 'Forecast_Conv_Rate'] / 100)
        )

        return self

    def apply_flexible_roas_target(self):
        """
        Apply a flexible ROAS target that allows some variance but ensures
        no week falls too far below the target.
        """
        # Define minimum acceptable ROAS (90% of target)
        min_acceptable_roas = self.min_roas_threshold * 0.9

        # Find weeks that fall below the minimum acceptable ROAS
        self.forecast['ROAS_Too_Low'] = self.forecast['Forecast_ROAS'] < min_acceptable_roas

        # Adjust spend for weeks below minimum ROAS
        if self.forecast['ROAS_Too_Low'].any():
            # For weeks below threshold, reduce spend to achieve minimum acceptable ROAS
            self.forecast.loc[self.forecast['ROAS_Too_Low'], 'Adjusted_Cost'] = (
                    self.forecast.loc[self.forecast['ROAS_Too_Low'], 'Forecast_Revenue'] / min_acceptable_roas
            )

            # For weeks between min acceptable and target, apply a graduated reduction
            between_target_mask = (
                    (self.forecast['Forecast_ROAS'] >= min_acceptable_roas) &
                    (self.forecast['Forecast_ROAS'] < self.min_roas_threshold)
            )

            if between_target_mask.any():
                # Calculate how far each week is from target (as a percentage)
                distance_from_target = (
                        (self.min_roas_threshold - self.forecast.loc[between_target_mask, 'Forecast_ROAS']) /
                        (self.min_roas_threshold - min_acceptable_roas)
                )

                # Apply a graduated spend reduction (up to 10%)
                reduction_factor = 1 - (distance_from_target * 0.1)

                # Apply the reduction
                self.forecast.loc[between_target_mask, 'Adjusted_Cost'] = (
                        self.forecast.loc[between_target_mask, 'Forecast_Cost'] * reduction_factor
                )

            # Apply spend adjustments and recalculate all metrics
            # For weeks that needed adjustment
            affected_weeks = self.forecast['ROAS_Too_Low'] | between_target_mask

            if affected_weeks.any():
                # Use the adjusted cost where applicable, otherwise use the original forecast cost
                self.forecast['Original_Forecast_Cost'] = self.forecast['Forecast_Cost'].copy()
                self.forecast.loc[affected_weeks, 'Forecast_Cost'] = self.forecast.loc[affected_weeks, 'Adjusted_Cost']

                # Recalculate clicks, transactions, etc.
                self.forecast['Forecast_Clicks'] = self.forecast['Forecast_Cost'] / self.forecast['Forecast_CPC']
                self.forecast['Forecast_Transactions'] = self.forecast['Forecast_Clicks'] * (
                            self.forecast['Forecast_Conv_Rate'] / 100)
                self.forecast['Forecast_Revenue'] = self.forecast['Forecast_Transactions'] * self.forecast[
                    'Forecast_AOV']
                self.forecast['Forecast_ROAS'] = self.forecast['Forecast_Revenue'] / self.forecast['Forecast_Cost']

        return self

    def prepare_final_forecast(self):
        """Prepare the final forecast output"""

        # Create final forecast output with expanded metrics
        self.final_forecast = self.forecast[['Week', 'Forecast_Cost', 'Forecast_Revenue', 'Forecast_ROAS',
                                             'Forecast_Clicks', 'Forecast_Transactions', 'Forecast_CPC',
                                             'Forecast_Conv_Rate', 'Forecast_Impr_Share', 'Search_Volume_Index']].copy()

        # Rename columns for clarity
        self.final_forecast.columns = ['Week', 'Weekly_Budget', 'Projected_Revenue', 'Projected_ROAS',
                                       'Projected_Clicks', 'Projected_Transactions', 'Projected_CPC',
                                       'Projected_Conv_Rate', 'Projected_Impr_Share', 'Search_Volume_Index']

        # Round numeric values for cleaner output
        self.final_forecast['Weekly_Budget'] = self.final_forecast['Weekly_Budget'].round(2)
        self.final_forecast['Projected_Revenue'] = self.final_forecast['Projected_Revenue'].round(2)
        self.final_forecast['Projected_ROAS'] = self.final_forecast['Projected_ROAS'].round(2)
        self.final_forecast['Projected_Clicks'] = self.final_forecast['Projected_Clicks'].round(0)
        self.final_forecast['Projected_Transactions'] = self.final_forecast['Projected_Transactions'].round(0)
        self.final_forecast['Projected_CPC'] = self.final_forecast['Projected_CPC'].round(2)
        self.final_forecast['Projected_Conv_Rate'] = self.final_forecast['Projected_Conv_Rate'].round(2)
        self.final_forecast['Projected_Impr_Share'] = self.final_forecast['Projected_Impr_Share'].round(2)

        # Calculate expected impressions based on CTR
        self.forecast['Avg_CTR'] = self.forecast['CTR'] / 100  # Convert percentage to decimal
        self.final_forecast['Projected_Impressions'] = (self.final_forecast['Projected_Clicks'] /
                                                        self.forecast['Avg_CTR']).round(0)

        # Reorder columns
        self.final_forecast = self.final_forecast[['Week', 'Weekly_Budget', 'Projected_Impressions',
                                                   'Projected_Clicks', 'Projected_CPC', 'Projected_Conv_Rate',
                                                   'Projected_Transactions', 'Projected_Revenue',
                                                   'Projected_ROAS', 'Projected_Impr_Share',
                                                   'Search_Volume_Index']]

        return self

    def run_forecast(self, start_date=None, periods=52):
        """
        Run the complete forecasting process

        Parameters:
        -----------
        start_date : datetime, optional
            Start date for the forecast, defaults to first Sunday of next July
        periods : int, optional
            Number of periods (weeks) to forecast, defaults to 52 (full year)

        Returns:
        --------
        pandas.DataFrame
            The final forecast with all projected metrics
        """
        self.preprocess_data()
        self.apply_cpc_scaling()
        self.apply_conversion_scaling()
        self.calculate_adjusted_metrics()
        self.apply_efficiency_threshold()
        self.generate_forecast_dates(start_date, periods)
        self.apply_seasonal_patterns()
        self.apply_growth_factor()
        self.apply_auction_dynamics()  # Apply realistic auction dynamics
        self.apply_flexible_roas_target()  # Apply flexible ROAS target with variance
        self.prepare_final_forecast()

        return self.final_forecast

    def get_summary_stats(self):
        """Calculate and return summary statistics for the forecast"""
        if self.final_forecast is None:
            return None

        total_annual_budget = self.final_forecast['Weekly_Budget'].sum()
        total_annual_revenue = self.final_forecast['Projected_Revenue'].sum()
        overall_annual_roas = total_annual_revenue / total_annual_budget
        total_annual_transactions = self.final_forecast['Projected_Transactions'].sum()

        # Add additional summary metrics
        avg_cpc = self.final_forecast['Projected_CPC'].mean()
        avg_conv_rate = self.final_forecast['Projected_Conv_Rate'].mean()
        avg_impr_share = self.final_forecast['Projected_Impr_Share'].mean()

        summary = {
            'Total_Annual_Budget': total_annual_budget,
            'Total_Annual_Revenue': total_annual_revenue,
            'Overall_Annual_ROAS': overall_annual_roas,
            'Total_Annual_Transactions': total_annual_transactions,
            'Average_CPC': avg_cpc,
            'Average_Conv_Rate': avg_conv_rate,
            'Average_Impr_Share': avg_impr_share
        }

        return summary

    def save_forecast(self, filename='Category_Forecast.csv', print_csv=False):
        """
        Save the forecast to a CSV file and optionally print it to console

        Parameters:
        -----------
        filename : str
            Filename to save the forecast to
        print_csv : bool
            Whether to print the CSV content to console
        """
        if self.final_forecast is None:
            print("No forecast to save. Please run the forecast first.")
            return None

        # Save to CSV file
        self.final_forecast.to_csv(filename, index=False)
        print(f"Forecast saved to {filename}")

        # Optionally print to console
        if print_csv:
            print("\nCSV Content:")
            print(self.final_forecast.to_csv(index=False))

        return filename

    def get_forecast_details(self):
        """Return detailed insights about the forecast"""
        if self.final_forecast is None:
            return "No forecast available. Please run the forecast first."

        # Get Black Friday week details
        bf_week = self.final_forecast[
            (self.final_forecast['Week'].dt.month == 11) &
            (self.final_forecast['Week'].dt.day >= 20) &
            (self.final_forecast['Week'].dt.day <= 26)
            ]

        if not bf_week.empty:
            # Try to find corresponding historical week for comparison
            if self.forecast is not None:
                bf_forecast = self.forecast[
                    (self.forecast['Week'].dt.month == 11) &
                    (self.forecast['Week'].dt.day >= 20) &
                    (self.forecast['Week'].dt.day <= 26)
                    ]

                if not bf_forecast.empty:
                    historical_cost = bf_forecast['Cost'].iloc[0] if 'Cost' in bf_forecast.columns else "N/A"
                    historical_cpc = bf_forecast['Avg_CPC'].iloc[0] if 'Avg_CPC' in bf_forecast.columns else "N/A"
                    historical_is = bf_forecast['Impr_Share'].iloc[0] if 'Impr_Share' in bf_forecast.columns else "N/A"

                    # Calculate growth percentages
                    if isinstance(historical_cost, (int, float)) and historical_cost > 0:
                        cost_growth = (bf_week['Weekly_Budget'].iloc[0] / historical_cost - 1) * 100
                        cost_growth_str = f"{cost_growth:.1f}%"
                    else:
                        cost_growth_str = "N/A"

                    if isinstance(historical_cpc, (int, float)) and historical_cpc > 0:
                        cpc_growth = (bf_week['Projected_CPC'].iloc[0] / historical_cpc - 1) * 100
                        cpc_growth_str = f"{cpc_growth:.1f}%"
                    else:
                        cpc_growth_str = "N/A"

                    if isinstance(historical_is, (int, float)) and historical_is > 0:
                        is_growth = bf_week['Projected_Impr_Share'].iloc[0] - historical_is
                        is_growth_str = f"{is_growth:.1f} points"
                    else:
                        is_growth_str = "N/A"

                    comparison_str = f"""
            Year-over-Year Comparison:
            Budget: ${historical_cost:,.2f} → ${bf_week['Weekly_Budget'].iloc[0]:,.2f} ({cost_growth_str})
            CPC: ${historical_cpc:.2f} → ${bf_week['Projected_CPC'].iloc[0]:.2f} ({cpc_growth_str})
            Impression Share: {historical_is:.2f}% → {bf_week['Projected_Impr_Share'].iloc[0]:.2f}% ({is_growth_str})
                    """
                else:
                    comparison_str = "No historical Black Friday data available for comparison."
            else:
                comparison_str = "No forecast data available for comparison."

            bf_details = f"""
            Black Friday Week Analysis:
            --------------------------
            Budget: ${bf_week['Weekly_Budget'].iloc[0]:,.2f}
            ROAS: ${bf_week['Projected_ROAS'].iloc[0]:.2f}
            CPC: ${bf_week['Projected_CPC'].iloc[0]:.2f}
            Conversion Rate: {bf_week['Projected_Conv_Rate'].iloc[0]:.2f}%
            Impression Share: {bf_week['Projected_Impr_Share'].iloc[0]:.2f}%

            {comparison_str}

            Key factors that led to this recommendation:
            1. Ultra-high CPC during Black Friday (nearly twice annual average)
            2. ROAS historically close to the minimum threshold
            3. Diminishing returns become severe above $100,000 spend
            4. Progressive CPC scaling based on spend growth percentage
            5. Logarithmic impression share growth modeling
            6. Integrated auction dynamics to balance metrics
            7. Additional 2% YOY CPC inflation factor applied
            8. 5% YOY AOV growth factor applied
            9. Protection logic for historically high-performing weeks

            The model recommended an aggressive but sustainable increase to maintain profitability
            while capturing additional high-value impressions.
            """
            return bf_details

        return "No specific period details available."


# =============================================
# NEW FUNCTIONS FOR LOADING CSV DATA
# =============================================

def load_data_from_csv(file_path):
    """
    Load data from a CSV file and format it for the SEM forecast model.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pandas.DataFrame
        Formatted data ready for the model
    """
    print(f"Loading data from: {file_path}")

    # Load the data
    data = pd.read_csv(file_path)

    # Convert string-formatted numbers to floats
    # Currency values (remove $ and commas)
    for col in ['Cost', 'Revenue', 'Avg_CPC']:
        if col in data.columns and data[col].dtype == 'object':
            data[col] = data[col].str.replace('$', '').str.replace(',', '').astype(float)

    # Percentage values (remove % sign)
    percent_columns = ['CTR', 'Conv_Rate', 'Impr_Share', 'Lost_IS_Budget', 'Lost_IS_Rank']
    for col in percent_columns:
        if col in data.columns and data[col].dtype == 'object':
            data[col] = data[col].str.rstrip('%').astype(float)

    # Handle ROAS (might be formatted as "25.4x" or just "25.4")
    if 'ROAS' in data.columns and data['ROAS'].dtype == 'object':
        # First remove any dollar signs
        data['ROAS'] = data['ROAS'].str.replace('$', '')
        # Then remove any 'x' characters
        data['ROAS'] = data['ROAS'].str.replace('x', '')
        # Finally convert to float
        data['ROAS'] = data['ROAS'].astype(float)

    # Convert Week to datetime
    if 'Week' in data.columns:
        data['Week'] = pd.to_datetime(data['Week'])

    # Sort by date
    data = data.sort_values('Week')

    print(f"Successfully loaded {len(data)} rows of data")
    print(f"Date range: {data['Week'].min()} to {data['Week'].max()}")

    return data


# Function definition for run_category_forecast
def run_category_forecast(file_path, min_roas=20, growth_factor=1.10, cpc_inflation=1.02, aov_growth=1.05,
                          print_comparison=True):
    """
    Run a forecast for a single category.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file for the category
    min_roas : float, optional
        Minimum ROAS threshold
    growth_factor : float, optional
        Year-over-year growth factor
    cpc_inflation : float, optional
        Year-over-year CPC inflation factor
    aov_growth : float, optional
        Year-over-year AOV growth factor
    print_comparison : bool, optional
        Whether to print YOY comparison statistics

    Returns:
    --------
    pandas.DataFrame
        Forecast results
    """
    # Get category name from the file name
    category_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing category: {category_name}")

    try:
        # Load the data
        data = load_data_from_csv(file_path)

        # Create the model with the data
        model = SEMForecastModel(data, min_roas_threshold=min_roas, yoy_growth=growth_factor,
                                 yoy_cpc_inflation=cpc_inflation, yoy_aov_growth=aov_growth)

        # Run the forecast
        forecast = model.run_forecast()

        # Get output file path - in the same folder as the input
        output_folder = os.path.dirname(file_path)
        output_file = os.path.join(output_folder, f"{category_name}_Forecast.csv")

        # Save the forecast
        model.save_forecast(output_file)

        # Print summary
        summary = model.get_summary_stats()
        print(f"\nSummary for {category_name}:")
        print(f"Total Annual Budget: ${summary['Total_Annual_Budget']:,.2f}")
        print(f"Total Annual Revenue: ${summary['Total_Annual_Revenue']:,.2f}")
        print(f"Overall Annual ROAS: ${summary['Overall_Annual_ROAS']:.2f}")
        print(f"Total Annual Transactions: {summary['Total_Annual_Transactions']:,.0f}")
        print(f"Average CPC: ${summary['Average_CPC']:.2f}")
        print(f"Average Conv. Rate: {summary['Average_Conv_Rate']:.2f}%")
        print(f"Average Impr. Share: {summary['Average_Impr_Share']:.2f}%")

        # Print comparison to historical performance
        if print_comparison:
            historical_cost = data['Cost'].sum()
            forecast_cost = summary['Total_Annual_Budget']
            cost_change_pct = ((forecast_cost / historical_cost) - 1) * 100

            historical_revenue = data['Revenue'].sum()
            forecast_revenue = summary['Total_Annual_Revenue']
            revenue_change_pct = ((forecast_revenue / historical_revenue) - 1) * 100

            historical_roas = historical_revenue / historical_cost
            roas_change_pct = ((summary['Overall_Annual_ROAS'] / historical_roas) - 1) * 100

            print("\nYOY Comparison:")
            print(f"Cost Change: {cost_change_pct:.1f}% (${historical_cost:,.2f} → ${forecast_cost:,.2f})")
            print(f"Revenue Change: {revenue_change_pct:.1f}% (${historical_revenue:,.2f} → ${forecast_revenue:,.2f})")
            print(
                f"ROAS Change: {roas_change_pct:.1f}% (${historical_roas:.2f} → ${summary['Overall_Annual_ROAS']:.2f})")

            # Count high-performing weeks and their forecast growth
            high_performing_weeks = data[data['ROAS'] > (min_roas * 1.5)]
            if not high_performing_weeks.empty:
                # Create a copy of the high-performing weeks with week number
                hp_data = high_performing_weeks.copy()

                # We need to convert the date to week number (isocalendar week)
                if 'Week' in hp_data.columns:
                    # Add Week_Number column to match with forecast
                    hp_data['Week_Number'] = hp_data['Week'].dt.isocalendar().week

                    # Get unique week numbers for high-performing weeks
                    hp_week_numbers = hp_data['Week_Number'].unique()

                    # Get forecast data for these week numbers
                    forecast_high_performing = model.forecast[model.forecast['Week_Number'].isin(hp_week_numbers)]

                    if not forecast_high_performing.empty and not hp_data.empty:
                        # Calculate total cost for high-performing weeks in historical and forecast
                        hp_historical_cost = hp_data['Cost'].sum()
                        hp_forecast_cost = forecast_high_performing['Forecast_Cost'].sum()

                        if hp_historical_cost > 0:  # Avoid division by zero
                            avg_hp_growth = ((hp_forecast_cost / hp_historical_cost) - 1) * 100

                            print(f"High-Performing Weeks: {len(hp_week_numbers)}")
                            print(f"Average Growth for High-Performing Weeks: {avg_hp_growth:.1f}%")

        return forecast

    except Exception as e:
        print(f"Error processing {category_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================
# MAIN FUNCTION - RUN THIS SCRIPT
# =============================================

def run_all_categories(folder_path, min_roas=20, growth_factor=1.10, cpc_inflation=1.02, aov_growth=1.05):
    """
    Run forecasts for all CSV files in a folder.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CSV files
    min_roas : float, optional
        Minimum ROAS threshold
    growth_factor : float, optional
        Year-over-year growth factor
    cpc_inflation : float, optional
        Year-over-year CPC inflation factor
    aov_growth : float, optional
        Year-over-year AOV growth factor

    Returns:
    --------
    dict
        Dictionary with category names as keys and forecast DataFrames as values
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return {}

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and not f.endswith('_Forecast.csv')]

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return {}

    # Run forecast for each file
    results = {}
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        category_name = os.path.splitext(csv_file)[0]

        forecast = run_category_forecast(file_path, min_roas, growth_factor, cpc_inflation, aov_growth)
        results[category_name] = forecast

    print(f"\nProcessed {len(results)} categories")
    return results


# =============================================
# CHANGE THIS SECTION TO RUN YOUR SPECIFIC CATEGORY
# =============================================

if __name__ == "__main__":
    # Use absolute path to your file
    category_file = "L2Shopping_Clothing.csv"  # Just the filename with no path
    forecast = run_category_forecast(category_file)

    print("\nForecast completed!")
