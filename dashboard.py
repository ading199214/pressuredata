#!/usr/bin/env python3
"""
Air Pressure Machine Dashboard - Redesigned Segmentation

New segmentation logic:
- Shutdown: -0.1 to 0.1 kPa
- Transition: 0.1 to 20 kPa  
- Stable: > 20 kPa
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="MBR sensor data and PLC data",
    layout="wide"
)

@st.cache_data
def load_xlsx_data():
    """Load and preprocess the XLSX and CSV comparison data."""
    try:
        # Check if files exist
        import os
        if not os.path.exists('Tank1.xlsx'):
            return None, None, None, "Tank1.xlsx file not found"
        if not os.path.exists('Tank1-02.xlsx'):
            return None, None, None, "Tank1-02.xlsx file not found"
        if not os.path.exists('tank1-03.csv'):
            return None, None, None, "tank1-03.csv file not found"
            
        # Load XLSX files
        tank1_df = pd.read_excel('Tank1.xlsx')
        tank2_df = pd.read_excel('Tank1-02.xlsx')
        
        # Load CSV file
        tank3_df = pd.read_csv('tank1-03.csv')
        
        # Extract relevant columns
        tank1_data = tank1_df[['曝气压力', '曝气流量']].copy()
        tank2_data = tank2_df[['曝气压力', '曝气流量']].copy()
        tank3_data = tank3_df[['AerationPressure1', 'AerationFlow1']].copy()
        
        # Rename columns for consistency
        tank3_data = tank3_data.rename(columns={
            'AerationPressure1': '曝气压力',
            'AerationFlow1': '曝气流量'
        })
        
        # Add file identifiers
        tank1_data['Tank'] = 'Tank1 (8.01~8.17)'
        tank2_data['Tank'] = 'Tank1-02 (8.26~9.06)'
        tank3_data['Tank'] = 'Tank1-03 (9.23~)'
        
        # Add time index (assuming regular sampling)
        tank1_data['Time_Index'] = range(len(tank1_data))
        tank2_data['Time_Index'] = range(len(tank2_data))
        tank3_data['Time_Index'] = range(len(tank3_data))
        
        # Clean data - remove any null values
        tank1_data = tank1_data.dropna()
        tank2_data = tank2_data.dropna()
        tank3_data = tank3_data.dropna()
        
        return tank1_data, tank2_data, tank3_data, None
        
    except Exception as e:
        return None, None, None, f"Error loading comparison data: {str(e)}"

def fill_missing_data_periods(df):
    """Fill missing data periods with shutdown state (0 pressure) while preserving all original data points."""
    try:
        if len(df) == 0:
            return df
        
        # Sort by datetime
        df = df.sort_values('datetime_local').reset_index(drop=True)
        
        # Mark all original data as not gap-filled
        df['is_gap_filled'] = False
        
        # Identify significant gaps (more than 2 hours between consecutive data points)
        time_diffs = df['datetime_local'].diff()
        gap_threshold = pd.Timedelta(hours=2)
        significant_gaps = time_diffs > gap_threshold
        
        # If no significant gaps, return original data with gap_filled column
        if not significant_gaps.any():
            return df
        
        # Find gap periods
        gap_indices = df[significant_gaps].index
        gap_fill_data = []
        
        for gap_idx in gap_indices:
            # Get the time before and after the gap
            time_before = df.loc[gap_idx - 1, 'datetime_local']
            time_after = df.loc[gap_idx, 'datetime_local']
            
            # Create hourly fill-in points for the gap period
            gap_timeline = pd.date_range(
                start=time_before + pd.Timedelta(hours=1),
                end=time_after - pd.Timedelta(hours=1),
                freq='H'
            )
            
            # Create shutdown data for the gap period
            for gap_time in gap_timeline:
                gap_fill_data.append({
                    'timestamp': int(gap_time.timestamp() * 1000),
                    'datetime_local': gap_time,
                    'pressure_kpa': 0.0,  # Shutdown state
                    'date': gap_time.date(),
                    'hour': gap_time.hour,
                    'day_of_week': gap_time.day_name(),
                    'is_gap_filled': True  # Mark as gap-filled data
                })
        
        # If we have gap fill data, combine it with original data
        if gap_fill_data:
            gap_df = pd.DataFrame(gap_fill_data)
            # Combine original data with gap fill data
            combined_df = pd.concat([df, gap_df], ignore_index=True)
            # Sort by datetime to maintain chronological order
            combined_df = combined_df.sort_values('datetime_local').reset_index(drop=True)
            return combined_df
        else:
            # No gaps to fill, return original data (already has is_gap_filled column)
            return df
        
    except Exception as e:
        # If there's any error in gap filling, return original data with is_gap_filled column
        print(f"Warning: Gap filling failed: {e}")
        if 'is_gap_filled' not in df.columns:
            df['is_gap_filled'] = False
        return df

@st.cache_data
def load_pressure_data():
    """Load and preprocess the pressure data with new segmentation."""
    try:
        # Check if files exist
        if not Path('alldata.csv').exists():
            return None, "File 'alldata.csv' not found in current directory!"
        
        # Load main data
        df1 = pd.read_csv('alldata.csv')
        
        # Check columns for first file
        expected_cols = ['Timestamp', 'DateTime (Local)', 'Pressure (kPa)']
        if list(df1.columns) != expected_cols:
            return None, f"Unexpected columns in alldata.csv. Expected: {expected_cols}, Got: {list(df1.columns)}"
        
        # Load second data file if it exists
        df_list = [df1]
        if Path('alldata-b.csv').exists():
            df2 = pd.read_csv('alldata-b.csv')
            
            # Check columns for second file
            if list(df2.columns) != expected_cols:
                return None, f"Unexpected columns in alldata-b.csv. Expected: {expected_cols}, Got: {list(df2.columns)}"
            
            df_list.append(df2)
        
        # Combine all dataframes
        df = pd.concat(df_list, ignore_index=True)
        
        df.columns = ['timestamp', 'datetime_local', 'pressure_kpa']
        
        # Convert datetime
        try:
            df['datetime_local'] = pd.to_datetime(df['datetime_local'], format='%Y-%m-%d %H:%M:%S.%f')
        except:
            df['datetime_local'] = pd.to_datetime(df['datetime_local'])
        
        # Sort by datetime to ensure chronological order
        df = df.sort_values('datetime_local').reset_index(drop=True)
        
        df['date'] = df['datetime_local'].dt.date
        df['hour'] = df['datetime_local'].dt.hour
        df['day_of_week'] = df['datetime_local'].dt.day_name()
        
        # Convert pressure to numeric
        df['pressure_kpa'] = pd.to_numeric(df['pressure_kpa'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['pressure_kpa', 'datetime_local'])
        
        if len(df) == 0:
            return None, "No valid data found after cleaning!"
        
        # Add source file information for tracking
        files_loaded = ['alldata.csv']
        if len(df_list) > 1:
            files_loaded.append('alldata-b.csv')
        df.attrs['files_loaded'] = files_loaded
        
        # Fill missing data periods with shutdown state
        df = fill_missing_data_periods(df)
        
        # NEW SEGMENTATION LOGIC
        conditions = [
            (df['pressure_kpa'] >= -0.1) & (df['pressure_kpa'] <= 0.1),  # Shutdown: -0.1 to 0.1 kPa
            ((df['pressure_kpa'] < -0.1) | ((df['pressure_kpa'] > 0.1) & (df['pressure_kpa'] <= 20.0))),  # Transition: <-0.1 OR 0.1-20 kPa
            (df['pressure_kpa'] > 20.0)                                  # Stable: > 20 kPa
        ]
        choices = ['Shutdown', 'Transition', 'Stable']
        df['operational_state'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate rolling statistics
        df = df.sort_values('datetime_local')
        df['pressure_rolling_mean'] = df['pressure_kpa'].rolling(window=50, center=True).mean()
        df['pressure_rolling_std'] = df['pressure_kpa'].rolling(window=50, center=True).std()
        
        return df, None
        
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

class PressureDashboard:
    def __init__(self):
        self.df = None
    
    def create_overview_metrics(self):
        """Create overview metrics cards."""
        if self.df is None:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_points = len(self.df)
            st.metric("Total Data Points", f"{total_points:,}")
        
        with col2:
            date_range = f"{len(self.df['date'].unique())} days"
            st.metric("Date Range", date_range)
        
        with col3:
            max_pressure = self.df['pressure_kpa'].max()
            st.metric("Max Pressure", f"{max_pressure:.2f} kPa")
        
        with col4:
            min_pressure = self.df['pressure_kpa'].min()
            st.metric("Min Pressure", f"{min_pressure:.2f} kPa")
    
    def create_main_timeline(self):
        """Create the main pressure timeline visualization."""
        st.subheader("Pressure Timeline")
        
        # Sample data for performance if too large
        df_plot = self.df
        if len(self.df) > 50000:
            df_plot = self.df.sample(n=50000).sort_values('datetime_local')
            st.info(f"Showing sample of {len(df_plot):,} points for performance (out of {len(self.df):,} total)")
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Pressure Over Time', 'Operational States'),
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Main pressure line
        fig.add_trace(
            go.Scatter(
                x=df_plot['datetime_local'],
                y=df_plot['pressure_kpa'],
                mode='lines',
                name='Pressure',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x}</b><br>Pressure: %{y:.2f} kPa<br>State: %{customdata}<extra></extra>',
                customdata=df_plot['operational_state']
            ),
            row=1, col=1
        )
        

        
        # Operational states as one continuous line showing state transitions
        state_colors = {
            'Shutdown': 'red',
            'Transition': 'orange', 
            'Stable': 'green'
        }
        
        # Sort by datetime to ensure proper timeline
        df_plot = df_plot.sort_values('datetime_local').reset_index(drop=True)
        
        # Create one continuous line that changes color based on operational state
        # We'll plot each segment with its appropriate color
        if len(df_plot) > 0:
            current_state = df_plot.iloc[0]['operational_state']
            segment_start = 0
            
            for i in range(1, len(df_plot)):
                if df_plot.iloc[i]['operational_state'] != current_state or i == len(df_plot) - 1:
                    # End of current state segment
                    end_idx = i if i == len(df_plot) - 1 else i
                    segment_data = df_plot.iloc[segment_start:end_idx + 1]
                    
                    # Add the segment as a continuous line (with thicker line)
                    fig.add_trace(
                        go.Scatter(
                            x=segment_data['datetime_local'],
                            y=[current_state] * len(segment_data),
                            mode='lines+markers',
                            name=current_state,
                            line=dict(color=state_colors[current_state], width=12),  # 3x thicker (was 4, now 12)
                            marker=dict(color=state_colors[current_state], size=3),
                            showlegend=current_state not in [trace.name for trace in fig.data if hasattr(trace, 'name')],
                            hovertemplate=f'<b>%{{x}}</b><br>{current_state}<br>Pressure: %{{customdata:.2f}} kPa<extra></extra>',
                            customdata=segment_data['pressure_kpa'],
                            connectgaps=False
                        ),
                        row=2, col=1
                    )
                    
                    # Start new segment
                    current_state = df_plot.iloc[i]['operational_state']
                    segment_start = i
        

        
        fig.update_layout(
            height=800,
            title="MBR sensor data chart",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date/Time", row=2, col=1)
        fig.update_yaxes(title_text="Pressure (kPa)", row=1, col=1)
        fig.update_yaxes(
            title_text="Operational State", 
            categoryorder="array", 
            categoryarray=["Shutdown", "Transition", "Stable"],  # Bottom to top order
            row=2, col=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    

    
    def create_daily_analysis(self):
        """Create daily analysis section."""
        
        
        # Daily statistics - calculate avg_pressure using only stable data
        daily_stats_list = []
        
        for date in self.df['date'].unique():
            daily_data = self.df[self.df['date'] == date]
            actual_data = daily_data[daily_data['is_gap_filled'] == False]  # Only actual sensor data
            stable_data = daily_data[daily_data['operational_state'] == 'Stable']
            
            # Calculate uptime percentage based on actual data only
            if len(actual_data) > 0:
                uptime_percent = (actual_data['operational_state'].isin(['Stable', 'Transition']).sum() / len(actual_data)) * 100
            else:
                uptime_percent = 0.0
            
            daily_stat = {
                'date': date,
                'avg_pressure': stable_data['pressure_kpa'].mean() if len(stable_data) > 0 else None,
                'std_pressure': stable_data['pressure_kpa'].std() if len(stable_data) > 0 else None,
                'min_pressure': actual_data['pressure_kpa'].min() if len(actual_data) > 0 else None,
                'max_pressure': actual_data['pressure_kpa'].max() if len(actual_data) > 0 else None,
                'total_points': len(actual_data),  # Only count actual data points
                'uptime_percent': uptime_percent,
                'stable_points': len(stable_data)
            }
            daily_stats_list.append(daily_stat)
        
        daily_stats = pd.DataFrame(daily_stats_list)
        
        # Add pressure level classification for visualization (based on stable avg only)
        daily_stats['pressure_level'] = daily_stats['avg_pressure'].apply(
            lambda x: 'High' if x > 25 else ('Medium' if x > 15 else ('Low' if x > 0 else 'No Stable Data'))
        )
        

        
        # Daily statistics table
        st.subheader("Pressure Sensor Daily Statistics Table")
        
        # Remove pressure_level column from display and reorder columns
        display_stats = daily_stats[['date', 'avg_pressure', 'std_pressure', 'min_pressure', 'max_pressure', 
                                   'total_points', 'stable_points', 'uptime_percent']].copy()
        
        # Format the dataframe with N/A for None values
        def format_value(val, format_str, na_value="N/A"):
            if val is None or np.isnan(val) if isinstance(val, (int, float)) else False:
                return na_value
            try:
                return format_str.format(val)
            except:
                return na_value
        
        # Apply custom formatting
        display_stats_formatted = display_stats.copy()
        display_stats_formatted['avg_pressure'] = display_stats_formatted['avg_pressure'].apply(
            lambda x: format_value(x, '{:.2f}')
        )
        display_stats_formatted['std_pressure'] = display_stats_formatted['std_pressure'].apply(
            lambda x: format_value(x, '{:.3f}')
        )
        display_stats_formatted['min_pressure'] = display_stats_formatted['min_pressure'].apply(
            lambda x: format_value(x, '{:.2f}')
        )
        display_stats_formatted['max_pressure'] = display_stats_formatted['max_pressure'].apply(
            lambda x: format_value(x, '{:.2f}')
        )
        display_stats_formatted['uptime_percent'] = display_stats_formatted['uptime_percent'].apply(
            lambda x: format_value(x, '{:.1f}%')
        )
        
        st.dataframe(display_stats_formatted, use_container_width=True)
        
        st.caption("**Note**: avg_pressure and std_pressure are calculated using only Stable operation data (>20 kPa). "
                  "total_points shows only actual sensor data (excludes gap-filled shutdown periods).")
    
    def create_xlsx_comparison(self):
        """Create comparison charts for all three tank datasets."""
        st.subheader("PLC Data Comparison")
        
        # Load all tank data
        with st.spinner("Loading tank comparison data..."):
            try:
                tank1_data, tank2_data, tank3_data, error = load_xlsx_data()
            except Exception as e:
                # Fallback: try loading directly without cache
                st.warning("Cache loading failed, trying direct load...")
                try:
                    import os
                    if not os.path.exists('Tank1.xlsx'):
                        error = "Tank1.xlsx file not found"
                        tank1_data, tank2_data, tank3_data = None, None, None
                    elif not os.path.exists('Tank1-02.xlsx'):
                        error = "Tank1-02.xlsx file not found"
                        tank1_data, tank2_data, tank3_data = None, None, None
                    elif not os.path.exists('tank1-03.csv'):
                        error = "tank1-03.csv file not found"
                        tank1_data, tank2_data, tank3_data = None, None, None
                    else:
                        # Direct load without cache
                        tank1_df = pd.read_excel('Tank1.xlsx')
                        tank2_df = pd.read_excel('Tank1-02.xlsx')
                        tank3_df = pd.read_csv('tank1-03.csv')
                        
                        tank1_data = tank1_df[['曝气压力', '曝气流量']].copy()
                        tank2_data = tank2_df[['曝气压力', '曝气流量']].copy()
                        tank3_data = tank3_df[['AerationPressure1', 'AerationFlow1']].copy()
                        
                        # Rename columns for consistency
                        tank3_data = tank3_data.rename(columns={
                            'AerationPressure1': '曝气压力',
                            'AerationFlow1': '曝气流量'
                        })
                        
                        tank1_data['Tank'] = 'Tank1 (8.01~8.17)'
                        tank2_data['Tank'] = 'Tank1-02 (8.26~9.06)'
                        tank3_data['Tank'] = 'Tank1-03 (9.23~)'
                        
                        tank1_data['Time_Index'] = range(len(tank1_data))
                        tank2_data['Time_Index'] = range(len(tank2_data))
                        tank3_data['Time_Index'] = range(len(tank3_data))
                        
                        tank1_data = tank1_data.dropna()
                        tank2_data = tank2_data.dropna()
                        tank3_data = tank3_data.dropna()
                        
                        error = None
                except Exception as e2:
                    error = f"Direct load also failed: {str(e2)}"
                    tank1_data, tank2_data, tank3_data = None, None, None
        
        if tank1_data is None or tank2_data is None or tank3_data is None:
            st.error(f"Failed to load tank data: {error}")
            return
        
        # Show data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Tank1 (8.01~8.17)**\n{len(tank1_data):,} data points")
        with col2:
            st.info(f"**Tank1-02 (8.26~9.06)**\n{len(tank2_data):,} data points")
        with col3:
            st.info(f"**Tank1-03 (9.23~)**\n{len(tank3_data):,} data points")
        
        # Create side-by-side comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("曝气压力 (Aeration Pressure) Comparison")
            
            # Sample data for performance if too large
            tank1_sample = tank1_data.sample(min(10000, len(tank1_data))) if len(tank1_data) > 10000 else tank1_data
            tank2_sample = tank2_data.sample(min(10000, len(tank2_data))) if len(tank2_data) > 10000 else tank2_data
            tank3_sample = tank3_data.sample(min(10000, len(tank3_data))) if len(tank3_data) > 10000 else tank3_data
            
            # Pressure comparison chart
            fig_pressure = go.Figure()
            
            fig_pressure.add_trace(go.Scatter(
                x=tank1_sample['Time_Index'],
                y=tank1_sample['曝气压力'],
                mode='lines',
                name='Tank1 (8.01~8.17)',
                line=dict(color='blue', width=1),
                opacity=0.7
            ))
            
            fig_pressure.add_trace(go.Scatter(
                x=tank2_sample['Time_Index'],
                y=tank2_sample['曝气压力'],
                mode='lines',
                name='Tank1-02 (8.26~9.06)',
                line=dict(color='red', width=1),
                opacity=0.7
            ))
            
            fig_pressure.add_trace(go.Scatter(
                x=tank3_sample['Time_Index'],
                y=tank3_sample['曝气压力'],
                mode='lines',
                name='Tank1-03 (9.23~)',
                line=dict(color='green', width=1),
                opacity=0.7
            ))
            
            fig_pressure.update_layout(
                title="曝气压力 Over Time",
                xaxis_title="Time Index",
                yaxis_title="曝气压力 (Aeration Pressure)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pressure, use_container_width=True)
            
            # Pressure statistics
            st.write("**Aeration Pressure Statistics:**")
            pressure_stats = pd.DataFrame({
                'Tank1 (8.01~8.17)': [
                    tank1_data['曝气压力'].mean(),
                    tank1_data['曝气压力'].std(),
                    tank1_data['曝气压力'].min(),
                    tank1_data['曝气压力'].max()
                ],
                'Tank1-02 (8.26~9.06)': [
                    tank2_data['曝气压力'].mean(),
                    tank2_data['曝气压力'].std(),
                    tank2_data['曝气压力'].min(),
                    tank2_data['曝气压力'].max()
                ],
                'Tank1-03 (9.23~)': [
                    tank3_data['曝气压力'].mean(),
                    tank3_data['曝气压力'].std(),
                    tank3_data['曝气压力'].min(),
                    tank3_data['曝气压力'].max()
                ]
            }, index=['Mean', 'Std Dev', 'Min', 'Max'])
            
            st.dataframe(pressure_stats.round(3), use_container_width=True)
        
        with col2:
            st.subheader("曝气流量 (Aeration Flow) Comparison")
            
            # Flow comparison chart
            fig_flow = go.Figure()
            
            fig_flow.add_trace(go.Scatter(
                x=tank1_sample['Time_Index'],
                y=tank1_sample['曝气流量'],
                mode='lines',
                name='Tank1 (8.01~8.17)',
                line=dict(color='blue', width=1),
                opacity=0.7
            ))
            
            fig_flow.add_trace(go.Scatter(
                x=tank2_sample['Time_Index'],
                y=tank2_sample['曝气流量'],
                mode='lines',
                name='Tank1-02 (8.26~9.06)',
                line=dict(color='red', width=1),
                opacity=0.7
            ))
            
            fig_flow.add_trace(go.Scatter(
                x=tank3_sample['Time_Index'],
                y=tank3_sample['曝气流量'],
                mode='lines',
                name='Tank1-03 (9.23~)',
                line=dict(color='green', width=1),
                opacity=0.7
            ))
            
            fig_flow.update_layout(
                title="曝气流量 Over Time",
                xaxis_title="Time Index",
                yaxis_title="曝气流量 (Aeration Flow)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_flow, use_container_width=True)
            
            # Flow statistics
            st.write("**Aeration Flow Statistics:**")
            flow_stats = pd.DataFrame({
                'Tank1 (8.01~8.17)': [
                    tank1_data['曝气流量'].mean(),
                    tank1_data['曝气流量'].std(),
                    tank1_data['曝气流量'].min(),
                    tank1_data['曝气流量'].max()
                ],
                'Tank1-02 (8.26~9.06)': [
                    tank2_data['曝气流量'].mean(),
                    tank2_data['曝气流量'].std(),
                    tank2_data['曝气流量'].min(),
                    tank2_data['曝气流量'].max()
                ],
                'Tank1-03 (9.23~)': [
                    tank3_data['曝气流量'].mean(),
                    tank3_data['曝气流量'].std(),
                    tank3_data['曝气流量'].min(),
                    tank3_data['曝气流量'].max()
                ]
            }, index=['Mean', 'Std Dev', 'Min', 'Max'])
            
            st.dataframe(flow_stats.round(3), use_container_width=True)
        
        # Combined distribution comparison
        st.subheader("Distribution Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pressure distribution histogram
            fig_pressure_dist = go.Figure()
            
            fig_pressure_dist.add_trace(go.Histogram(
                x=tank1_data['曝气压力'],
                name='Tank1 (8.01~8.17)',
                opacity=0.6,
                nbinsx=30,
                marker_color='blue'
            ))
            
            fig_pressure_dist.add_trace(go.Histogram(
                x=tank2_data['曝气压力'],
                name='Tank1-02 (8.26~9.06)',
                opacity=0.6,
                nbinsx=30,
                marker_color='red'
            ))
            
            fig_pressure_dist.add_trace(go.Histogram(
                x=tank3_data['曝气压力'],
                name='Tank1-03 (9.23~)',
                opacity=0.6,
                nbinsx=30,
                marker_color='green'
            ))
            
            fig_pressure_dist.update_layout(
                title="曝气压力 Distribution",
                xaxis_title="曝气压力 (Aeration Pressure)",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_pressure_dist, use_container_width=True)
        
        with col2:
            # Flow distribution histogram
            fig_flow_dist = go.Figure()
            
            fig_flow_dist.add_trace(go.Histogram(
                x=tank1_data['曝气流量'],
                name='Tank1 (8.01~8.17)',
                opacity=0.6,
                nbinsx=30,
                marker_color='blue'
            ))
            
            fig_flow_dist.add_trace(go.Histogram(
                x=tank2_data['曝气流量'],
                name='Tank1-02 (8.26~9.06)',
                opacity=0.6,
                nbinsx=30,
                marker_color='red'
            ))
            
            fig_flow_dist.add_trace(go.Histogram(
                x=tank3_data['曝气流量'],
                name='Tank1-03 (9.23~)',
                opacity=0.6,
                nbinsx=30,
                marker_color='green'
            ))
            
            fig_flow_dist.update_layout(
                title="曝气流量 Distribution",
                xaxis_title="曝气流量 (Aeration Flow)",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_flow_dist, use_container_width=True)
        
        # Key differences summary
        st.subheader("Key Differences Summary")
        
        # Calculate differences (excluding zero values)
        tank1_pressure_nonzero = tank1_data[tank1_data['曝气压力'] != 0]['曝气压力']
        tank2_pressure_nonzero = tank2_data[tank2_data['曝气压力'] != 0]['曝气压力']
        tank3_pressure_nonzero = tank3_data[tank3_data['曝气压力'] != 0]['曝气压力']
        tank1_flow_nonzero = tank1_data[tank1_data['曝气流量'] != 0]['曝气流量']
        tank2_flow_nonzero = tank2_data[tank2_data['曝气流量'] != 0]['曝气流量']
        tank3_flow_nonzero = tank3_data[tank3_data['曝气流量'] != 0]['曝气流量']
        
        # Calculate mean values
        tank1_pressure_mean = tank1_pressure_nonzero.mean()
        tank2_pressure_mean = tank2_pressure_nonzero.mean()
        tank3_pressure_mean = tank3_pressure_nonzero.mean()
        tank1_flow_mean = tank1_flow_nonzero.mean()
        tank2_flow_mean = tank2_flow_nonzero.mean()
        tank3_flow_mean = tank3_flow_nonzero.mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Tank1 vs Tank1-02:**")
            pressure_diff_12 = tank2_pressure_mean - tank1_pressure_mean
            flow_diff_12 = tank2_flow_mean - tank1_flow_mean
            st.metric("Aeration Pressure Diff", f"{pressure_diff_12:.3f}", delta=f"{pressure_diff_12:.3f}")
            st.metric("Aeration Flow Diff", f"{flow_diff_12:.3f}", delta=f"{flow_diff_12:.3f}")
        
        with col2:
            st.write("**Tank1 vs Tank1-03:**")
            pressure_diff_13 = tank3_pressure_mean - tank1_pressure_mean
            flow_diff_13 = tank3_flow_mean - tank1_flow_mean
            st.metric("Aeration Pressure Diff", f"{pressure_diff_13:.3f}", delta=f"{pressure_diff_13:.3f}")
            st.metric("Aeration Flow Diff", f"{flow_diff_13:.3f}", delta=f"{flow_diff_13:.3f}")
        
        with col3:
            st.write("**Tank1-02 vs Tank1-03:**")
            pressure_diff_23 = tank3_pressure_mean - tank2_pressure_mean
            flow_diff_23 = tank3_flow_mean - tank2_flow_mean
            st.metric("Aeration Pressure Diff", f"{pressure_diff_23:.3f}", delta=f"{pressure_diff_23:.3f}")
            st.metric("Aeration Flow Diff", f"{flow_diff_23:.3f}", delta=f"{flow_diff_23:.3f}")
        
        # Show data points info
        st.caption(f"**Note**: All calculations exclude zero values. "
                  f"Tank1: {len(tank1_pressure_nonzero):,} pressure, {len(tank1_flow_nonzero):,} flow points. "
                  f"Tank1-02: {len(tank2_pressure_nonzero):,} pressure, {len(tank2_flow_nonzero):,} flow points. "
                  f"Tank1-03: {len(tank3_pressure_nonzero):,} pressure, {len(tank3_flow_nonzero):,} flow points.")

    
    def run_dashboard(self):
        """Run the complete dashboard."""
        st.title("MBR sensor and PLC data")

        
        # Load data
        with st.spinner("Loading pressure data..."):
            df, error = load_pressure_data()
        
        if df is None:
            st.error(f"Failed to load data: {error}")
            return
        
        self.df = df
        
        # Display loading success message with file information
        files_loaded = getattr(df, 'attrs', {}).get('files_loaded', ['alldata.csv'])
        files_str = " + ".join(files_loaded) if len(files_loaded) > 1 else files_loaded[0]
        
        
        # Daily analysis (moved to top)

        self.create_daily_analysis()
        st.markdown("Shutdown: -0.1 to 0.1 kPa | Transition: 0.1 to 20 kPa | Stable: > 20 kPa")
        st.markdown("---")        
        # Overview metrics
        self.create_overview_metrics()
        st.markdown("---")
        
        # Main timeline
        self.create_main_timeline()
        st.markdown("---")
        
        # XLSX Tank Comparison (separate tank data analysis)
        self.create_xlsx_comparison()
        
        # Footer
        st.markdown("---")
        st.markdown("***Digital Hub China***")
def main():
    dashboard = PressureDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
