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
    """Load and preprocess the XLSX comparison data."""
    try:
        # Check if files exist
        import os
        if not os.path.exists('Tank1.xlsx'):
            return None, None, "Tank1.xlsx file not found"
        if not os.path.exists('Tank1-02.xlsx'):
            return None, None, "Tank1-02.xlsx file not found"
            
        # Load both XLSX files
        tank1_df = pd.read_excel('Tank1.xlsx')
        tank2_df = pd.read_excel('Tank1-02.xlsx')
        
        # Extract relevant columns
        tank1_data = tank1_df[['曝气压力', '曝气流量']].copy()
        tank2_data = tank2_df[['曝气压力', '曝气流量']].copy()
        
        # Add file identifiers
        tank1_data['Tank'] = 'Tank1 (8.01~8.17)'
        tank2_data['Tank'] = 'Tank1-02 (8.26~9.06)'
        
        # Add time index (assuming regular sampling)
        tank1_data['Time_Index'] = range(len(tank1_data))
        tank2_data['Time_Index'] = range(len(tank2_data))
        
        # Clean data - remove any null values
        tank1_data = tank1_data.dropna()
        tank2_data = tank2_data.dropna()
        
        return tank1_data, tank2_data, None
        
    except Exception as e:
        return None, None, f"Error loading XLSX data: {str(e)}"

@st.cache_data
def load_pressure_data():
    """Load and preprocess the pressure data with new segmentation."""
    try:
        # Check if file exists
        if not Path('alldata.csv').exists():
            return None, "File 'alldata.csv' not found in current directory!"
        
        # Load main data
        df = pd.read_csv('alldata.csv')
        
        # Check columns
        expected_cols = ['Timestamp', 'DateTime (Local)', 'Pressure (kPa)']
        if list(df.columns) != expected_cols:
            return None, f"Unexpected columns. Expected: {expected_cols}, Got: {list(df.columns)}"
        
        df.columns = ['timestamp', 'datetime_local', 'pressure_kpa']
        
        # Convert datetime
        try:
            df['datetime_local'] = pd.to_datetime(df['datetime_local'], format='%Y-%m-%d %H:%M:%S.%f')
        except:
            df['datetime_local'] = pd.to_datetime(df['datetime_local'])
        
        df['date'] = df['datetime_local'].dt.date
        df['hour'] = df['datetime_local'].dt.hour
        df['day_of_week'] = df['datetime_local'].dt.day_name()
        
        # Convert pressure to numeric
        df['pressure_kpa'] = pd.to_numeric(df['pressure_kpa'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['pressure_kpa', 'datetime_local'])
        
        if len(df) == 0:
            return None, "No valid data found after cleaning!"
        
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
        st.subheader("Complete Pressure Timeline")
        
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
            title="MBR sensor and PLC data",
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
            stable_data = daily_data[daily_data['operational_state'] == 'Stable']
            
            # Calculate uptime percentage
            uptime_percent = (daily_data['operational_state'].isin(['Stable', 'Transition']).sum() / len(daily_data)) * 100
            
            daily_stat = {
                'date': date,
                'avg_pressure': stable_data['pressure_kpa'].mean() if len(stable_data) > 0 else None,
                'std_pressure': stable_data['pressure_kpa'].std() if len(stable_data) > 0 else None,
                'min_pressure': daily_data['pressure_kpa'].min(),
                'max_pressure': daily_data['pressure_kpa'].max(),
                'total_points': len(daily_data),
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
        
        st.caption("**Note**: avg_pressure and std_pressure are calculated using only Stable operation data (>20 kPa)")
    
    def create_xlsx_comparison(self):
        """Create XLSX files comparison charts."""
        st.subheader("Tank Data Comparison (XLSX Files)")
        
        # Load XLSX data
        with st.spinner("Loading XLSX comparison data..."):
            try:
                tank1_data, tank2_data, error = load_xlsx_data()
            except Exception as e:
                # Fallback: try loading directly without cache
                st.warning("Cache loading failed, trying direct load...")
                try:
                    import os
                    if not os.path.exists('Tank1.xlsx'):
                        error = "Tank1.xlsx file not found"
                        tank1_data, tank2_data = None, None
                    elif not os.path.exists('Tank1-02.xlsx'):
                        error = "Tank1-02.xlsx file not found"
                        tank1_data, tank2_data = None, None
                    else:
                        # Direct load without cache
                        tank1_df = pd.read_excel('Tank1.xlsx')
                        tank2_df = pd.read_excel('Tank1-02.xlsx')
                        
                        tank1_data = tank1_df[['曝气压力', '曝气流量']].copy()
                        tank2_data = tank2_df[['曝气压力', '曝气流量']].copy()
                        
                        tank1_data['Tank'] = 'Tank1 (8.01~8.17)'
                        tank2_data['Tank'] = 'Tank1-02 (8.26~9.06)'
                        
                        tank1_data['Time_Index'] = range(len(tank1_data))
                        tank2_data['Time_Index'] = range(len(tank2_data))
                        
                        tank1_data = tank1_data.dropna()
                        tank2_data = tank2_data.dropna()
                        
                        error = None
                except Exception as e2:
                    error = f"Direct load also failed: {str(e2)}"
                    tank1_data, tank2_data = None, None
        
        if tank1_data is None or tank2_data is None:
            st.error(f"Failed to load XLSX data: {error}")
            return
        
        # Show data overview
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Tank1 (8.01~8.17)**\n{len(tank1_data):,} data points")
        with col2:
            st.info(f"**Tank1-02 (8.26~9.06)**\n{len(tank2_data):,} data points")
        
        # Create side-by-side comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("曝气压力 (Aeration Pressure) Comparison")
            
            # Sample data for performance if too large
            tank1_sample = tank1_data.sample(min(10000, len(tank1_data))) if len(tank1_data) > 10000 else tank1_data
            tank2_sample = tank2_data.sample(min(10000, len(tank2_data))) if len(tank2_data) > 10000 else tank2_data
            
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
            
            fig_pressure.update_layout(
                title="曝气压力 Over Time",
                xaxis_title="Time Index",
                yaxis_title="曝气压力 (Aeration Pressure)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pressure, use_container_width=True)
            
            # Pressure statistics
            st.write("**曝气压力 Statistics:**")
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
            
            fig_flow.update_layout(
                title="曝气流量 Over Time",
                xaxis_title="Time Index",
                yaxis_title="曝气流量 (Aeration Flow)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_flow, use_container_width=True)
            
            # Flow statistics
            st.write("**曝气流量 Statistics:**")
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
                opacity=0.7,
                nbinsx=30,
                marker_color='blue'
            ))
            
            fig_pressure_dist.add_trace(go.Histogram(
                x=tank2_data['曝气压力'],
                name='Tank1-02 (8.26~9.06)',
                opacity=0.7,
                nbinsx=30,
                marker_color='red'
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
                opacity=0.7,
                nbinsx=30,
                marker_color='blue'
            ))
            
            fig_flow_dist.add_trace(go.Histogram(
                x=tank2_data['曝气流量'],
                name='Tank1-02 (8.26~9.06)',
                opacity=0.7,
                nbinsx=30,
                marker_color='red'
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
        # Filter out zero values for more accurate comparison
        tank1_pressure_nonzero = tank1_data[tank1_data['曝气压力'] != 0]['曝气压力']
        tank2_pressure_nonzero = tank2_data[tank2_data['曝气压力'] != 0]['曝气压力']
        tank1_flow_nonzero = tank1_data[tank1_data['曝气流量'] != 0]['曝气流量']
        tank2_flow_nonzero = tank2_data[tank2_data['曝气流量'] != 0]['曝气流量']
        
        pressure_diff = tank2_pressure_nonzero.mean() - tank1_pressure_nonzero.mean()
        flow_diff = tank2_flow_nonzero.mean() - tank1_flow_nonzero.mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average 曝气压力 Difference", 
                f"{pressure_diff:.3f}",
                f"{'Higher' if pressure_diff > 0 else 'Lower'} in Tank1-02"
            )
        
        with col2:
            st.metric(
                "Average 曝气流量 Difference", 
                f"{flow_diff:.3f}",
                f"{'Higher' if flow_diff > 0 else 'Lower'} in Tank1-02"
            )
        
        with col3:
            pressure_change_pct = (pressure_diff / tank1_pressure_nonzero.mean()) * 100
            st.metric(
                "曝气压力 Change %", 
                f"{pressure_change_pct:.2f}%"
            )
        
        with col4:
            flow_change_pct = (flow_diff / tank1_flow_nonzero.mean()) * 100
            st.metric(
                "曝气流量 Change %", 
                f"{flow_change_pct:.2f}%"
            )
        
        # Show data points info
        st.caption(f"**Note**: Calculations exclude zero values. "
                  f"Tank1: {len(tank1_pressure_nonzero):,} non-zero pressure points, {len(tank1_flow_nonzero):,} non-zero flow points. "
                  f"Tank1-02: {len(tank2_pressure_nonzero):,} non-zero pressure points, {len(tank2_flow_nonzero):,} non-zero flow points.")

    def create_pressure_distribution_analysis(self):
        """Create pressure distribution analysis."""
        st.subheader("Pressure Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall pressure histogram
            fig_hist = px.histogram(
                self.df,
                x='pressure_kpa',
                nbins=50,
                title="Overall Pressure Distribution",
                labels={'pressure_kpa': 'Pressure (kPa)', 'count': 'Frequency'},
                color_discrete_sequence=['lightblue']
            )
            
            # Add threshold lines
            fig_hist.add_vline(x=-0.1, line_dash="dash", line_color="gray", 
                              annotation_text="Shutdown Lower (-0.1)")
            fig_hist.add_vline(x=0.1, line_dash="dash", line_color="gray", 
                              annotation_text="Shutdown Upper (0.1)")
            fig_hist.add_vline(x=20.0, line_dash="dash", line_color="blue", 
                              annotation_text="Stable Threshold (20)")
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Pressure distribution by operational state
            fig_dist = px.histogram(
                self.df,
                x='pressure_kpa',
                color='operational_state',
                nbins=30,
                title="Pressure Distribution by Operational State",
                labels={'pressure_kpa': 'Pressure (kPa)', 'count': 'Frequency'},
                color_discrete_map={
                    'Shutdown': 'red',
                    'Transition': 'orange',
                    'Stable': 'green'
                },
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    
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
        st.success(f"{len(df):,} data points loaded")
        
        
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
        
        # Pressure distribution analysis (for pressure sensor data)
        self.create_pressure_distribution_analysis()
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
