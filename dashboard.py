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
    page_title="Air Pressure Machine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        st.subheader("📈 Complete Pressure Timeline")
        
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
            title="Air Pressure Machine - New Segmentation (Shutdown: -0.1 to 0.1 kPa, Transition: <-0.1 OR 0.1-20 kPa, Stable: >20 kPa)",
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
        st.subheader("📅 Daily Analysis")
        
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
        st.subheader("📊 Daily Statistics Table")
        
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
        
        st.caption("📋 **Note**: avg_pressure and std_pressure are calculated using only Stable operation data (>20 kPa)")
    
    def create_pressure_distribution_analysis(self):
        """Create pressure distribution analysis."""
        st.subheader("📊 Pressure Distribution Analysis")
        
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
    
    def create_interactive_filters(self):
        """Create interactive filters in sidebar."""
        st.sidebar.header("🎛️ Dashboard Filters")
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(self.df['date'].min(), self.df['date'].max()),
            min_value=self.df['date'].min(),
            max_value=self.df['date'].max()
        )
        
        # Operational state filter
        states = st.sidebar.multiselect(
            "Operational States",
            options=self.df['operational_state'].unique(),
            default=self.df['operational_state'].unique()
        )
        
        # Pressure range filter
        pressure_range = st.sidebar.slider(
            "Pressure Range (kPa)",
            min_value=float(self.df['pressure_kpa'].min()),
            max_value=float(self.df['pressure_kpa'].max()),
            value=(float(self.df['pressure_kpa'].min()), float(self.df['pressure_kpa'].max())),
            step=0.1
        )
        
        # Apply filters
        mask = (
            (self.df['date'] >= date_range[0]) &
            (self.df['date'] <= date_range[1]) &
            (self.df['operational_state'].isin(states)) &
            (self.df['pressure_kpa'] >= pressure_range[0]) &
            (self.df['pressure_kpa'] <= pressure_range[1])
        )
        
        return mask
    
    def run_dashboard(self):
        """Run the complete dashboard."""
        st.title("🏭 Air Pressure Machine Dashboard - New Segmentation")
        st.markdown("### Shutdown: -0.1 to 0.1 kPa | Transition: other datapoints | Stable: > 20 kPa")
        st.markdown("---")
        
        # Load data
        with st.spinner("Loading pressure data..."):
            df, error = load_pressure_data()
        
        if df is None:
            st.error(f"Failed to load data: {error}")
            return
        
        self.df = df
        st.success(f"✅ Successfully loaded {len(df):,} data points!")
        
        # Apply filters
        filter_mask = self.create_interactive_filters()
        self.df = self.df[filter_mask]
        
        # Daily analysis (moved to top)
        self.create_daily_analysis()
        st.markdown("---")
        
        # Overview metrics
        self.create_overview_metrics()
        st.markdown("---")
        
        # Main timeline
        self.create_main_timeline()
        st.markdown("---")
        
        # Pressure distribution analysis
        self.create_pressure_distribution_analysis()
        
        # Footer
        st.markdown("---")
        st.markdown("**Dashboard created for Air Pressure Machine Analysis** | 📊 Redesigned segmentation for better visualization")

def main():
    dashboard = PressureDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
