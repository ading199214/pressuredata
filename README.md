# 🏭 Air Pressure Machine Analysis Dashboard

A comprehensive dashboard for analyzing air pressure machine data, detecting anomalies, and monitoring operational states.

## 📊 Features

### **Main Dashboard Components:**
- **📈 Complete Pressure Timeline**: Interactive visualization of all pressure data with operational states
- **🔄 Operational State Analysis**: Distribution and patterns of shutdown, ramp-up, transitioning, and stable states
- **📅 Daily Analysis**: Day-by-day breakdown with uptime metrics and statistics
- **🚨 Anomaly Detection**: Automatic detection of abnormal operation (e.g., broken membrane)
- **🎛️ Interactive Filters**: Date range, operational states, and pressure range filtering

### **Key Capabilities:**
- ✅ **Real-time anomaly detection** with confidence scoring
- ✅ **Broken membrane detection** (as demonstrated with Aug 26, 2025 data)
- ✅ **Normal vs abnormal operation classification**
- ✅ **Interactive charts** with zoom, pan, and hover details
- ✅ **Comprehensive statistics** and daily reports
- ✅ **Alert generation** with recommended actions

## 🚀 Quick Start

### **Option 1: Launch with Script**
```bash
python3 launch_dashboard.py
```

### **Option 2: Direct Streamlit Launch**
```bash
streamlit run dashboard.py
```

### **Option 3: Manual Setup**
1. Install dependencies:
   ```bash
   pip install streamlit plotly pandas numpy matplotlib seaborn scikit-learn
   ```

2. Launch dashboard:
   ```bash
   streamlit run dashboard.py
   ```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 📁 File Structure

```
pressureAnalysis/
├── alldata.csv                           # Main pressure data
├── dashboard.py                          # Interactive dashboard
├── launch_dashboard.py                   # Easy launcher script
├── enhanced_pressure_analysis.py         # Complete analysis with anomaly detection
├── pressure_analysis.py                  # Basic analysis script
├── anomaly_detection_plan.md            # Technical documentation
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── reports/                             # Daily analysis reports
├── plots/                              # Static visualizations
├── anomaly_reports/                    # Anomaly detection reports
└── anomaly_plots/                      # Anomaly visualization plots
```

## 🎛️ Dashboard Sections

### **1. Overview Metrics**
- Total data points analyzed
- Date range coverage
- Average pressure
- Overall uptime percentage
- Number of anomalous days detected

### **2. Complete Pressure Timeline**
- Interactive line chart of all pressure readings
- Color-coded operational states
- Highlighted anomalous periods
- Rolling average trend line
- Zoom and pan capabilities

### **3. Operational State Analysis**
- Pie chart distribution of operational states
- Normal vs abnormal stable operation breakdown
- Hourly operation patterns
- State transition analysis

### **4. Daily Analysis**
- Daily uptime bar chart
- Daily average pressure trends
- Interactive statistics table
- Anomaly highlighting

### **5. Anomaly Detection Analysis**
- Detailed analysis of each anomalous day
- Pressure statistics and alerts
- Timeline and distribution plots for anomalies
- Recommended maintenance actions

## 🚨 Anomaly Detection Logic

### **Detection Criteria:**
- **Membrane Failure**: Average pressure < 27 kPa with low variance (< 0.5 kPa)
- **Severe Pressure Loss**: Average pressure < 25 kPa
- **Unable to Reach Normal**: Maximum pressure < 28 kPa
- **Abnormal Variance**: Extremely low variance in stable operation

### **Confidence Levels:**
- **High Confidence** (≥75%): Clear equipment failure requiring immediate action
- **Medium Confidence** (≥50%): Pressure issues needing investigation
- **Low Confidence** (≥25%): Minor anomalies for monitoring

### **Example: Aug 26, 2025 Broken Membrane**
- **Average Pressure**: 25.41 kPa (vs normal 30 kPa)
- **Standard Deviation**: 0.177 kPa (extremely tight)
- **Max Pressure**: 26.12 kPa (unable to reach normal range)
- **Classification**: HIGH CONFIDENCE ANOMALY
- **Alerts**: Membrane failure + Unable to reach normal pressure

## 🔧 Interactive Features

### **Sidebar Filters:**
- **Date Range**: Select specific time periods
- **Operational States**: Filter by shutdown, ramp-up, transitioning, stable
- **Pressure Range**: Focus on specific pressure ranges

### **Chart Interactions:**
- **Zoom**: Click and drag to zoom into specific time periods
- **Pan**: Drag to navigate through data
- **Hover**: Detailed information on data points
- **Legend**: Click to show/hide data series

## 📊 Data Requirements

### **Input Format:**
The dashboard expects `alldata.csv` with columns:
```
Timestamp, DateTime (Local), Pressure (kPa)
1754635331213, "2025-08-08 14:42:11.213", 30.32
```

### **Data Processing:**
- Automatic operational state classification
- Rolling statistics calculation
- Anomaly detection scoring
- Daily aggregation and analysis

## 🎯 Use Cases

### **1. Real-Time Monitoring**
- Monitor current machine status
- Detect equipment failures early
- Track operational efficiency

### **2. Historical Analysis**
- Analyze trends over time
- Identify patterns in equipment behavior
- Plan maintenance schedules

### **3. Fault Diagnosis**
- Investigate specific anomaly events
- Compare normal vs abnormal operation
- Generate maintenance reports

### **4. Performance Optimization**
- Identify optimal operating conditions
- Track improvement after maintenance
- Monitor equipment degradation

## 🔍 Technical Details

### **Performance Optimization:**
- Automatic data sampling for large datasets (>50k points)
- Efficient data processing with pandas
- Cached data loading for better responsiveness

### **Visualization Technology:**
- **Plotly**: Interactive charts with zoom, pan, hover
- **Streamlit**: Web-based dashboard framework
- **Responsive design**: Works on desktop and tablet

### **Analysis Algorithms:**
- Statistical threshold detection
- Baseline comparison analysis
- Pattern recognition
- Confidence scoring system

## 🛠️ Troubleshooting

### **Common Issues:**

**Dashboard won't start:**
```bash
# Check if Streamlit is installed
pip list | grep streamlit

# Reinstall dependencies
pip install -r requirements.txt
```

**Data not loading:**
- Ensure `alldata.csv` is in the same directory
- Check file format matches expected structure
- Verify file permissions

**Performance issues:**
- Dashboard automatically samples large datasets
- Consider filtering date ranges for better performance
- Close other browser tabs to free memory

### **Browser Requirements:**
- Modern browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- Local network access to localhost:8501

## 📞 Support

The dashboard is designed to be self-explanatory with intuitive controls and clear visualizations. All analysis results include confidence scores and recommended actions for detected anomalies.

---

**🎯 Goal**: Provide comprehensive monitoring and early detection of equipment failures through intelligent data analysis and intuitive visualization.
