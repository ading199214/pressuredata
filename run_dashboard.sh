#!/bin/bash
# Pressure Analysis Dashboard Launcher
# This ensures we use the correct Python environment

echo "🚀 Starting Pressure Analysis Dashboard..."
echo "📍 Working directory: $(pwd)"
echo "🐍 Using Python: /opt/anaconda3/bin/python"
echo "📊 Using Streamlit: /opt/anaconda3/bin/streamlit"
echo ""

# Use Anaconda environment explicitly
/opt/anaconda3/bin/streamlit run dashboard.py

echo "Dashboard stopped."
