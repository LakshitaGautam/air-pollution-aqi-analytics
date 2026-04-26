# 🌫️ Air Pollution & Public Health Risk Analytics

A comprehensive, data-driven analytics platform analyzing India's air quality 
across all 36 states and union territories from 2022 to 2025, with machine 
learning-based AQI forecasting for 2026.

Built as part of the Fundamentals of Data Analytics Lab at Jaypee Institute 
of Information Technology, Noida (B.Tech CSE, Sem 4, 2025–26).

---

## 📊 Project Overview

India consistently ranks among the most polluted countries globally. This 
project addresses the lack of structured, visual, and predictive air quality 
analysis by building an end-to-end analytics pipeline — from raw data 
preprocessing to an interactive multi-section HTML dashboard.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data ingestion, cleaning, aggregation |
| NumPy | Numerical operations and array handling |
| Plotly Express | High-level interactive charts |
| Plotly Graph Objects | Custom multi-trace visualizations |
| Scikit-learn | Polynomial Regression for ML forecasting |
| HTML/CSS/JavaScript | Self-contained dashboard output |

---

## 📁 Dataset

- **Source:** Central Pollution Control Board (CPCB), India
- **File:** `AQI.csv`
- **Coverage:** 36 states/UTs, 200+ cities, Jan 2022 – Dec 2025
- **Pollutants tracked:** PM2.5, PM10, NO2, SO2, CO, O3, NH3
- **Columns:** country, state, city, last_update, pollutant_id, 
  pollutant_min, pollutant_max, pollutant_avg

---

## ⚙️ Key Features

- **Data Preprocessing:** Datetime parsing, temporal feature engineering 
  (year, month, quarter), null handling, geographic region mapping, 
  severity classification (High/Medium/Low)
- **41 Interactive Charts** across 14 chart types — bar, line, scatter, 
  heatmap, choropleth map, sunburst, treemap, radar, waterfall, funnel, 
  box plot, violin, pie, area
- **8 Analytical Sections:** Overview, Predictions, States, Regions, 
  Pollutants, Advanced, Time Series, Rankings
- **ML Forecasting:** Polynomial Regression (degree-2) via Scikit-learn 
  forecasting 2026 AQI for every state, region, and nationally
- **Self-contained Dashboard:** Single HTML file output with embedded 
  Plotly JS, custom CSS, and vanilla JS interactivity — no server needed

---

## 🔍 Key Findings

- North region is the most polluted across all years
- Island territories maintain the cleanest air quality
- PM2.5 and PM10 are the dominant pollutants nationwide
- Pollution peaks sharply in winter months (Oct–Jan) due to temperature 
  inversion and stubble burning
- Many high-risk states projected to see stable or worsening AQI in 2026

---

## 🚀 How to Run

1. Clone the repository
   git clone https://github.com/LakshitaGautam/air-pollution-aqi-analytics.git

2. Install dependencies
   pip install pandas numpy plotly scikit-learn

3. Run the script
   python projectfda.py

4. Open the output dashboard
   Open AQIHTML.html in any browser
