# AI Ecommerce Pricing Engine

Dynamically adjusts product prices to optimise margins and conversions using machine learning.

## What It Does

The engine sweeps 200 candidate prices for a product and picks the one that maximises a weighted score combining **expected revenue** (Ridge Regression) and **conversion probability** (Logistic Regression). Users can tune the margin-vs-conversion trade-off via a slider.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Scikit-Learn (Ridge Regression, Logistic Regression) |
| Data | Pandas, NumPy (synthetic dataset of 2 000 transactions) |
| Web UI | Flask + Jinja2 + Chart.js |
| Visualisation | Chart.js (multi-axis line chart with optimal price marker) |

## Setup & Run

```bash
# 1. Make sure Python 3.8+ is installed
python --version

# 2. Install dependencies
pip install flask scikit-learn pandas numpy

# 3. Run the app
python app.py

# 4. Open in browser
# http://127.0.0.1:5000
```

## Project Structure

```
project/
├── app.py                  # Flask routes & server entry point
├── pricing_engine.py       # Core ML logic (data gen, training, prediction)
├── templates/
│   └── index.html          # Single-page dashboard (HTML + CSS + JS)
└── README.md               # This file
```

## How the Models Work

1. **Data Generation** — 2 000 synthetic transactions across 5 product categories are created with realistic price, demand, seasonality, rating, stock, and competitor-price relationships.
2. **Ridge Regression** — Trained to predict the revenue for any candidate price given the product features.
3. **Logistic Regression** — Trained to estimate the probability a customer converts (buys) at a given price.
4. **Optimisation** — At inference time, 200 candidate prices are evaluated. The engine picks the price that maximises: `alpha × revenue + (1 − alpha) × conversion_probability`.

## How to Use

1. Select a product **category**.
2. Enter the **competitor price**, demand score, seasonality, rating, and stock level.
3. Adjust the **Strategy Weight** slider (left = prioritise conversions, right = prioritise margin).
4. Click **Optimize Price** — the dashboard shows the recommended price, expected revenue, conversion probability, and a full price-sweep chart.
