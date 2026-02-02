"""
app.py — Flask Web Application
================================
Serves the AI Ecommerce Pricing Engine dashboard.

Routes:
  GET  /          → Main dashboard (HTML)
  POST /optimize  → Run price optimization, return JSON for chart
  GET  /stats     → Category statistics JSON
  GET  /metrics   → Model performance metrics JSON
"""

from flask import Flask, render_template, request, jsonify
from pricing_engine import engine

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    categories = engine.get_categories()
    metrics    = engine.get_metrics()
    stats      = engine.get_category_stats()
    return render_template("index.html",
                           categories=categories,
                           metrics=metrics,
                           stats=stats)


# ---------------------------------------------------------------------------
# API endpoints (called by the frontend JS)
# ---------------------------------------------------------------------------
@app.route("/optimize", methods=["POST"])
def optimize():
    d = request.get_json()
    result = engine.optimize_price(
        category   = d.get("category", "Electronics"),
        competitor = float(d.get("competitor", 100)),
        demand     = float(d.get("demand", 60)),
        season     = float(d.get("season", 1.0)),
        rating     = float(d.get("rating", 4.0)),
        stock      = int(d.get("stock", 50)),
        alpha      = float(d.get("alpha", 0.6)),
    )
    return jsonify(result)


@app.route("/stats")
def stats():
    return jsonify(engine.get_category_stats())


@app.route("/metrics")
def metrics():
    return jsonify(engine.get_metrics())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
