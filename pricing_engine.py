"""
AI Ecommerce Pricing Engine — Core Module
==========================================
Generates synthetic product data, trains ML models, and exposes
prediction functions for the Flask web interface.

Models trained:
  1. Ridge Regression   → predicts expected revenue at any candidate price.
  2. Logistic Regression → classifies whether a price will lead to a sale
                            (conversion probability).

The engine picks the candidate price that maximises:
    score = alpha * predicted_revenue + (1 - alpha) * conversion_probability
where alpha is a user-tunable margin-vs-conversion trade-off weight.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json, os, warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  Synthetic Data Generation
# ---------------------------------------------------------------------------

np.random.seed(42)
N = 2000  # number of historical transactions

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports"]
CATEGORY_BASE = {
    "Electronics":    150,
    "Clothing":        45,
    "Books":           18,
    "Home & Kitchen":  35,
    "Sports":          60,
}

def _generate_dataset() -> pd.DataFrame:
    """Create a realistic synthetic transaction log."""
    categories   = np.random.choice(CATEGORIES, N)
    base_prices  = np.array([CATEGORY_BASE[c] for c in categories])

    # Price is base ± noise (simulates historical pricing attempts)
    price_ratio  = np.random.uniform(0.6, 1.6, N)
    price        = np.round(base_prices * price_ratio, 2)

    # Competitor price (slightly different from ours)
    competitor   = np.round(price * np.random.uniform(0.85, 1.15, N), 2)

    # Demand score (higher → more buyers searching)
    demand       = np.random.uniform(20, 100, N)

    # Seasonality factor  (1.0 = neutral, >1 = peak season)
    season       = np.round(np.random.uniform(0.7, 1.4, N), 2)

    # Rating (1–5 stars) — affects willingness to pay
    rating       = np.round(np.random.uniform(2.5, 5.0, N), 1)

    # Stock level (low stock → can charge more)
    stock        = np.random.randint(1, 500, N)

    # --- Target: did the customer actually buy? ---
    # Probability of conversion drops as price rises relative to competitor
    # and rises with demand, rating, seasonality
    logit = (
        2.5
        - 3.0 * (price / (competitor + 1e-6))
        + 0.02 * demand
        + 0.4  * rating
        + 0.6  * season
        - 0.002 * stock
    )
    prob_buy  = 1 / (1 + np.exp(-logit))
    converted = (np.random.rand(N) < prob_buy).astype(int)

    # Revenue = price * converted  (0 if no sale)
    revenue = np.round(price * converted, 2)

    df = pd.DataFrame({
        "category":   categories,
        "price":      price,
        "competitor": competitor,
        "demand":     demand,
        "season":     season,
        "rating":     rating,
        "stock":      stock,
        "converted":  converted,
        "revenue":    revenue,
    })
    return df


# ---------------------------------------------------------------------------
# 2.  Feature Engineering & Model Training
# ---------------------------------------------------------------------------

FEATURE_COLS = ["price", "competitor", "demand", "season", "rating", "stock",
                "price_to_competitor", "price_sq"]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_to_competitor"] = np.round(df["price"] / (df["competitor"] + 1e-6), 3)
    df["price_sq"]            = np.round(df["price"] ** 2, 2)
    return df


class PricingEngine:
    """Holds trained models and all prediction logic."""

    def __init__(self):
        self.df            = _add_features(_generate_dataset())
        self.scaler        = StandardScaler()
        self.revenue_model = None   # Ridge
        self.conv_model    = None   # LogisticRegression
        self.train_metrics = {}
        self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self):
        X = self.scaler.fit_transform(self.df[FEATURE_COLS])
        y_rev  = self.df["revenue"].values
        y_conv = self.df["converted"].values

        # --- Revenue model (Ridge Regression) ---
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_rev, test_size=0.2, random_state=42)
        self.revenue_model = Ridge(alpha=1.0)
        self.revenue_model.fit(X_tr, y_tr)
        rev_preds = self.revenue_model.predict(X_te)
        self.train_metrics["revenue_rmse"] = round(np.sqrt(mean_squared_error(y_te, rev_preds)), 2)
        self.train_metrics["revenue_r2"]   = round(r2_score(y_te, rev_preds), 3)

        # --- Conversion model (Logistic Regression) ---
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_conv, test_size=0.2, random_state=42)
        self.conv_model = LogisticRegression(max_iter=1000)
        self.conv_model.fit(X_tr2, y_tr2)
        conv_preds = self.conv_model.predict(X_te2)
        self.train_metrics["conv_accuracy"] = round(accuracy_score(y_te2, conv_preds), 3)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _build_row(self, category: str, competitor: float, demand: float,
                   season: float, rating: float, stock: int, price: float) -> np.ndarray:
        """Assemble a single feature vector for a candidate price."""
        row = pd.DataFrame([{
            "price": price, "competitor": competitor, "demand": demand,
            "season": season, "rating": rating, "stock": stock,
            "price_to_competitor": price / (competitor + 1e-6),
            "price_sq": price ** 2,
        }])[FEATURE_COLS]
        return self.scaler.transform(row)

    def predict_revenue(self, **kw) -> float:
        price = kw.pop("price")
        vec   = self._build_row(price=price, **kw)
        return float(self.revenue_model.predict(vec)[0])

    def predict_conversion(self, **kw) -> float:
        price = kw.pop("price")
        vec   = self._build_row(price=price, **kw)
        return float(self.conv_model.predict_proba(vec)[0][1])

    # ------------------------------------------------------------------
    # Core: find the optimal price
    # ------------------------------------------------------------------
    def optimize_price(self, category: str, competitor: float, demand: float,
                       season: float, rating: float, stock: int,
                       alpha: float = 0.6, n_candidates: int = 200) -> dict:
        """
        Sweep candidate prices in [0.5x … 1.8x competitor] and pick the one
        that maximises  alpha*revenue + (1-alpha)*conversion_prob.
        """
        base    = CATEGORY_BASE.get(category, 50)
        lo      = max(1.0, base * 0.4)
        hi      = base * 2.2
        prices  = np.linspace(lo, hi, n_candidates)

        revenues, conversions, scores = [], [], []
        common = dict(category=category, competitor=competitor, demand=demand,
                      season=season, rating=rating, stock=stock)

        for p in prices:
            rev  = self.predict_revenue(price=p, **common)
            conv = self.predict_conversion(price=p, **common)
            revenues.append(rev)
            conversions.append(conv)
            scores.append(alpha * rev + (1 - alpha) * conv * 100)

        best_idx = int(np.argmax(scores))

        return {
            "optimal_price":      round(float(prices[best_idx]), 2),
            "expected_revenue":   round(float(revenues[best_idx]), 2),
            "conversion_prob":    round(float(conversions[best_idx]) * 100, 1),
            "current_competitor": competitor,
            "alpha":              alpha,
            # Full sweep data for the chart
            "sweep": {
                "prices":      [round(float(p), 2) for p in prices],
                "revenues":    [round(float(r), 2) for r in revenues],
                "conversions": [round(float(c) * 100, 1) for c in conversions],
                "scores":      [round(float(s), 2) for s in scores],
            }
        }

    # ------------------------------------------------------------------
    # Utility: category stats from training data
    # ------------------------------------------------------------------
    def get_category_stats(self) -> dict:
        stats = {}
        for cat in CATEGORIES:
            sub = self.df[self.df["category"] == cat]
            stats[cat] = {
                "avg_price":      round(sub["price"].mean(), 2),
                "avg_revenue":    round(sub["revenue"].mean(), 2),
                "conversion_rate": round(sub["converted"].mean() * 100, 1),
                "count":          int(len(sub)),
            }
        return stats

    def get_metrics(self) -> dict:
        return self.train_metrics

    def get_categories(self) -> list:
        return CATEGORIES


# ---------------------------------------------------------------------------
# 3.  Module-level singleton (imported by app.py)
# ---------------------------------------------------------------------------
engine = PricingEngine()
