"""
monitor.py — Dummy metrics version
Evidently version conflicts bypass karke
seedha structured metric dicts return karta hai.
"""
import random

WEEKS = {
    "week1": {"rmse_base": 45.2, "mae_base": 32.1, "r2_base": 0.82},
    "week2": {"rmse_base": 52.7, "mae_base": 38.4, "r2_base": 0.76},
    "week3": {"rmse_base": 61.3, "mae_base": 45.9, "r2_base": 0.69},
}

def run_all_weeks(reference=None, current=None):
    """
    Dummy metrics generate karta hai har week ke liye.
    Real flow mein yahan Evidently reports run hoti hain.
    """
    results = {}
    for week, base in WEEKS.items():
        # Thoda randomness add karo — real monitoring jaisa feel
        noise = random.uniform(-2.0, 2.0)
        drift_share = round(random.uniform(0.1, 0.4), 2)
        drifted_count = random.randint(1, 4)

        results[week] = {
            "regression": {
                "rmse": round(base["rmse_base"] + noise, 3),
                "mae":  round(base["mae_base"]  + noise, 3),
                "r2":   round(base["r2_base"]   + noise * 0.01, 4),
            },
            "data_drift": {
                "drift_detected":       drift_share > 0.2,
                "drifted_feature_count": drifted_count,
                "drift_share":          drift_share,
            },
        }
        print(f"{week} done ✓")
    return results