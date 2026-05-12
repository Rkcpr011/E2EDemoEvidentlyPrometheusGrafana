# 🚴 Bike Demand Monitoring — End-to-End ML Observability

A production-style ML monitoring pipeline that tracks model performance and data drift in real time using **Evidently**, **Prometheus**, and **Grafana** — without Docker.

---

## 📌 Project Objective

Monitor a trained bike demand forecasting model week-over-week by tracking:
- **Regression metrics** — RMSE, MAE, R² Score
- **Data drift** — whether incoming data has drifted from training distribution

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LOCAL MACHINE                      │
│                                                         │
│   ┌─────────────┐     scrape      ┌──────────────────┐  │
│   │   FastAPI   │ ◄────────────── │   Prometheus     │  │
│   │  :8000      │  /metrics       │   :9090          │  │
│   │             │                 └────────┬─────────┘  │
│   │ /health     │                          │ data source │
│   │ /run-monitor│                 ┌────────▼─────────┐  │
│   │ /metrics    │                 │    Grafana        │  │
│   └─────────────┘                 │    :3000          │  │
│                                   │  (Dashboards)     │  │
│                                   └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. POST /run-monitor
        │
        ▼
2. Evidently generates metrics (RMSE, MAE, R², Drift)
        │
        ▼
3. Prometheus client exposes metrics at /metrics
        │
        ▼
4. Prometheus scrapes /metrics every 15s
        │
        ▼
5. Grafana queries Prometheus → Visualizes dashboards
```

---

## 🛠️ Tech Stack

| Tool | Role | Version |
|---|---|---|
| FastAPI | REST API to serve and expose metrics | Latest |
| Evidently | ML monitoring — drift & regression reports | 0.4.33 |
| Prometheus Client | Expose metrics in Prometheus format | Latest |
| Prometheus | Metrics scraping and storage | 3.11.3 |
| Grafana | Dashboard visualization | 13.x OSS |
| scikit-learn | Model training (Random Forest) | 1.5.2 |
| MLflow | Experiment tracking | Latest |
| Python | Core language | 3.13 |

---

## 📁 Project Structure

```
bicycle_monitoring/
│
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app — 3 endpoints
│
├── data/
│   └── data_loader.py       # Load & split bike demand dataset
│
├── model/
│   ├── train.py             # Train RandomForest + MLflow logging
│   └── predict.py           # Add predictions to dataframe
│
├── monitoring/
│   ├── monitor.py           # Evidently reports → metric dicts
│   └── metrics.py           # Prometheus Gauge definitions
│
├── grafana/
│   └── provisioning/        # Grafana config (for Docker use)
│
├── mlruns/                  # MLflow experiment runs
├── reports/                 # Evidently HTML reports (optional)
├── myvenv/                  # Virtual environment
├── prometheus.yml           # Prometheus scrape config
├── docker-compose.yml       # Docker setup (alternative)
├── Dockerfile               # FastAPI container (for Docker)
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables
```

---

## ⚙️ Setup — Without Docker (Local Windows)

### Prerequisites

- Python 3.13
- Prometheus ([download .zip](https://prometheus.io/download/))
- Grafana OSS ([download .msi](https://grafana.com/grafana/download?platform=windows))

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Rkcpr011/E2EDemoEvidentlyPrometheusGrafana.git
cd E2EDemoEvidentlyPrometheusGrafana/bicycle_monitoring
```

---

### Step 2 — Create Virtual Environment & Install Dependencies

```bash
python -m venv myvenv
myvenv\Scripts\activate
pip install -r requirements.txt
```

**Key dependency versions (version-sensitive):**

```
evidently==0.4.33
scikit-learn==1.5.2
```

> ⚠️ **Why these exact versions?**
> Evidently 0.4.33 internally calls `mean_squared_error(..., squared=False)` which was removed in scikit-learn 1.6. scikit-learn 1.5.2 is the latest version that still supports this — and has pre-built wheels for Python 3.13.

---

### Step 3 — Configure Prometheus

Edit `prometheus.yml` in the Prometheus download folder:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "bike_monitor"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
```

> ⚠️ Use `localhost:8000` — NOT `fastapi:8000` (that's Docker-only).

---

### Step 4 — Run All Three Services

**Terminal 1 — FastAPI:**

```bash
cd bicycle_monitoring
myvenv\Scripts\activate
uvicorn api.main:app --reload
```

**Terminal 2 — Prometheus:**

```bash
cd C:\path\to\prometheus-3.11.3.windows-amd64
.\prometheus.exe --config.file=prometheus.yml
```

**Grafana** starts automatically as a Windows service after installation.

---

## 🌐 Service URLs

| Service | URL | Credentials |
|---|---|---|
| FastAPI | http://localhost:8000 | — |
| FastAPI Docs | http://localhost:8000/docs | — |
| Prometheus UI | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## 🔁 How to Use

### 1. Trigger Monitoring Run

```
GET http://localhost:8000/run-monitor
```

This generates metrics for 3 weeks and pushes to Prometheus.

Expected response:

```json
{
  "status": "done",
  "results": {
    "week1": {
      "regression": { "rmse": 45.82, "mae": 32.72, "r2": 0.826 },
      "data_drift": { "drift_detected": false, "drifted_feature_count": 1, "drift_share": 0.19 }
    },
    "week2": { ... },
    "week3": { ... }
  }
}
```

### 2. Check Raw Metrics

```
GET http://localhost:8000/metrics
```

Output (Prometheus format):

```
bike_model_rmse{week="week1"} 45.82
bike_model_mae{week="week1"} 32.72
bike_model_r2{week="week1"} 0.826
bike_data_drift_detected{week="week1"} 0.0
bike_data_drift_share{week="week1"} 0.19
bike_data_drifted_feature_count{week="week1"} 1.0
```

### 3. Verify Prometheus is Scraping

```
http://localhost:9090/targets
```

`bike_monitor` should show **UP** ✅

### 4. Query in Prometheus UI

```
bike_model_rmse
bike_model_mae
bike_data_drift_detected
```

---

## 📊 Grafana Dashboard Setup

1. Open `http://localhost:3000` → Login with `admin/admin`
2. **Connections** → **Data Sources** → **Add** → **Prometheus**
3. URL: `http://localhost:9090` → **Save & Test**
4. **Dashboards** → **New** → **Add visualization**
5. Select metric: `bike_model_rmse` with label `week`
6. Repeat for `bike_model_mae`, `bike_model_r2`, `bike_data_drift_detected`

---

## 📦 Prometheus Metrics Reference

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `bike_model_rmse` | Gauge | `week` | Root Mean Squared Error |
| `bike_model_mae` | Gauge | `week` | Mean Absolute Error |
| `bike_model_r2` | Gauge | `week` | R² Score |
| `bike_data_drift_detected` | Gauge | `week` | 1 if drift detected, 0 otherwise |
| `bike_data_drift_share` | Gauge | `week` | Share of drifted features (0–1) |
| `bike_data_drifted_feature_count` | Gauge | `week` | Number of drifted features |

---

## 🐛 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No module named 'evidently.pipeline'` | Evidently version too new | `pip install evidently==0.4.33` |
| `got unexpected keyword argument 'squared'` | scikit-learn >= 1.6 | `pip install scikit-learn==1.5.2` |
| `No module named 'api'` | Wrong folder for uvicorn | Run from `bicycle_monitoring/` root |
| Prometheus target DOWN | FastAPI not running | Start FastAPI first |
| `fastapi:8000` not found | Docker hostname in config | Change to `localhost:8000` in prometheus.yml |
| MLflow path error (WIN%2010) | Space in Windows username | Set `MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db` in `.env` |

---

## 🎯 Key Concepts Learned

| Concept | Tool | What It Does |
|---|---|---|
| Model monitoring | Evidently | Detects regression degradation and data drift |
| Metrics exposition | prometheus-client | Exposes metrics at `/metrics` in Prometheus format |
| Metrics scraping | Prometheus | Pulls metrics every 15s and stores as time series |
| Visualization | Grafana | Queries Prometheus and renders dashboards |
| Experiment tracking | MLflow | Logs model params, metrics, and artifacts |
| API serving | FastAPI | Serves monitoring endpoints |

---

## 🚀 Future Improvements

- [ ] Add Docker support for one-command startup
- [ ] Set up Grafana alerting (email/Slack) on drift detection
- [ ] Replace dummy metrics with real Evidently reports (fix version compatibility)
- [ ] Add scheduled monitoring (APScheduler or Celery)
- [ ] Deploy to cloud (AWS EC2 / GCP VM)
- [ ] Add target drift detection alongside data drift

---

## 👤 Author

**Rakesh Kumar**
SDE2 @ CGI | Masters in AI/ML
- GitHub: [@Rkcpr011](https://github.com/Rkcpr011)

---

## 📄 License

MIT License
