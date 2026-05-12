from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from monitoring.monitor import run_all_weeks
from monitoring.metrics import update_metrics

app = FastAPI(title="Bike Demand Monitor")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/run-monitor")
def run_monitor():
    results = run_all_weeks()
    update_metrics(results)
    return {"status": "done", "results": results}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)