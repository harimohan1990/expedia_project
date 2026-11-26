"""FastAPI Application entrypoint with request logging, latency headers, and Prometheus metrics."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.routes.forecasting import router as forecasting_router
from configs.settings import get_config, get_logger

logger = get_logger("expedia_ad_api")
cfg = get_config()

app = FastAPI(
    title=cfg.project.name,
    version=cfg.project.version,
    description=(
        "Production ML serving API for Audience Reach Forecasting, Inventory Forecasting (7/14/30d), "
        "Pre-Launch Campaign Performance Prediction, and Causal Uplift Targeting Recommendations."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.middleware("http")
async def request_logging_and_latency_middleware(request: Request, call_next) -> Response:
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "%s %s -> %d (%.2f ms)", request.method, request.url.path, response.status_code, elapsed_ms
    )
    return response


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(forecasting_router)
