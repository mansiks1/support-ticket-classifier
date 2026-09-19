import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse
from src.config import MODELS
from src.predict import Predictor


def create_app(model_path=None):
    path = Path(model_path or os.environ.get("MODEL_PATH", MODELS / "practical.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = Predictor(path) if path.exists() else None
        yield

    app = FastAPI(title="Support Ticket Classifier", version="1.0.0", lifespan=lifespan)

    def ready():
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=503, detail="Model not available; see setup instructions"
            )
        return predictor

    @app.get("/health")
    def health():
        ready()
        return {"status": "ok", "model_loaded": True}

    @app.get("/model-info")
    def model_info():
        return ready().metadata

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        return {"predictions": ready().predict(request.text)}

    return app


app = create_app()
