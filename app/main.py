from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pre_processing import pre_process_msg
from pydantic import BaseModel, Field

# Global dict which contains all the ml models loaded into the memory
ml_models = {}


class Response(BaseModel):
    is_spam: bool
    label: str
    confidence: float


class MessageRequest(BaseModel):
    message: str = Field(
        ..., example="Congratulations, you won a 1000$ free gift card."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP CODE
    try:
        # load all the ML models into memory
        ml_models["label_encoder"] = joblib.load("model/label_encoder.joblib")
        ml_models["tfidf_vectorizer"] = joblib.load("model/tfidf_vectorizer.joblib")
        ml_models["spam_msg_detector"] = joblib.load("model/spam_msg_detector.joblib")
    except FileNotFoundError as e:
        print(f"ML Model artifacts not found. Try running app/training.py first.{e}")

    # YIELD
    yield

    # SHUTDOWN CODE
    ml_models.clear()


app = FastAPI(title="Spam Message Detection Service", lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", response_model=Response)
async def predict_spam(payload: MessageRequest):
    if not ml_models:
        raise HTTPException(status_code=503, detail="ML Models pipeline is not ready.")

    # Convert the message into pandas Data frame
    input_df = pd.DataFrame({"v2": [payload.message]})

    # Pre-process the text
    input_df = pre_process_msg(input_df, "v2")

    # Vectorize the message using Tfidf
    vectorizer = ml_models["tfidf_vectorizer"]
    vectorized_df = vectorizer.transform(input_df["v2"])

    # Get the label encoder
    le = ml_models["label_encoder"]

    # Get the prediction model
    model = ml_models["spam_msg_detector"]

    # This gets us the 1 or 0 for spam or ham (Whatever it has internally for dictionary)
    prediction_res = model.predict(vectorized_df)[0]

    # This transaltes that label into the human readable form 'spam' or 'ham'
    human_label = le.inverse_transform([prediction_res])[0]
    # Inverse transform returns us the 2-d array, and we extract the useful part for us

    probabilities = model.predict_proba(vectorized_df)[0]
    confidence_score = float(probabilities[prediction_res])

    return Response(
        is_spam=bool(human_label == "spam"),
        label=human_label,
        confidence=round(confidence_score * 100, 2),
    )
