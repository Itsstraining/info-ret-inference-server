from fastapi import FastAPI, APIRouter, Query
from model import StoredModel, load_model, fetch_model
import os

app = FastAPI()
router = APIRouter()

print("Fetching the model...")
# fetch and load model
# get environment variable
model_url = os.environ.get("MODEL_URL")
# ./assets/models/48
model_path = os.environ.get("MODEL_PATH")
# ./assets/models/48/fulltext_0.csv
full_text_path = os.environ.get("FULL_TEXT_PATH")

print("model url: ", model_url)
fetch_model(model_url)

loaded_model = load_model(model_path, full_text_path)


@router.get("/infer")
async def get_inference(query: str):
    # Perform inference on the query prompt and get the results
    results = loaded_model.infer(query)
    # Return the results as a list
    return {"results": results}


app.include_router(router)
