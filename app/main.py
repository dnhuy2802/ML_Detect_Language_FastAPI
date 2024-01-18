from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class TextIn(BaseModel):
    text: str

class TextOut(BaseModel):
    language: str

@app.get("/")
def root():
    return {"model_version": model_version}

@app.post("/predict", response_model=TextOut)
def predict(text_in: TextIn):
    print("text in is: ",text_in)
    language = predict_pipeline(text_in.text)
    return {"language": language}
