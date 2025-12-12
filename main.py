from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputModel(BaseModel):
    param1: str
    param2: str

@app.post("/echo")
def echo_data(data: InputModel):
    return {
        "param1": data.param1,
        "param2": data.param2
    }
