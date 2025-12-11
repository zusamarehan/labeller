from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ImageRequest(BaseModel):
    url: str
    format: str

@app.post("/process")
def process_image(request: ImageRequest):
    # You can do logic here
    # Example: print(request.url, request.format)

    return {"success": True}

