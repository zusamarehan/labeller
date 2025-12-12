from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import io
import torch
from sam3.model_builder import build_sam3_image_model

from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results, draw_box_on_image, normalize_bbox

app = FastAPI(title="SAM3 Segmentation API")

# Load model once at startup
bpe_path = "<PATH_TO_BPE_FILE>"  # e.g., sam3/assets/bpe_simple_vocab_16e6.txt.gz
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Request body for text/box prompts
class SegmentRequest(BaseModel):
    text_prompt: str | None = None
    boxes: list[list[float]] | None = None  # [[x,y,w,h, label], ...] label=1 for positive, 0 for negative

@app.post("/segment")
async def segment(file: UploadFile = File(...), payload: SegmentRequest = None):
    # Load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # Setup SAM inference
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)

    # Apply text prompt if given
    if payload and payload.text_prompt:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=payload.text_prompt)

    # Apply box prompts if given
    if payload and payload.boxes:
        for box in payload.boxes:
            x, y, w, h, label = box
            cxcywh = box_xywh_to_cxcywh(torch.tensor([x, y, w, h]).view(-1,4))
            norm_box = normalize_bbox(cxcywh, width, height).flatten().tolist()
            inference_state = processor.add_geometric_prompt(
                state=inference_state, box=norm_box, label=bool(label)
            )

    # Save segmented image to bytes
    fig = plot_results(image, inference_state, show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    buf.seek(0)

    return {"success": True, "image_bytes": buf.getvalue().hex()}
