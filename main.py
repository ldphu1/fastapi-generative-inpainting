from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import io
import base64
from PIL import Image
from ai_engine import InpaintingApp

app = FastAPI(title="AI Inpainting API", version="1.1")
engine = None


@app.on_event("startup")
def startup():
    global engine
    engine = InpaintingApp()


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/api/v1/replace")
async def replace(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form("")):
    try:
        img_raw = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), 1)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imdecode(np.frombuffer(await mask.read(), np.uint8), 0)
        _, mask_bin = cv2.threshold(mask_raw, 127, 1, cv2.THRESH_BINARY)

        m_lama, m_sdxl, m_blend = engine.refine_mask(mask_bin)
        img_pil = Image.fromarray(img_rgb)

        lama_res = engine.lama(img_pil, Image.fromarray(m_lama))
        engine.flush_memory()

        final_res = engine.stable_diffusion(lama_res, Image.fromarray(m_sdxl), prompt)

        return JSONResponse({"status": "ok", "image": to_b64(final_res)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)