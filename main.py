from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import base64
from contextlib import asynccontextmanager

from ai_engine import InpaintingApp

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models... This may take a while.")
    app_state["model"] = InpaintingApp()
    print("Models loaded successfully!")
    yield
    app_state.clear()


app = FastAPI(title="Inpainting API", lifespan=lifespan)


def pil_to_base64(img: Image.Image) -> str:
    if img is None:
        return None
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def prepare_image_data(image_file: UploadFile, mask_file: UploadFile):
    image_bytes = await image_file.read()
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    bg_np = np.array(img_pil)

    mask_bytes = await mask_file.read()
    mask_pil = Image.open(io.BytesIO(mask_bytes))
    mask_np = np.array(mask_pil)

    image_data = {
        "background": bg_np,
        "layers": [mask_np]
    }
    return image_data


@app.post("/api/remove")
async def remove_object_api(
        image: UploadFile = File(...),
        mask: UploadFile = File(...)
):
    try:
        model = app_state.get("model")
        if not model:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

        image_data = await prepare_image_data(image, mask)

        lama_img, final_img = model.remove_object(image_data)

        if lama_img is None:
            raise HTTPException(status_code=400, detail="Invalid image or mask")

        return JSONResponse(content={
            "status": "success",
            "lama_image": pil_to_base64(lama_img),
            "final_image": pil_to_base64(final_img) if final_img else None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/replace")
async def replace_object_api(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form("")
):
    try:
        model = app_state.get("model")
        if not model:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

        image_data = await prepare_image_data(image, mask)

        lama_img, final_img = model.replace_object(image_data, prompt)

        if lama_img is None or final_img is None:
            raise HTTPException(status_code=400, detail="Invalid image or mask")

        return JSONResponse(content={
            "status": "success",
            "lama_image": pil_to_base64(lama_img),
            "final_image": pil_to_base64(final_img)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
