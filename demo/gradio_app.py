import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionXLInpaintPipeline
from simple_lama_inpainting import SimpleLama
from transformers import BlipProcessor, BlipForConditionalGeneration
import gc

class InpaintingApp:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):

        self.lama = SimpleLama()

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        self.pipe.enable_model_cpu_offload()

        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(self.device)

    def generate_auto_prompt(self, image_pil, mask):
        mask = mask.astype(np.uint8)

        x, y, w, h = cv2.boundingRect(mask)

        if w == 0 or h == 0:
            print("Mask not found")
            return "object"

        box = (x, y, x + w, y + h)

        cropped_image = image_pil.crop(box)

        input = self.caption_processor(cropped_image, return_tensors="pt").to(self.device, torch.float16)
        out = self.caption_model.generate(**input)
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)

        return caption

    def enrich_prompt(self, user_prompt):
        words = user_prompt.split()
        if len(words) < 8:
            quality_tags = "matching the exact lighting and shadows of the surrounding environment, highly detailed, masterpiece, 8k, professional lighting"
            return f"{user_prompt}, {quality_tags}"
        return user_prompt

    def get_brush_mask(self, image_data):
        if image_data is None:
            return None, None

        if isinstance(image_data, dict):
            image = image_data.get('background')
            if image is None:
                return None, None

            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            if 'layers' in image_data and image_data['layers']:
                brush_mask = np.sum(image_data['layers'], axis=0)
            else:
                brush_mask = None
        else:
            return None, None

        if brush_mask is not None and np.max(brush_mask) > 0:
            if len(brush_mask.shape) >= 3:
                if brush_mask.shape[2] == 4:
                    brush_mask = brush_mask[:, :, 3]
                else:
                    brush_mask = cv2.cvtColor(brush_mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

            _, final_mask = cv2.threshold(brush_mask.astype(np.uint8), 1, 1, cv2.THRESH_BINARY)
        else:
            final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        return image, final_mask

    def refine_mask(self, mask):
        mask = (mask * 255).astype(np.uint8) if np.max(mask) <= 1 else mask.astype(np.uint8)

        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return mask, mask, mask

        max_dim = max(w, h)

        k_lama = int(max_dim * 0.05)
        k_lama = max(7, min(k_lama, 45))
        if k_lama % 2 == 0: k_lama += 1

        k_sdxl = int(max_dim * 0.01)
        k_sdxl = max(5, min(k_sdxl, 20))
        if k_sdxl % 2 == 0: k_sdxl += 1

        kernel_lama = np.ones((k_lama, k_lama), np.uint8)
        kernel_sdxl = np.ones((k_sdxl, k_sdxl), np.uint8)

        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        mask_lama = cv2.dilate(mask_clean, kernel_lama, iterations=1)
        mask_lama = cv2.morphologyEx(mask_lama, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        mask_sdxl = cv2.dilate(mask_clean, kernel_sdxl, iterations=1)

        blur_ksize = int(max_dim * 0.05)
        blur_ksize = max(5, min(blur_ksize, 55))
        if blur_ksize % 2 == 0: blur_ksize += 1

        mask_blend = cv2.GaussianBlur(mask_sdxl, (blur_ksize, blur_ksize), 0)

        return mask_lama, mask_sdxl, mask_blend

    def blend(self, original, generated, mask):
        original = np.array(original)
        generated = np.array(generated)

        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        blended = original * (1 - mask) + generated * mask
        return Image.fromarray(blended.astype(np.uint8))

    def poisson_blend(self, original, generated, mask):
        original = np.array(original)
        generated = np.array(generated)

        _, mask_binary = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        mask_binary[0, :] = 0
        mask_binary[-1, :] = 0
        mask_binary[:, 0] = 0
        mask_binary[:, -1] = 0

        x, y, w, h = cv2.boundingRect(mask_binary)

        center = (x + w // 2, y + h // 2)

        try:
            blended = cv2.seamlessClone(generated, original, mask_binary, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"Poisson failed: {e}. Fallback to alpha blend.")
            return self.blend(Image.fromarray(original), Image.fromarray(generated), mask)

        return Image.fromarray(blended)

    def hybrid_blend(self, original, generated, mask_raw, mask_refined):
        mask_uint8 = mask_raw.astype(np.uint8)
        area = np.sum(mask_uint8)
        total = mask_uint8.shape[0] * mask_uint8.shape[1]
        ratio = area / total

        x, y, w, h = cv2.boundingRect(mask_uint8)
        bbox_ratio = (w * h) / total

        if ratio < 0.05 and bbox_ratio < 0.1:
            return self.blend(original, generated, mask_refined)
        else:
            return self.poisson_blend(original, generated, mask_refined)

    def stable_diffusion(self, image, mask, prompt):
        default_negative = "ugly, deformed, bad anatomy, bad proportions, artifacts, low resolution"

        original_width, original_height = image.size

        mask_np = np.array(mask.convert('L'))

        x, y, w, h = cv2.boundingRect(mask_np)

        if w == 0 or h == 0:
            return image

        pad_x = int(w * 0.5)
        pad_y = int(h * 0.5)

        if pad_x * 2 + w < 512:
            pad_x = (512 - w) // 2
        if pad_y * 2 + h < 512:
            pad_y = (512 - h) // 2

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(original_width, x + w + pad_x)
        y2 = min(original_height, y + h + pad_y)

        crop_w = x2 - x1
        crop_h = y2 - y1

        box = (x1, y1, x2, y2)

        crop_mask = mask.crop(box)
        crop_image = image.crop(box)

        scale = 1024.0 / max(crop_w, crop_h)

        target_width = (int(crop_w * scale) // 8) * 8
        target_height = (int(crop_h * scale) // 8) * 8

        crop_image_resized = crop_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        crop_mask_resized = crop_mask.resize((target_width, target_height), Image.Resampling.NEAREST)

        result = self.pipe(
            prompt=prompt,
            negative_prompt= default_negative,
            image=crop_image_resized,
            mask_image=crop_mask_resized,
            width = target_width,
            height = target_height,
            guidance_scale=8,
            strength = 0.99,
            num_inference_steps=25
        ).images[0]

        result_cropped = result.resize((crop_w, crop_h), Image.Resampling.LANCZOS)

        final_result = image.copy()

        final_result.paste(result_cropped, (x1, y1))

        return final_result

    def remove_object(self, image_data):
        image, final_mask = self.get_brush_mask(image_data)

        if image is None or np.max(final_mask) == 0:
            return None, None

        mask_lama, _ = self.refine_mask(final_mask)

        image_pil = Image.fromarray(image)
        mask_lama_pil = Image.fromarray(mask_lama)

        lama_img = self.lama(image_pil, mask_lama_pil)

        self.flush_memory()

        if lama_img.size != image_pil.size:
            lama_img = lama_img.resize(image_pil.size, Image.Resampling.LANCZOS)

        return lama_img, None

    def replace_object(self, image_data, prompt):
        image, final_mask = self.get_brush_mask(image_data)

        if image is None or np.max(final_mask) == 0:
            return None, None

        if not prompt or prompt.strip() == "":
            auto_label = self.generate_auto_prompt(Image.fromarray(image), final_mask)

            final_prompt = self.enrich_prompt(f"a beautiful {auto_label}, natural lighting, high detail, same perspective as background")
        else:
            final_prompt = self.enrich_prompt(prompt)

        mask_lama, mask_sdxl, mask_blend = self.refine_mask(final_mask)

        image_pil = Image.fromarray(image)
        mask_lama_pil = Image.fromarray(mask_lama)
        mask_sdxl_pil = Image.fromarray(mask_sdxl)

        lama_img = self.lama(image_pil, mask_lama_pil)

        if lama_img.size != image_pil.size:
            lama_img = lama_img.resize(image_pil.size, Image.Resampling.LANCZOS)

        self.flush_memory()

        sdxl_img = self.stable_diffusion(lama_img, mask_sdxl_pil, final_prompt)

        final_img = self.hybrid_blend(lama_img, sdxl_img, final_mask, mask_blend)

        return lama_img, final_img

    def flush_memory(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def main():
    app = InpaintingApp()
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## DEMO STABLE DIFFUSION PURE BRUSH")

        with gr.Row():
            with gr.Column():
                img_input = gr.ImageEditor(label="Input Image", type="numpy", interactive=True)

                with gr.Accordion("Generation setting", open=True):
                    prompt = gr.Textbox(label="Prompt")

                with gr.Row():
                    remove_btn = gr.Button("Remove", variant="secondary")
                    replace_btn = gr.Button("Replace", variant="primary")

            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("LaMa image"):
                        lama_img = gr.Image(label="Lama Image", type="pil")
                    with gr.TabItem("Final image"):
                        final_img = gr.Image(label="Final Image", type="pil")

        remove_btn.click(
            fn=app.remove_object,
            inputs=[img_input],
            outputs=[lama_img, final_img]
        )

        replace_btn.click(
            fn=app.replace_object,
            inputs=[img_input, prompt],
            outputs=[lama_img, final_img]
        )

    demo.launch()

if __name__ == "__main__":
    main()

