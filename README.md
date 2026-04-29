# fastapi-generative-inpainting
This project provides a powerful generative inpainting system that allows users to remove objects or replace objects within images. By combining state-of-the-art Deep Learning models including LaMa, Stable Diffusion XL (SDXL), and BLIP, the project offers both a RESTful API and an interactive Web UI.

# Key Features
* **Object Removal**: Utilizes the (`Simple LaMa`) model to seamlessly erase unwanted elements from images without leaving artifacts.
* **Object Replacement**: Leverages (`Stable Diffusion XL Inpainting`) (SDXL 1.0) to generate new objects that blend perfectly with the existing environment.
* **Auto-Prompting**: If a prompt is not provided during replacement, the system uses the (`BLIP`) model to analyze the selected area and automatically generate a contextually appropriate prompt.
*  **Advanced Blending**: Implements a hybrid approach between Poisson Blending and standard Alpha Blending to ensure smooth edges and consistent lighting between generated content and the background.
# Project Structure
* (`main.py`): The entry point for the FastAPI server, providing (`/api/remove`) and (`/api/replace`) endpoints.
* (`ai_engine.py`): The core AI engine containing the (`InpaintingApp class`), which handles model loading, mask refinement, and image processing logic.
* (`demo/gradio_app.py`): A standalone Gradio application for a web-based demo featuring a brush tool for manual mask selection.

# System Architecture
   ```bash
   %% Remove Pipeline
    Action -->|Remove Object| Refine1[Refine Mask]
    Refine1 --> LaMa1[Simple LaMa Model]
    LaMa1 --> Output[Final Image]
    
    %% Replace Pipeline
    Action -->|Replace Object| CheckPrompt{Prompt Provided?}
    CheckPrompt -->|No| BLIP[BLIP: Auto-Generate Prompt] --> Enrich[Enrich Prompt]
    CheckPrompt -->|Yes| Enrich[Enrich Prompt]
    
    Enrich --> Refine2[Refine Masks]
    Refine2 --> LaMa2[LaMa: Erase base object]
    LaMa2 --> SDXL[SDXL: Generate new object]
    SDXL --> Blend[Hybrid Blending]
    Blend --> Output
   ```

# Usage
1. Running the API Server (FastAPI)
   
   Start the Uvicorn server to host the REST API:

   ```bash
   python main.py
   ```
   Note: The first run will take some time to download the model weights (SDXL, BLIP, etc.) from HuggingFace.

   * API Base URL: (`http://localhost:8000`)

   * Swagger Documentation: (`http://localhost:8000/docs`)

   Primary Endpoints:

   * (`POST /api/remove`): Accepts image and mask files via form-data. Returns the inpainted result as a Base64 string.

   * (`POST /api/replace`): Accepts image, mask, and an optional prompt. Returns the generated result as a Base64 string.
     
2. Running the Web UI Demo (Gradio)
   
   For an interactive experience with a brush tool, run:
   ```bash
   python demo/gradio_app.py
   ```
   Access the interface at the local URL provided in your terminal (typically (`http://127.0.0.1:7860`)).
   
   
