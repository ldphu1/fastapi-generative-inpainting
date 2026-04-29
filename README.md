# fastapi-generative-inpainting
This project provides a powerful generative inpainting system that allows users to remove objects or replace objects within images. By combining state-of-the-art Deep Learning models including LaMa, Stable Diffusion XL (SDXL), and BLIP, the project offers both a RESTful API and an interactive Web UI.

# Key Features
* **Object Removal**: Utilizes the (`Simple LaMa`) model to seamlessly erase unwanted elements from images without leaving artifacts.
* **Object Replacement**: Leverages (`Stable Diffusion XL Inpainting`) (SDXL 1.0) to generate new objects that blend perfectly with the existing environment.
* **Auto-Prompting**: If a prompt is not provided during replacement, the system uses the (`BLIP`) model to analyze the selected area and automatically generate a contextually appropriate prompt.
*  **Advanced Blending**: Implements a hybrid approach between Poisson Blending and standard Alpha Blending to ensure smooth edges and consistent lighting between generated content and the background.
# Project Structure
* (`main.py`): The entry point for the FastAPI server, providing /api/remove and /api/replace endpoints.
* (`ai_engine.py`): The core AI engine containing the (`InpaintingApp class`), which handles model loading, mask refinement, and image processing logic.
* (`demo/gradio_app.py`): A standalone Gradio application for a web-based demo featuring a brush tool for manual mask selection.
# Usage
1. Running the API Server (FastAPI)
   
   Start the Uvicorn server to host the REST API:

   ```bash
   python main.py
   ```
