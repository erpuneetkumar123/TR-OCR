# TR-OCR
FastAPI OCR service using Microsoft TrOCR with a fine-tuned model. Images are preprocessed (grayscale, equalization, thresholding, resize, padding) before text generation. Supports GPU, simple corrections, and a /predict endpoint to return extracted handwritten text.
