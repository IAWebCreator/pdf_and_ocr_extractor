# PDF Text Extraction API

A Python backend service that extracts text from PDF files. The service automatically detects whether the PDF contains actual text or is a scanned document, and uses the appropriate extraction method.

## Features

- Extracts text from regular PDFs using PyPDF2
- Extracts text from scanned PDFs using OCR (pytesseract)
- Automatically detects PDF type
- Simple REST API interface

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:

- For Ubuntu/Debian:
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- For macOS:
  ```bash
  brew install tesseract
  ```
- For Windows: Download and install from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

3. Create a `.env` file (optional): 