from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import logging
from app.services.pdf_service import PDFService
from app.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PDF Text Extraction API")

# Initialize settings
settings = Settings()

# Initialize PDF service
pdf_service = PDFService()

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from a PDF file.
    Automatically detects if the PDF is a regular PDF or a scanned document.
    """
    try:
        logger.info(f"Received PDF file: {file.filename}")
        
        # Read the file content
        content = await file.read()
        
        # Process the PDF
        text = pdf_service.extract_text(content)
        
        return JSONResponse(content={"text": text})
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process PDF: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port) 