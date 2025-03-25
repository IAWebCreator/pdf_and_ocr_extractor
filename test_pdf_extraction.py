import sys
import os
import logging
import time
from app.services.pdf_service import PDFService

# Configure logging to be minimal
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pdf_extraction(file_path):
    """
    Simple test function that extracts text from a PDF and saves it to a file.
    Optimized for speed and accuracy.
    
    Args:
        file_path: Path to the PDF file to test
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        # Initialize service
        pdf_service = PDFService()
        
        # Start timer
        start_time = time.time()
        
        # Read the file
        with open(file_path, 'rb') as f:
            pdf_content = f.read()
        
        # Extract text
        text = pdf_service.extract_text(pdf_content)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Save output to file
        output_file = f"{os.path.splitext(file_path)[0]}_extracted_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Print minimal information
        print(f"Extraction completed in {process_time:.2f} seconds")
        print(f"Extracted text saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the path provided by the user
        file_path = r"C:\Users\Usuario1\Downloads\10. HOJA DE VIDA - CANINOS EXPLOSIVOS.pdf"
    
    test_pdf_extraction(file_path) 