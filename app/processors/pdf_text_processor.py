import io
import logging
from typing import BinaryIO, Union
import PyPDF2

logger = logging.getLogger(__name__)

class PDFTextProcessor:
    """
    Processor for extracting text from regular (non-scanned) PDFs.
    """
    
    def process(self, pdf_file: Union[bytes, BinaryIO]) -> str:
        """
        Extract text from a regular PDF file.
        
        Args:
            pdf_file: The PDF file as bytes or a file-like object
            
        Returns:
            The extracted text as a string
        """
        try:
            # Ensure we have a file-like object
            if isinstance(pdf_file, bytes):
                pdf_file = io.BytesIO(pdf_file)
            
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text: {str(e)}") 