import io
import logging
from typing import BinaryIO, Union
import PyPDF2
import re

logger = logging.getLogger(__name__)

class PDFTypeDetector:
    """
    Enhanced utility class to detect if a PDF is scanned (image-based) or regular (contains text).
    """
    
    def is_scanned_pdf(self, pdf_file: Union[bytes, BinaryIO]) -> bool:
        """
        Detect if a PDF is scanned (image-based) or regular (contains text).
        Uses improved detection heuristics.
        
        Args:
            pdf_file: The PDF file as bytes or a file-like object
            
        Returns:
            True if the PDF is scanned, False otherwise
        """
        try:
            # Ensure we have a file-like object
            if isinstance(pdf_file, bytes):
                pdf_file = io.BytesIO(pdf_file)
            
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Check a sample of pages (up to 5)
            page_count = min(len(reader.pages), 5)
            
            # More sophisticated detection logic
            for page_num in range(page_count):
                page = reader.pages[page_num]
                text_content = page.extract_text()
                
                # Check for meaningful text content
                if text_content:
                    # Remove common OCR errors and noise
                    cleaned_text = re.sub(r'[\W_]+', ' ', text_content).strip()
                    words = cleaned_text.split()
                    
                    # If we have a significant number of words, it's likely not scanned
                    if len(words) > 15:
                        word_lengths = [len(word) for word in words]
                        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
                        
                        # If average word length is reasonable, probably real text
                        if 3 <= avg_word_length <= 10:
                            logger.info("PDF appears to be a regular document with text")
                            return False
            
            # If we reach here, either no text was found or text quality was poor
            logger.info("PDF appears to be a scanned document or has signatures obscuring text")
            return True
                
        except Exception as e:
            logger.error(f"Error detecting PDF type: {str(e)}")
            # Default to OCR if we can't determine the type
            return True 