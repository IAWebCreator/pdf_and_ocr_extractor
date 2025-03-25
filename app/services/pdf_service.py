import io
import logging
from typing import BinaryIO, Union
import tempfile
import os

from app.processors.pdf_text_processor import PDFTextProcessor
from app.processors.pdf_ocr_processor import PDFOCRProcessor
from app.utils.pdf_type_detector import PDFTypeDetector
from app.processors.signature_processor import SignatureProcessor
from app.config.ocr_settings import OCRSettings

logger = logging.getLogger(__name__)

class PDFService:
    """
    Optimized service for handling PDF processing and text extraction.
    """
    
    def __init__(self):
        self.text_processor = PDFTextProcessor()
        self.ocr_processor = PDFOCRProcessor()
        self.type_detector = PDFTypeDetector()
        self.signature_processor = SignatureProcessor()
        self.ocr_settings = OCRSettings()
        
        # Cache for processed files
        self._cache = {}
        self._cache_size = 10  # Maximum number of items to cache
    
    def extract_text(self, pdf_content: Union[bytes, BinaryIO]) -> str:
        """
        Extract text from a PDF file with optimal performance.
        
        Args:
            pdf_content: The PDF file content as bytes or a file-like object
            
        Returns:
            The extracted text as a string
        """
        # Check if we've processed this file before
        content_hash = None
        try:
            # Only hash the first 10KB for performance
            if isinstance(pdf_content, bytes):
                content_hash = hash(pdf_content[:10240])
            else:
                content_hash = hash(pdf_content.read(10240))
                pdf_content.seek(0)  # Reset position
                
            if content_hash in self._cache:
                return self._cache[content_hash]
        except:
            # If hashing fails, just continue with processing
            pass
        
        # Ensure we have a bytes-like object
        if not isinstance(pdf_content, bytes):
            pdf_content = pdf_content.read()
        
        # Use a temporary file for large PDFs
        # This can improve performance with large files
        if len(pdf_content) > 10 * 1024 * 1024:  # 10MB
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_path = temp_file.name
            
            try:
                # Detect if the PDF is scanned or regular
                with open(temp_path, 'rb') as f:
                    is_scanned = self.type_detector.is_scanned_pdf(f)
                
                # Process based on type
                with open(temp_path, 'rb') as f:
                    if is_scanned:
                        result = self.ocr_processor.process(f)
                    else:
                        result = self.text_processor.process(f)
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
        else:
            # For smaller files, process in memory
            pdf_file = io.BytesIO(pdf_content)
            
            # Detect if the PDF is scanned or regular
            is_scanned = self.type_detector.is_scanned_pdf(pdf_file)
            
            # Reset file pointer
            pdf_file.seek(0)
            
            # Process based on type
            if is_scanned:
                result = self.ocr_processor.process(pdf_file)
            else:
                result = self.text_processor.process(pdf_file)
        
        # Cache the result
        if content_hash is not None:
            # Limit cache size
            if len(self._cache) >= self._cache_size:
                # Remove oldest item
                self._cache.pop(next(iter(self._cache)))
            self._cache[content_hash] = result
            
        return result 