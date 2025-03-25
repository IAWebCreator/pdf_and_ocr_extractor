import io
import logging
import tempfile
from typing import BinaryIO, Union
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import re
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class PDFOCRProcessor:
    """
    Optimized processor for extracting text from scanned PDFs with improved accuracy and speed.
    """
    
    def __init__(self):
        from app.processors.signature_processor import SignatureProcessor
        from app.processors.table_processor import TableProcessor
        from app.config.ocr_settings import OCRSettings
        self.signature_processor = SignatureProcessor()
        self.table_processor = TableProcessor()
        self.ocr_settings = OCRSettings()
        
        # Set optimal OCR parameters
        self.ocr_config = '--oem 1 --psm 3 -l eng+spa --dpi 300'
        
        # Dictionary for common OCR error corrections
        self.corrections = {
            # Common OCR errors and their corrections
            'l': 'I',  # Lowercase 'L' to uppercase 'I' when isolated
            '0': 'O',  # Zero to capital O in certain contexts
            'rn': 'm',  # 'rn' is often misrecognized as 'm'
            # Add more as needed based on your documents
        }
        
        # Maximum number of workers for parallel processing
        self.max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    def process(self, pdf_file: Union[bytes, BinaryIO]) -> str:
        """
        Extract text from a scanned PDF file using optimized OCR processing.
        
        Args:
            pdf_file: The PDF file as bytes or a file-like object
            
        Returns:
            The extracted text as a string
        """
        try:
            # Ensure we have bytes
            if not isinstance(pdf_file, bytes):
                pdf_content = pdf_file.read()
            else:
                pdf_content = pdf_file
            
            # Convert PDF pages to images with optimized settings
            images = convert_from_bytes(
                pdf_content, 
                dpi=300,  
                thread_count=self.max_workers,  # Use multiple threads for conversion
                grayscale=True  # Convert to grayscale immediately to save memory
            )
            
            # Process pages in parallel for better performance
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all page processing tasks
                future_to_page = {
                    executor.submit(self._process_page, i, image): (i, image) 
                    for i, image in enumerate(images)
                }
                
                # Collect results in page order
                results = []
                for future in as_completed(future_to_page):
                    i, _ = future_to_page[future]
                    try:
                        page_text = future.result()
                        results.append((i, page_text))
                    except Exception as e:
                        logger.error(f"Error processing page {i+1}: {str(e)}")
                        results.append((i, f"ERROR: Failed to process page {i+1}"))
            
            # Sort results by page number and join
            results.sort(key=lambda x: x[0])
            full_text = "\n\n".join(text for _, text in results)
            
            # Apply post-processing for common OCR errors
            processed_text = self._post_process_text(full_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error OCR processing PDF: {str(e)}")
            raise Exception(f"Failed to extract text with OCR: {str(e)}")
    
    def _process_page(self, page_num, image):
        """
        Process a single page with optimized steps.
        """
        logger.info(f"OCR processing page {page_num+1}")
        
        # Convert PIL Image to numpy array
        img_np = np.array(image)
        
        # Process the image with optimized settings
        processed_img = self._preprocess_image_optimized(img_np)
        
        # Run OCR with optimized settings
        page_text = pytesseract.image_to_string(
            processed_img,
            config=self.ocr_config
        )
        
        # Add page marker
        return f"--- PAGE {page_num+1} ---\n\n{page_text}"
    
    def _preprocess_image_optimized(self, img):
        """
        Optimized preprocessing to improve OCR quality and speed.
        """
        # Convert to grayscale if needed (our images should already be grayscale from pdf2image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Resize large images to improve performance
        h, w = gray.shape
        if max(h, w) > 3000:
            scale = 3000 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Apply bilateral filter to preserve edges while removing noise
        # This is slower but gives better quality than gaussian blur
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding to better separate text from background
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Auto-rotate if needed for better OCR results
        auto_rotated = self._auto_rotate(cleaned)
        
        return auto_rotated
    
    def _auto_rotate(self, img):
        """
        Auto-rotate the image if needed using text orientation detection.
        """
        try:
            # Use Tesseract to detect orientation
            osd = pytesseract.image_to_osd(img)
            angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
            
            # Only rotate if necessary
            if angle == 0:
                return img
                
            # Get image dimensions
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            
            # Rotate the image to correct orientation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        except Exception as e:
            # If orientation detection fails, return the original image
            logger.warning(f"Auto-rotation failed: {str(e)}")
            return img
    
    def _post_process_text(self, text):
        """
        Apply post-processing to improve text quality.
        """
        # Fix common OCR errors
        for error, correction in self.corrections.items():
            # Only replace if the error is isolated (surrounded by spaces or punctuation)
            text = re.sub(rf'(?<=[.,;:!?\s]){re.escape(error)}(?=[.,;:!?\s])', correction, text)
        
        # Fix broken paragraphs (lines ending with hyphen)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common number formatting issues
        text = re.sub(r'([0-9])\.([0-9])', r'\1.\2', text)  # Fix decimal points
        
        return text 