import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class SignatureProcessor:
    """
    Specialized processor for handling documents with signatures that might cover text.
    """
    
    def process_image_with_signatures(self, img):
        """
        Apply specialized processing for images with signatures.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Processed image with improved text visibility
        """
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Apply multiple thresholding techniques to handle different scenarios
            # 1. Otsu's thresholding for general cases
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 2. Adaptive thresholding for local variations (better for signatures)
            binary_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
            
            # 3. Another adaptive threshold with different parameters for text under signatures
            binary_adaptive2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
            
            # Combine results using bitwise operations
            combined = cv2.bitwise_or(cv2.bitwise_or(binary_otsu, binary_adaptive), binary_adaptive2)
            
            # Apply morphological operations to enhance text
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Remove small connected components (likely noise)
            contours, _ = cv2.findContours(255 - opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.ones_like(opening) * 255
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:  # Threshold for small components
                    cv2.drawContours(mask, [contour], -1, 0, -1)
            
            result = cv2.bitwise_and(opening, mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in signature processing: {str(e)}")
            # Return original image if processing fails
            return img
    
    def detect_signatures(self, img):
        """
        Detect regions that might contain signatures.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            List of bounding boxes of potential signature regions, confidence score
        """
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Apply blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply edge detection
            edges = cv2.Canny(blurred, 30, 150)
            
            # Dilate to connect edges
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find potential signatures
            signature_regions = []
            signature_confidence = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Typical signature size range
                if 500 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    # Typical signature aspect ratio
                    if 0.2 < aspect_ratio < 5:
                        # Calculate a confidence score based on characteristics
                        # Higher weight for medium-sized contours with moderate aspect ratios
                        size_score = min(1.0, area / 10000)
                        ratio_score = 1.0 - abs(aspect_ratio - 2.5) / 2.5
                        
                        # Calculate density (ratio of contour area to bounding rect area)
                        density = area / (w * h)
                        # Signatures tend to have lower density
                        density_score = 1.0 - min(1.0, density * 2)
                        
                        # Get region of interest
                        roi = gray[y:y+h, x:x+w]
                        if roi.size > 0:
                            # Calculate variance in the region (signatures have high variance)
                            variance = np.var(roi)
                            variance_score = min(1.0, variance / 2000)
                        else:
                            variance_score = 0.0
                            
                        # Combine scores with appropriate weights
                        confidence = (size_score * 0.3 + 
                                     ratio_score * 0.2 + 
                                     density_score * 0.3 + 
                                     variance_score * 0.2)
                        
                        signature_regions.append((x, y, w, h, confidence))
                        
                        # Update overall confidence if this is higher
                        signature_confidence = max(signature_confidence, confidence)
            
            # Sort by confidence (highest first)
            signature_regions.sort(key=lambda x: x[4], reverse=True)
            
            # Return the regions and overall confidence
            return signature_regions, signature_confidence
            
        except Exception as e:
            logger.error(f"Error detecting signatures: {str(e)}")
            return [], 0.0
    
    def remove_signature_from_image(self, img, signature_regions):
        """
        Process image to minimize the impact of signatures for better text extraction.
        
        Args:
            img: Input image as numpy array
            signature_regions: List of (x, y, w, h, confidence) tuples representing signature boxes
            
        Returns:
            Image with signatures processed to improve text extraction
        """
        try:
            # Make a copy to avoid modifying the original
            result = img.copy()
            
            # For each signature region
            for x, y, w, h, conf in signature_regions:
                if conf < 0.5:  # Skip low confidence regions
                    continue
                    
                # Get the region of interest (ROI)
                roi = result[y:y+h, x:x+w]
                
                # Apply special processing to this region
                if len(roi.shape) == 3:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = roi
                
                # Apply stronger thresholding to this region
                # This can help reveal text under lighter signatures
                _, binary = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
                
                # Apply morphological operations to enhance any text
                kernel = np.ones((1, 1), np.uint8)
                enhanced = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Place the enhanced region back
                if len(result.shape) == 3 and len(enhanced.shape) == 2:
                    # Convert back to BGR if needed
                    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    result[y:y+h, x:x+w] = enhanced_bgr
                else:
                    result[y:y+h, x:x+w] = enhanced
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing signatures: {str(e)}")
            return img 