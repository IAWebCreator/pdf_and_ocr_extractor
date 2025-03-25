import cv2
import numpy as np
import pandas as pd
import logging
import pytesseract
from PIL import Image
from tabulate import tabulate
import re

logger = logging.getLogger(__name__)

class TableProcessor:
    """
    Processor for detecting and extracting tables from PDF images.
    Uses computer vision techniques to identify table structures.
    """
    
    def detect_tables(self, img):
        """
        Detect table regions in an image.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            List of bounding boxes for detected tables and confidence score
        """
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
                
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine horizontal and vertical lines
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Find contours in the combined image
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to identify table regions
            table_regions = []
            table_confidence = 0.0
            
            min_table_area = 5000  # Minimum area for a table
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_table_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Tables usually have reasonable aspect ratios
                    if 0.2 < aspect_ratio < 5:
                        # Check for line density within the region
                        roi = table_mask[y:y+h, x:x+w]
                        density = np.sum(roi) / (255 * roi.size)
                        
                        # Check for grid-like structure
                        line_confidence = self._evaluate_grid_structure(roi)
                        
                        # Calculate overall confidence
                        confidence = (0.4 * density + 0.6 * line_confidence)
                        
                        if confidence > 0.3:  # Threshold for table detection
                            table_regions.append((x, y, w, h, confidence))
                            table_confidence = max(table_confidence, confidence)
            
            # For overlapping tables, keep only the one with highest confidence
            table_regions = self._merge_overlapping_regions(table_regions)
            
            # Sort by confidence
            table_regions.sort(key=lambda x: x[4], reverse=True)
            
            return table_regions, table_confidence
            
        except Exception as e:
            logger.error(f"Error detecting tables: {str(e)}")
            return [], 0.0
    
    def _evaluate_grid_structure(self, table_mask):
        """
        Evaluate how grid-like a potential table region is.
        
        Args:
            table_mask: Binary image of table lines
            
        Returns:
            Confidence score for grid structure (0-1)
        """
        try:
            # Count horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, vertical_kernel)
            
            # Count horizontal line segments
            h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h_line_count = len(h_contours)
            
            # Count vertical line segments
            v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            v_line_count = len(v_contours)
            
            # If there are reasonable numbers of both horizontal and vertical lines
            if h_line_count >= 2 and v_line_count >= 2:
                # Calculate intersections between horizontal and vertical lines
                intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
                intersection_count = cv2.countNonZero(intersections)
                
                # Estimate expected intersections vs. actual
                expected_intersections = h_line_count * v_line_count
                if expected_intersections > 0:
                    intersection_ratio = min(1.0, intersection_count / (expected_intersections * 10))
                    return 0.5 + (0.5 * intersection_ratio)
                    
            # If we don't have enough lines in both directions
            if h_line_count >= 3 or v_line_count >= 3:
                return 0.5  # Could be a simple table with few divisions
                
            return 0.0  # Not a table
            
        except Exception as e:
            logger.error(f"Error evaluating grid structure: {str(e)}")
            return 0.0
    
    def _merge_overlapping_regions(self, regions):
        """
        Merge overlapping table regions, keeping the one with higher confidence.
        """
        if not regions:
            return []
            
        # Sort by confidence (highest first)
        sorted_regions = sorted(regions, key=lambda r: r[4], reverse=True)
        
        # List to store non-overlapping regions
        merged_regions = []
        
        for region in sorted_regions:
            x1, y1, w1, h1, conf1 = region
            
            # Check if this region overlaps significantly with any in merged_regions
            overlap = False
            for i, merged_region in enumerate(merged_regions):
                x2, y2, w2, h2, conf2 = merged_region
                
                # Calculate overlap area
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                # Calculate smaller area of the two regions
                area1 = w1 * h1
                area2 = w2 * h2
                smaller_area = min(area1, area2)
                
                # If overlap is significant (>50% of smaller region)
                if smaller_area > 0 and overlap_area / smaller_area > 0.5:
                    overlap = True
                    break
                    
            if not overlap:
                merged_regions.append(region)
                
        return merged_regions
    
    def extract_table_structure(self, img, table_regions):
        """
        Extract the structure of detected tables.
        
        Args:
            img: Input image as numpy array
            table_regions: List of (x,y,w,h,conf) tuples for table regions
            
        Returns:
            List of extracted tables as dataframes and text representations
        """
        tables = []
        
        for i, (x, y, w, h, conf) in enumerate(table_regions):
            # Extract the table region
            table_img = img[y:y+h, x:x+w].copy()
            
            # Process the table image to enhance grid lines
            processed_table = self._preprocess_table_image(table_img)
            
            # Get grid line positions
            h_lines, v_lines = self._get_table_grid_lines(processed_table)
            
            # If we couldn't find grid lines, use a different approach
            if len(h_lines) < 2 or len(v_lines) < 2:
                # Fall back to direct OCR with heuristic cell detection
                table_text, table_df = self._extract_table_direct_ocr(table_img)
            else:
                # Extract cell contents using the grid structure
                table_text, table_df = self._extract_table_by_grid(table_img, h_lines, v_lines)
            
            tables.append({
                'region': (x, y, w, h),
                'confidence': conf,
                'dataframe': table_df,
                'text': table_text
            })
            
        return tables
    
    def _preprocess_table_image(self, img):
        """
        Preprocess table image to enhance grid lines.
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect broken lines
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        return dilated
    
    def _get_table_grid_lines(self, processed_table):
        """
        Detect horizontal and vertical grid lines in a table.
        
        Returns:
            Tuple of (horizontal_lines, vertical_lines) where each is a list of positions
        """
        height, width = processed_table.shape
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 3, 1))
        horizontal_lines_img = cv2.morphologyEx(processed_table, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # Find horizontal line positions
        h_projection = np.sum(horizontal_lines_img, axis=1)
        h_lines = []
        
        # Threshold for line detection
        threshold = np.max(h_projection) * 0.3
        for i in range(1, height - 1):
            if h_projection[i] > threshold and h_projection[i-1] < h_projection[i] > h_projection[i+1]:
                h_lines.append(i)
        
        # Add top and bottom lines if not detected
        if len(h_lines) > 0 and h_lines[0] > 10:
            h_lines.insert(0, 0)
        if len(h_lines) > 0 and h_lines[-1] < height - 10:
            h_lines.append(height - 1)
            
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 3))
        vertical_lines_img = cv2.morphologyEx(processed_table, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        # Find vertical line positions
        v_projection = np.sum(vertical_lines_img, axis=0)
        v_lines = []
        
        # Threshold for line detection
        threshold = np.max(v_projection) * 0.3
        for i in range(1, width - 1):
            if v_projection[i] > threshold and v_projection[i-1] < v_projection[i] > v_projection[i+1]:
                v_lines.append(i)
        
        # Add left and right lines if not detected
        if len(v_lines) > 0 and v_lines[0] > 10:
            v_lines.insert(0, 0)
        if len(v_lines) > 0 and v_lines[-1] < width - 10:
            v_lines.append(width - 1)
            
        return h_lines, v_lines
    
    def _extract_table_by_grid(self, table_img, h_lines, v_lines):
        """
        Extract table cell contents using detected grid lines.
        """
        # Create empty DataFrame to store results
        num_rows = len(h_lines) - 1
        num_cols = len(v_lines) - 1
        
        if num_rows <= 0 or num_cols <= 0:
            return "", pd.DataFrame()
            
        # Extract all cells
        cells_text = []
        for i in range(num_rows):
            row_cells = []
            for j in range(num_cols):
                # Get cell coordinates
                y1, y2 = h_lines[i], h_lines[i+1]
                x1, x2 = v_lines[j], v_lines[j+1]
                
                # Ensure minimum dimensions and valid box
                if y2 <= y1 or x2 <= x1 or y2 - y1 < 5 or x2 - x1 < 5:
                    row_cells.append("")
                    continue
                
                # Extract cell image with padding
                padding = 2
                y_start = max(0, y1 - padding)
                y_end = min(table_img.shape[0], y2 + padding)
                x_start = max(0, x1 - padding)
                x_end = min(table_img.shape[1], x2 + padding)
                
                cell_img = table_img[y_start:y_end, x_start:x_end]
                
                # OCR the cell
                if cell_img.size > 0:
                    # Apply preprocessing specific for text in cells
                    cell_img_processed = self._preprocess_cell_image(cell_img)
                    
                    # Extract text using pytesseract
                    cell_text = pytesseract.image_to_string(
                        cell_img_processed, 
                        config='--psm 6 --oem 1'  # Single line mode for better results in cells
                    ).strip()
                    
                    row_cells.append(cell_text)
                else:
                    row_cells.append("")
                    
            cells_text.append(row_cells)
            
        # Create DataFrame (first row as header if it looks like a header)
        if len(cells_text) > 1:
            # Check if first row looks like a header
            first_row = cells_text[0]
            is_header = True
            
            # Heuristic: header rows often have shorter text than data rows
            for i, cell in enumerate(first_row):
                if not cell or any(c.isdigit() for c in cell):
                    # Headers typically don't have empty cells or numeric data
                    is_header = False
                    break
            
            if is_header:
                headers = first_row
                data = cells_text[1:]
                df = pd.DataFrame(data, columns=headers)
            else:
                df = pd.DataFrame(cells_text)
        else:
            df = pd.DataFrame(cells_text)
        
        # Generate text representation
        table_text = tabulate(df, headers='firstrow' if len(df) > 0 else [], tablefmt='grid')
        
        return table_text, df
    
    def _extract_table_direct_ocr(self, table_img):
        """
        Fallback method for tables without clear grid lines.
        Uses OCR directly and attempts to structure the result.
        """
        # Apply preprocessing
        if len(table_img.shape) == 3:
            gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = table_img.copy()
            
        # Threshold to enhance text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR with table detection mode
        ocr_text = pytesseract.image_to_string(
            binary,
            config='--psm 6 --oem 1'  # Assume single uniform block of text
        )
        
        # Split into lines
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        
        # Try to detect columns based on consistent spacing
        data = []
        for line in lines:
            # Detect spaces that might represent column separators
            spaces = []
            for i, char in enumerate(line):
                if char.isspace() and (i == 0 or not line[i-1].isspace()):
                    spaces.append(i)
            
            # If no spaces found, treat as single cell
            if not spaces:
                data.append([line])
                continue
                
            # Split by spaces with larger runs indicating column boundaries
            cells = []
            prev_idx = 0
            for idx in spaces:
                if idx - prev_idx > 1:  # Avoid empty cells from adjacent spaces
                    cells.append(line[prev_idx:idx].strip())
                prev_idx = idx
            
            # Add the last cell
            if prev_idx < len(line):
                cells.append(line[prev_idx:].strip())
                
            data.append(cells)
        
        # Create a DataFrame (handling rows with different numbers of cells)
        max_cols = max(len(row) for row in data) if data else 0
        for row in data:
            while len(row) < max_cols:
                row.append("")
                
        df = pd.DataFrame(data)
        
        # Generate text representation
        table_text = tabulate(df, headers='firstrow' if len(df) > 0 else [], tablefmt='grid')
        
        return table_text, df
    
    def _preprocess_cell_image(self, cell_img):
        """
        Apply preprocessing specific for table cells to improve OCR.
        """
        # Convert to grayscale if needed
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()
            
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Increase contrast
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def detect_tables_optimized(self, img):
        """
        Faster table detection focusing on performance.
        """
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Resize large images for faster processing
            h, w = gray.shape
            if max(h, w) > 2500:
                scale = 2500 / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
            # Apply simpler thresholding for speed
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Use faster morphological operations
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            
            # Faster implementation of horizontal and vertical line detection
            horizontal = cv2.erode(binary, kernel_h, iterations=1)
            horizontal = cv2.dilate(horizontal, kernel_h, iterations=1)
            
            vertical = cv2.erode(binary, kernel_v, iterations=1) 
            vertical = cv2.dilate(vertical, kernel_v, iterations=1)
            
            # Combine lines
            table_mask = cv2.bitwise_or(horizontal, vertical)
            
            # Find contours - use CHAIN_APPROX_SIMPLE for better performance
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Faster filtering of contours
            table_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area > 5000 and 0.2 < w/h < 5:
                    # Simple confidence calculation based on area
                    confidence = min(1.0, area / 50000)
                    table_regions.append((x, y, w, h, confidence))
            
            # Simple non-maximum suppression
            filtered_regions = []
            if table_regions:
                # Sort by confidence
                table_regions.sort(key=lambda x: x[4], reverse=True)
                
                # Keep track of which regions to discard
                discard = set()
                
                # For each region
                for i, region1 in enumerate(table_regions):
                    if i in discard:
                        continue
                        
                    x1, y1, w1, h1, conf1 = region1
                    filtered_regions.append(region1)
                    
                    # Compare with all other regions
                    for j, region2 in enumerate(table_regions[i+1:], i+1):
                        if j in discard:
                            continue
                            
                        x2, y2, w2, h2, conf2 = region2
                        
                        # Check if regions overlap significantly
                        if (x1 < x2+w2 and x1+w1 > x2 and 
                            y1 < y2+h2 and y1+h1 > y2):
                            # Calculate overlap area
                            intersection_width = min(x1+w1, x2+w2) - max(x1, x2)
                            intersection_height = min(y1+h1, y2+h2) - max(y1, y2)
                            overlap_area = intersection_width * intersection_height
                            
                            # If significant overlap, discard the lower confidence region
                            min_area = min(w1*h1, w2*h2)
                            if overlap_area > 0.7 * min_area:
                                discard.add(j)
            
            # Calculate overall confidence
            table_confidence = max([r[4] for r in filtered_regions]) if filtered_regions else 0.0
            
            return filtered_regions, table_confidence
            
        except Exception as e:
            logger.error(f"Error detecting tables: {str(e)}")
            return [], 0.0

    def extract_table_structure_optimized(self, img, table_regions):
        """
        Optimize table extraction for performance.
        """
        # Similar to existing method but with performance optimizations
        # Only extract actual data without creating visualizations
        tables = []
        
        for i, (x, y, w, h, conf) in enumerate(table_regions):
            try:
                # Extract the table region
                table_img = img[y:y+h, x:x+w].copy()
                
                # Process with a simpler, faster method
                _, binary = cv2.threshold(
                    cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY) if len(table_img.shape) == 3 else table_img,
                    0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                
                # Direct OCR with table heuristics
                text = pytesseract.image_to_string(
                    binary,
                    config='--psm 6 --oem 1'
                )
                
                # Simple table structure extraction
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Create dataframe for structured data
                import pandas as pd
                data = []
                for line in lines:
                    cells = re.split(r'\s{2,}', line)
                    data.append(cells)
                
                # Ensure all rows have the same number of columns
                max_cols = max([len(row) for row in data]) if data else 0
                for row in data:
                    while len(row) < max_cols:
                        row.append("")
                
                df = pd.DataFrame(data)
                
                tables.append({
                    'region': (x, y, w, h),
                    'confidence': conf,
                    'dataframe': df,
                    'text': text
                })
            except Exception as e:
                logger.warning(f"Error extracting table {i}: {str(e)}")
        
        return tables 