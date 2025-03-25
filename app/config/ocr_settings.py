"""
OCR settings to configure OCR behavior and improve quality.
"""

class OCRSettings:
    # OCR Engine mode
    OEM_LSTM_ONLY = 1  # LSTM neural network only
    OEM_TESSERACT_LSTM_COMBINED = 3  # Default, based on what is available
    
    # Page segmentation modes
    PSM_AUTO = 3  # Fully automatic page segmentation, but no OSD
    PSM_SINGLE_BLOCK = 6  # Assume a single uniform block of text
    PSM_SINGLE_LINE = 7  # Treat the image as a single text line
    PSM_SPARSE_TEXT = 11  # Sparse text with OSD
    PSM_SPARSE_TEXT_OSD = 12  # Sparse text with OSD
    
    # Default settings
    DEFAULT_DPI = 300
    DEFAULT_LANG = "eng+spa"  # English + Spanish
    DEFAULT_OEM = OEM_LSTM_ONLY
    DEFAULT_PSM = PSM_AUTO
    
    # Preprocessing settings
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 15
    ADAPTIVE_THRESHOLD_CONSTANT = 8
    DENOISE_STRENGTH = 10
    
    # Table detection settings
    TABLE_MIN_CONFIDENCE = 0.45
    TABLE_MIN_AREA = 5000
    TABLE_MIN_ROWS = 2
    TABLE_MIN_COLS = 2
    
    # Special cases settings
    SIGNATURE_MODE = {
        "threshold_block_size": 11,  # Smaller block size for signatures
        "threshold_constant": 5,
        "denoise_strength": 7,
        "dilate_iterations": 1,
        "psm": PSM_SINGLE_BLOCK,  # Try to capture text as a single block
    }
    
    TABLE_MODE = {
        "threshold_block_size": 11,
        "threshold_constant": 2,
        "dilate_iterations": 1,
        "psm": PSM_SPARSE_TEXT,  # Better for detecting table cells
    } 