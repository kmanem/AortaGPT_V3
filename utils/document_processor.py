"""
Simple document processor for AortaGPT.
"""

import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Simplified document processor."""
    
    def __init__(self, 
                 data_folder: str = "data/raw/", 
                 processed_folder: str = "data/processed/",
                 openai_api_key: Optional[str] = None):
        """Initialize simple processor."""
        self.data_folder = data_folder
        self.processed_folder = processed_folder
        self.openai_api_key = openai_api_key
        
        # Create processed folder if needed
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
    
    def process_all_documents(self) -> bool:
        """Empty method for compatibility."""
        logger.info("Document processing skipped - using simplified approach")
        return True