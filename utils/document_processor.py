"""
Document Processor for AortaGPT.
Handles loading and preprocessing clinical documents for the vector store.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import re

# For document processing
from langchain.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader, 
    CSVLoader, 
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes and prepares documents for vector storage."""
    
    def __init__(self, 
                 data_folder: str = "data/raw/", 
                 processed_folder: str = "data/processed/",
                 openai_api_key: Optional[str] = None):
        """
        Initialize the Document Processor.
        
        Args:
            data_folder: Directory containing the raw document files
            processed_folder: Directory to store processed files
            openai_api_key: OpenAI API key (for potential future embedding)
        """
        self.data_folder = data_folder
        self.processed_folder = processed_folder
        self.openai_api_key = openai_api_key
        
        # Create processed folder if it doesn't exist
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        # Document metadata for better retrieval
        self.document_categories = {
            "guidelines": ["ACC_AHA_2022", "European_Hypertension_2024"],
            "genetics": ["GeneReviews", "ClinGen", "MAC_Consortium"],
            "diagnostic": ["HTAD_Diagnostic_Pathway"],
            "presentations": ["Dallas_Aorta_2025"],
            "data": ["GenomicMedicineGuida"]
        }
    
    def load_documents(self) -> List[Any]:
        """
        Load all documents from the data folder.
        
        Returns:
            List of loaded document objects
        """
        logger.info(f"Loading documents from {self.data_folder}")
        documents = []
        
        try:
            # Get all files in the data folder
            if not os.path.exists(self.data_folder):
                logger.warning(f"Data folder {self.data_folder} does not exist")
                return []
                
            files = os.listdir(self.data_folder)
            
            for filename in files:
                file_path = os.path.join(self.data_folder, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Determine file type and use appropriate loader
                if filename.lower().endswith('.pdf'):
                    logger.info(f"Loading PDF: {filename}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(self._process_document(loader, filename, "pdf"))
                    
                elif filename.lower().endswith(('.docx', '.doc')):
                    logger.info(f"Loading Word document: {filename}")
                    loader = UnstructuredWordDocumentLoader(file_path)
                    documents.extend(self._process_document(loader, filename, "word"))
                    
                elif filename.lower().endswith('.csv'):
                    logger.info(f"Loading CSV: {filename}")
                    loader = CSVLoader(file_path)
                    documents.extend(self._process_document(loader, filename, "csv"))
                    
                elif filename.lower().endswith(('.pptx', '.ppt')):
                    logger.info(f"Loading PowerPoint: {filename}")
                    loader = UnstructuredPowerPointLoader(file_path)
                    documents.extend(self._process_document(loader, filename, "powerpoint"))
                    
                else:
                    logger.warning(f"Unsupported file format: {filename}")
            
            logger.info(f"Loaded {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []
    
    def _process_document(self, loader, filename, doc_type):
        """Process a document with the appropriate text splitter."""
        try:
            # Load the document
            doc = loader.load()
            
            # Choose appropriate chunking strategy based on document type
            if doc_type == "pdf" or doc_type == "word":
                # For guidelines and text-heavy documents, use larger chunks with more overlap
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=250,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            elif doc_type == "csv":
                # For structured data, use smaller chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
            else:
                # Default splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200
                )
            
            # Split the document
            chunks = splitter.split_documents(doc)
            
            # Add metadata to chunks
            self._add_metadata(chunks, filename)
            
            # Preprocess the content
            self._preprocess_content(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            return []
    
    def _add_metadata(self, chunks, filename):
        """Add useful metadata to document chunks."""
        # Determine document category
        category = "other"
        for cat, file_patterns in self.document_categories.items():
            if any(pattern.lower() in filename.lower() for pattern in file_patterns):
                category = cat
                break
        
        # Extract gene information if present in filename
        gene_markers = ["FBN1", "ACTA2", "TGFBR", "SMAD3", "COL3A1", "MYH11"]
        genes = [gene for gene in gene_markers if gene.lower() in filename.lower()]
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            # Ensure metadata exists
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            # Add common metadata
            chunk.metadata["source"] = filename
            chunk.metadata["category"] = category
            chunk.metadata["chunk_id"] = i
            
            # Add gene information if available
            if genes:
                chunk.metadata["genes"] = genes
            
            # Add date information if available
            if "2022" in filename:
                chunk.metadata["year"] = 2022
            elif "2024" in filename:
                chunk.metadata["year"] = 2024
            elif "2025" in filename:
                chunk.metadata["year"] = 2025
    
    def _preprocess_content(self, chunks):
        """Preprocess the content of document chunks."""
        for chunk in chunks:
            # Clean up text
            text = chunk.page_content
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove page numbers and headers/footers (simple pattern)
            text = re.sub(r'Page \d+ of \d+', '', text)
            
            # Clean up special characters
            text = text.replace('•', '- ')
            
            # Update the chunk's text
            chunk.page_content = text.strip()
    
    def process_all_documents(self) -> bool:
        """
        Process all documents and prepare them for vector storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            documents = self.load_documents()
            if not documents:
                logger.warning("No documents were loaded for processing")
                return False
            
            logger.info(f"Successfully processed {len(documents)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False