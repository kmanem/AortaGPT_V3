"""
Vector Store Manager for AortaGPT - Simplified for just PDFs and CSVs.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional

# Import only what we need
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Simplified vector store manager focused only on PDFs and CSVs."""
    
    def __init__(self, 
                 data_folder: str = "data/raw/", 
                 processed_folder: str = "data/processed/",
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small"):
        """Initialize with bare essentials."""
        self.data_folder = data_folder
        self.processed_folder = processed_folder
        self.openai_api_key = openai_api_key
        
        # Create processed folder if needed
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        # Initialize embeddings with API key
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Cache for vector store
        self.vector_store = None
    
    def create_vector_store(self, force_recreate=False) -> bool:
        """
        Create a vector store from PDFs and CSVs only.
        """
        vector_store_path = os.path.join(self.processed_folder, "vector_store.pkl")
        
        # Check if vector store already exists
        if os.path.exists(vector_store_path) and not force_recreate:
            logger.info(f"Vector store already exists at {vector_store_path}")
            try:
                with open(vector_store_path, "rb") as f:
                    self.vector_store = pickle.load(f)
                return True
            except Exception as e:
                logger.error(f"Failed to load existing vector store: {str(e)}")
                # Continue with creation
        
        try:
            # Process only PDFs and CSVs
            documents = []
            
            # Check if data folder exists
            if not os.path.exists(self.data_folder):
                logger.warning(f"Data folder {self.data_folder} does not exist")
                return False
            
            # Get all files
            files = os.listdir(self.data_folder)
            
            # Process PDFs
            for filename in [f for f in files if f.lower().endswith('.pdf')]:
                try:
                    file_path = os.path.join(self.data_folder, filename)
                    logger.info(f"Loading PDF: {filename}")
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    
                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200
                    )
                    pdf_chunks = splitter.split_documents(pdf_docs)
                    
                    # Add basic metadata
                    for i, chunk in enumerate(pdf_chunks):
                        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                            chunk.metadata = {}
                        chunk.metadata["source"] = filename
                        chunk.metadata["chunk_id"] = i
                    
                    documents.extend(pdf_chunks)
                except Exception as e:
                    logger.error(f"Error processing PDF {filename}: {str(e)}")
            
            # Process CSVs
            for filename in [f for f in files if f.lower().endswith('.csv')]:
                try:
                    file_path = os.path.join(self.data_folder, filename)
                    logger.info(f"Loading CSV: {filename}")
                    loader = CSVLoader(file_path)
                    csv_docs = loader.load()
                    
                    # Split if needed
                    if csv_docs:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=100
                        )
                        csv_chunks = splitter.split_documents(csv_docs)
                        
                        # Add basic metadata
                        for i, chunk in enumerate(csv_chunks):
                            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                                chunk.metadata = {}
                            chunk.metadata["source"] = filename
                            chunk.metadata["chunk_id"] = i
                        
                        documents.extend(csv_chunks)
                except Exception as e:
                    logger.error(f"Error processing CSV {filename}: {str(e)}")
            
            # Skip all other file types (docx, pptx) to avoid NLTK issues
            logger.info(f"Loaded {len(documents)} document chunks from PDFs and CSVs")
            
            if not documents:
                logger.error("No documents loaded")
                return False
            
            # Create vector store
            logger.info("Creating FAISS vector store")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
            with open(vector_store_path, "wb") as f:
                pickle.dump(vector_store, f)
            
            # Update instance variable
            self.vector_store = vector_store
            
            logger.info("Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False
    
    def get_vector_store(self):
        """Get the vector store, loading it if necessary."""
        if self.vector_store is None:
            vector_store_path = os.path.join(self.processed_folder, "vector_store.pkl")
            if os.path.exists(vector_store_path):
                try:
                    with open(vector_store_path, "rb") as f:
                        self.vector_store = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading vector store: {str(e)}")
        
        return self.vector_store
    
    def retrieve_relevant_context(self, query: str, n_results: int = 5) -> str:
        """Simple retrieval of relevant context."""
        vector_store = self.get_vector_store()
        if vector_store is None:
            return "No vector store available."
        
        try:
            results = vector_store.similarity_search(query, k=n_results)
            
            # Format into simple context
            context = ""
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content
                context += f"[SOURCE {i+1}] {source}\n{content}\n\n"
            
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
    
    def retrieve_by_section(self, patient_data: Dict[str, Any]) -> str:
        """Retrieve context for patient data."""
        # Extract basics
        gene = patient_data.get("gene", "")
        variant = patient_data.get("variant", "")
        
        # Create a simple query
        query = f"{gene} {variant} thoracic aortic disease recommendations"
        
        # Just do a simple retrieval
        return self.retrieve_relevant_context(query, n_results=10)
    
    def build_patient_context(self, patient_data: Dict[str, Any], clinical_options: List[str] = None) -> str:
        """Build a simple context with patient data and retrieved info."""
        # Get relevant content
        rag_context = self.retrieve_by_section(patient_data)
        
        # Build the context
        context = """You are AortaGPT, a clinical decision support tool for HTAD.
Provide evidence-based recommendations based on the information below.
"""
        
        # Add retrieved context
        if rag_context:
            context += "\n## RETRIEVED GUIDELINE CONTEXT:\n\n"
            context += rag_context
        
        # Add patient info
        context += "\n## PATIENT INFORMATION:\n\n"
        context += f"Age: {patient_data.get('age', 'Unknown')} years\n"
        context += f"Sex: {patient_data.get('sex', 'Unknown')}\n"
        context += f"Gene: {patient_data.get('gene', 'Unknown')}\n"
        if patient_data.get('variant'):
            context += f"Variant: {patient_data.get('variant')}\n"
        context += f"Aortic Root Diameter: {patient_data.get('root_diam', 'Unknown')} mm\n"
        context += f"Ascending Aorta Diameter: {patient_data.get('asc_diam', 'Unknown')} mm\n"
        
        # Add any other data
        if patient_data.get('history'):
            context += f"\nClinical History: {patient_data.get('history')}\n"
        if patient_data.get('meds'):
            context += f"\nMedications: {patient_data.get('meds')}\n"
        
        return context