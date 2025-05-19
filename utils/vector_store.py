"""
Vector Store Manager for AortaGPT.
Handles document embeddings and retrieval for the RAG system.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# For document processing
from langchain.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader, 
    CSVLoader, 
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector database for retrieving relevant clinical guidelines."""
    
    def __init__(self, 
                 data_folder: str = "data/raw/", 
                 processed_folder: str = "data/processed/",
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the Vector Store Manager.
        
        Args:
            data_folder: Directory containing the raw document files
            processed_folder: Directory to store processed vector stores
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
        """
        self.data_folder = data_folder
        self.processed_folder = processed_folder
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Create processed folder if it doesn't exist
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        # Vector store cache
        self.vector_store = None
        
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
            files = os.listdir(self.data_folder)
            
            for filename in files:
                file_path = os.path.join(self.data_folder, filename)
                
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
    
    def create_vector_store(self, force_recreate=False) -> bool:
        """
        Create a FAISS vector store from documents.
        
        Args:
            force_recreate: If True, recreate the vector store even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        vector_store_path = os.path.join(self.processed_folder, "vector_store.pkl")
        
        # Check if vector store already exists
        if os.path.exists(vector_store_path) and not force_recreate:
            logger.info(f"Vector store already exists at {vector_store_path}")
            return True
        
        try:
            # Load and process documents
            documents = self.load_documents()
            if not documents:
                logger.error("No documents loaded")
                return False
            
            # Create vector store
            logger.info("Creating FAISS vector store")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
            logger.info(f"Saving vector store to {vector_store_path}")
            with open(vector_store_path, "wb") as f:
                pickle.dump(vector_store, f)
            
            # Update instance variable
            self.vector_store = vector_store
            
            logger.info("Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False
    
    def load_vector_store(self) -> bool:
        """
        Load the FAISS vector store from disk.
        
        Returns:
            True if successful, False otherwise
        """
        vector_store_path = os.path.join(self.processed_folder, "vector_store.pkl")
        
        if not os.path.exists(vector_store_path):
            logger.error(f"Vector store does not exist at {vector_store_path}")
            return False
        
        try:
            logger.info(f"Loading vector store from {vector_store_path}")
            with open(vector_store_path, "rb") as f:
                self.vector_store = pickle.load(f)
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_vector_store(self):
        """
        Get the current vector store, loading it if necessary.
        
        Returns:
            FAISS vector store or None if unavailable
        """
        if self.vector_store is None:
            self.load_vector_store()
        
        return self.vector_store
    
    def retrieve_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Formatted context string
        """
        vector_store = self.get_vector_store()
        if vector_store is None:
            logger.error("Vector store not available")
            return ""
        
        try:
            logger.info(f"Retrieving context for query: {query}")
            results = vector_store.similarity_search(query, k=n_results)
            
            # Format results
            context = self._format_context(results)
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
    
    def _format_context(self, results) -> str:
        """Format retrieved documents into a context string."""
        if not results:
            return ""
        
        context = ""
        for i, doc in enumerate(results):
            # Extract source and content
            source = doc.metadata.get("source", "Unknown")
            category = doc.metadata.get("category", "other")
            content = doc.page_content
            
            # Format based on category
            if category == "guidelines":
                context += f"[GUIDELINE {i+1}] Source: {source}\n{content}\n\n"
            elif category == "genetics":
                context += f"[GENETIC DATA {i+1}] Source: {source}\n{content}\n\n"
            else:
                context += f"[DOCUMENT {i+1}] Source: {source}\n{content}\n\n"
        
        return context
    
    def retrieve_by_category(self, query: str, category: str, n_results: int = 3) -> str:
        """
        Retrieve documents from a specific category.
        
        Args:
            query: Search query
            category: Category to search in
            n_results: Number of results to return
            
        Returns:
            Formatted context string
        """
        vector_store = self.get_vector_store()
        if vector_store is None:
            return ""
        
        try:
            # Create metadata filter
            filter_dict = {"category": category}
            
            # Search with metadata filter
            results = vector_store.similarity_search(
                query, 
                k=n_results,
                filter=filter_dict
            )
            
            # Format results
            context = self._format_context(results)
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving by category: {str(e)}")
            return ""
    
    def retrieve_by_section(self, patient_data: Dict[str, Any]) -> str:
        """
        Retrieve relevant content for each clinical recommendation section,
        with improved handling of unstructured "other" information.
        """
        vector_store = self.get_vector_store()
        if vector_store is None:
            return ""
        
        # Extract key patient information
        gene = patient_data.get("gene", "")
        variant = patient_data.get("variant", "")
        root_diam = patient_data.get("root_diam", 0)
        asc_diam = patient_data.get("asc_diam", 0)
        sex = patient_data.get("sex", "")
        age = patient_data.get("age", 0)
        
        # Get other relevant details
        other_details = patient_data.get("history", "")
        
        # Create gene-specific query component
        gene_query = f"{gene} thoracic aortic disease"
        if variant:
            gene_query += f" {variant}"
        
        # Create measurement-specific query component
        measurement_query = ""
        max_diam = max(root_diam, asc_diam) if root_diam and asc_diam else root_diam or asc_diam or 0
        if max_diam > 45:
            measurement_query = "surgical intervention threshold aortic aneurysm"
        elif max_diam > 40:
            measurement_query = "surveillance monitoring aortic dilation"
        else:
            measurement_query = "early monitoring aortic disease prevention"
        
        # Define standard section-specific queries
        section_queries = {
            "initial_workup": f"{gene_query} initial diagnostic workup evaluation",
            "risk_stratification": f"{gene_query} risk factors aortic dissection aneurysm {measurement_query}",
            "surgical_thresholds": f"{gene_query} surgical intervention threshold {measurement_query}",
            "imaging": f"{gene_query} imaging surveillance frequency intervals MRI CT echocardiogram",
            "lifestyle": f"{gene_query} physical activity restrictions exercise guidelines",
            "pregnancy": f"{gene_query} pregnancy management guidelines risks {sex}",
            "genetic": f"{gene_query} family screening genetic testing inheritance pattern",
            "blood_pressure": f"{gene_query} blood pressure management targets medications",
            "medication": f"{gene_query} beta blocker ARB medication recommendations",
            "variant": f"{gene} {variant} pathogenicity clinical significance phenotype correlation"
        }
        
        # Additional demographic-specific queries
        if int(age) < 18:
            section_queries["pediatric"] = f"{gene_query} pediatric management children adolescent"
        
        if sex.lower() == "female" and int(age) < 50:
            section_queries["reproductive"] = f"{gene_query} pregnancy planning contraception management"
        
        # ---- IMPROVED HANDLING OF "OTHER" INFORMATION ----
        
        # 1. Break the other details into meaningful chunks
        if other_details:
            # Split into sentences or segments
            segments = [s.strip() for s in other_details.replace('\n', '. ').split('.') if s.strip()]
            
            # Create a specific query for each significant segment
            for i, segment in enumerate(segments):
                if len(segment) > 15:  # Only consider reasonably substantial segments
                    # Clean the segment
                    clean_segment = segment.strip()
                    # Create a specific query for this segment
                    section_queries[f"other_detail_{i}"] = f"{gene_query} {clean_segment}"
            
            # 2. Also create synthetic "concept" queries from the other details
            try:
                # Common clinical concepts to check for in other details
                concepts = [
                    "family history", "sudden death", "cardiovascular", "exercise", "sports",
                    "pregnancy", "smoking", "hypertension", "surgery", "imaging", "symptoms",
                    "medication", "pain", "syncope", "fatigue", "lifestyle", "mental health",
                    "trauma", "injury", "stroke", "heart attack", "intervention", "emergency",
                    "monitoring", "surveillance", "complications", "drug", "substance",
                    "alcohol", "diet", "nutrition", "sleep", "quality of life", "genetic",
                    "screening", "testing", "travel", "occupation", "work", "activity"
                ]
                
                # Check which concepts are mentioned in the other details
                mentioned_concepts = [c for c in concepts if c.lower() in other_details.lower()]
                
                # Create concept-specific queries
                for concept in mentioned_concepts:
                    section_queries[f"concept_{concept}"] = f"{gene_query} {concept} management recommendations"
                    
            except Exception as e:
                logger.error(f"Error processing concepts in other details: {str(e)}")
        
        # Retrieve content for each section
        all_chunks = []
        
        # First retrieve the standard sections (ensures core recommendations are included)
        standard_sections = list(section_queries.keys())[:10]  # First 10 are standard
        for section in standard_sections:
            query = section_queries[section]
            try:
                # Use category filtering when appropriate
                if section == "initial_workup" or section == "imaging":
                    results = vector_store.similarity_search(
                        query, 
                        k=2,
                        filter={"category": "diagnostic"}
                    )
                elif section == "surgical_thresholds":
                    results = vector_store.similarity_search(
                        query, 
                        k=3,
                        filter={"category": "guidelines"}
                    )
                elif section == "variant":
                    results = vector_store.similarity_search(
                        query, 
                        k=2,
                        filter={"category": "genetics"}
                    )
                else:
                    # General search across all documents
                    results = vector_store.similarity_search(query, k=2)
                
                all_chunks.extend(results)
                logger.info(f"Retrieved {len(results)} chunks for section: {section}")
                
            except Exception as e:
                logger.error(f"Error retrieving chunks for section {section}: {str(e)}")
        
        # Then retrieve content for non-standard sections (other details, concepts)
        non_standard_sections = [s for s in section_queries.keys() if s not in standard_sections]
        
        # Limit the number of additional queries to prevent context overflow
        max_additional_sections = 10
        if len(non_standard_sections) > max_additional_sections:
            # Prioritize shorter, more specific queries
            non_standard_sections = sorted(non_standard_sections, 
                                          key=lambda s: len(section_queries[s]))[:max_additional_sections]
        
        for section in non_standard_sections:
            query = section_queries[section]
            try:
                # For other detail sections, search across all categories
                results = vector_store.similarity_search(query, k=2)
                all_chunks.extend(results)
                logger.info(f"Retrieved {len(results)} chunks for section: {section}")
                
            except Exception as e:
                logger.error(f"Error retrieving chunks for section {section}: {str(e)}")
        
        # 3. ONE FINAL SEMANTIC SEARCH WITH FULL OTHER DETAILS
        # This catches anything missed by the segmented approach
        if other_details and len(other_details) > 20:
            try:
                # Clean up the text
                full_query = f"{gene_query} patient with {other_details[:300]}"
                results = vector_store.similarity_search(full_query, k=5)
                all_chunks.extend(results)
                logger.info(f"Retrieved {len(results)} chunks for full other details query")
                
            except Exception as e:
                logger.error(f"Error retrieving chunks for full other details: {str(e)}")
        
        # Remove duplicates
        unique_chunks = []
        seen_content = set()
        for chunk in all_chunks:
            content = chunk.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_chunks.append(chunk)
        
        # Format context
        context = self._format_context(unique_chunks)
        logger.info(f"Retrieved {len(unique_chunks)} unique chunks across all sections")
        
        return context
    
    def build_patient_context(self, patient_data: Dict[str, Any], clinical_options: List[str]) -> str:
        """
        Build a complete context for a patient query, including RAG retrieval.
        
        Args:
            patient_data: Dictionary with patient data
            clinical_options: List of clinical history options
            
        Returns:
            Complete context string for GPT prompt
        """
        # Start with system instructions
        context = """You are AortaGPT, a specialized clinical decision support tool for Heritable Thoracic Aortic Disease (HTAD). 
Your purpose is to provide evidence-based recommendations for patients with genetic variants associated with HTAD.
Base your recommendations ONLY on the provided guideline context and patient information below.
"""
        
        # Get relevant guideline context from RAG system
        rag_context = self.retrieve_by_section(patient_data)
        if rag_context:
            context += "\n## RETRIEVED GUIDELINE CONTEXT:\n\n"
            context += rag_context
        
        # Add patient information
        context += "\n## PATIENT INFORMATION:\n\n"
        
        # Demographics
        context += f"Age: {patient_data.get('age', 'Unknown')} years\n"
        context += f"Sex: {patient_data.get('sex', 'Unknown')}\n\n"
        
        # Genetic data
        context += "Genetic Profile:\n"
        context += f"Gene: {patient_data.get('gene', 'Unknown')}\n"
        if patient_data.get('variant'):
            context += f"Variant: {patient_data.get('variant')}\n"
        
        # Variant information
        if patient_data.get('selected_variant_info'):
            context += "\nClinVar Information:\n"
            variant_info = patient_data.get('selected_variant_info', {})
            if 'clinical_significance' in variant_info:
                context += f"Clinical Significance: {variant_info['clinical_significance']}\n"
            if 'review_status' in variant_info:
                context += f"Review Status: {variant_info['review_status']}\n"
        
        # Aortic measurements
        context += "\nAortic Measurements:\n"
        context += f"Aortic Root Diameter: {patient_data.get('root_diam', 'Unknown')} mm\n"
        context += f"Ascending Aorta Diameter: {patient_data.get('asc_diam', 'Unknown')} mm\n"
        if patient_data.get('zscore'):
            context += f"Z-score: {patient_data.get('zscore')}\n"
        
        # Medications
        if patient_data.get('meds'):
            context += "\nCurrent Medications:\n"
            for med in patient_data.get('meds', "").split(","):
                if med.strip():
                    context += f"- {med.strip()}\n"
        
        # Clinical history
        if patient_data.get('history'):
            context += "\nClinical History:\n"
            context += patient_data.get('history')
        
        # Freeform
        if patient_data.get('freeform'):
            context += "\nAdditional Clinical Details:\n"
            context += patient_data.get('freeform')
        
        return context