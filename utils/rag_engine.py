"""
RAG Engine for AortaGPT.
Coordinates all components of the Retrieval-Augmented Generation system.
"""

import logging
import time
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Coordinates the RAG system components to generate recommendations.
    """
    
    def __init__(self, 
                 document_processor,
                 vector_store_manager,
                 clinvar_api,
                 openai_client,
                 km_visualizer):
        """
        Initialize the RAG Engine.
        
        Args:
            document_processor: Document processor for handling clinical documents
            vector_store_manager: Vector store manager for retrieval
            clinvar_api: ClinVar API client for variant information
            openai_client: OpenAI client for generation
            km_visualizer: KM Visualizer for survival curves
        """
        self.document_processor = document_processor
        self.vector_store_manager = vector_store_manager
        self.clinvar_api = clinvar_api
        self.openai_client = openai_client
        self.km_visualizer = km_visualizer
        
        # Track initialization state
        self.initialized = False
    
    def initialize_system(self, force_reindex: bool = False) -> bool:
        """
        Initialize the RAG system by loading documents and creating the vector store.
        
        Args:
            force_reindex: Whether to force reindexing of documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process documents if needed
            if force_reindex:
                logger.info("Forcing reindexing of documents")
                self.document_processor.process_all_documents()
            
            # Create or load vector store
            vector_store_success = self.vector_store_manager.create_vector_store(force_recreate=force_reindex)
            
            if not vector_store_success:
                logger.error("Failed to initialize vector store")
                return False
            
            self.initialized = True
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return False
    
    def process_patient_query(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a patient query and generate personalized recommendations.
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            Dictionary with recommendations and other outputs
        """
        # Start timing
        start_time = time.time()
        
        try:
            if not self.initialized:
                logger.error("RAG system not initialized")
                return {
                    "status": "error",
                    "message": "RAG system not initialized"
                }
            
            # 1. Fetch ClinVar data for the variant
            clinvar_data = None
            if patient_data.get("gene") and patient_data.get("variant"):
                logger.info(f"Fetching ClinVar data for {patient_data['gene']}:{patient_data['variant']}")
                clinvar_data = self.clinvar_api.search_variant(
                    gene=patient_data["gene"],
                    variant_str=patient_data["variant"]
                )
            
            # 2. Generate KM curve
            km_curve_data = None
            km_curve_fig = None
            
            if patient_data.get("gene") and patient_data.get("sex") and patient_data.get("age"):
                logger.info(f"Generating KM curve for {patient_data['gene']} ({patient_data['sex']}, age {patient_data['age']})")
                try:
                    km_curve_fig, km_curve_data = self.km_visualizer.generate_km_curve(
                        gene=patient_data["gene"],
                        sex=patient_data["sex"],
                        age=int(patient_data["age"]),
                        variant=patient_data.get("variant"),
                        clinvar_data=clinvar_data
                    )
                    
                    # Convert figure to base64 for display
                    km_curve_base64 = self.km_visualizer.fig_to_base64(km_curve_fig)
                    
                    # Close the figure to free up memory
                    plt.close(km_curve_fig)
                except Exception as e:
                    logger.error(f"Error generating KM curve: {str(e)}")
                    km_curve_base64 = ""
            else:
                logger.warning("Not enough data to generate KM curve")
                km_curve_base64 = ""
            
            # 3. Prepare context with RAG content
            logger.info("Building patient context with RAG retrieval")
            clinical_options = ["hypertension", "family_history", "symptoms", "prior_imaging", "prior_surgery"]
            context = self.vector_store_manager.build_patient_context(patient_data, clinical_options)
            
            # 4. Generate recommendations
            logger.info("Generating recommendations")
            recommendations = self.openai_client.generate_recommendations(context)
            
            # 5. Package results
            execution_time = time.time() - start_time
            
            result = {
                "status": "success",
                "recommendations": recommendations,
                "clinvar_data": clinvar_data,
                "km_curve": {
                    "base64": km_curve_base64,
                    "data": km_curve_data
                },
                "execution_time": execution_time
            }
            
            logger.info(f"Query processed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing patient query: {str(e)}")
            execution_time = time.time() - start_time
            
            return {
                "status": "error",
                "message": f"Error processing patient query: {str(e)}",
                "execution_time": execution_time
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dictionary with system status information
        """
        return {
            "initialized": self.initialized,
            "document_processor": {
                "data_folder": self.document_processor.data_folder,
                "processed_folder": self.document_processor.processed_folder
            },
            "vector_store": {
                "loaded": self.vector_store_manager.vector_store is not None
            }
        }