"""
AortaGPT: Clinical Decision Support Tool for HTAD

A Streamlit application that generates guideline-based recommendations for
patients with Heritable Thoracic Aortic Disease (HTAD).
"""

import streamlit as st
import os
import time
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import logging
import sys

# Import custom modules
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.clinvar_api import ClinVarAPI
from utils.openai_client import OpenAIClient
from utils.km_visualizer import KMVisualizer
from utils.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aortagpt.log")
    ]
)
logger = logging.getLogger(__name__)

# ---- CONFIGURATION ----
DATA_FOLDER = "data/raw/"
PROCESSED_FOLDER = "data/processed/"

# Function to initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

# Function to initialize RAG engine
def initialize_rag_engine():
    """Initialize the RAG engine and its components."""
    try:
        # Get OpenAI API key and assistant ID from Streamlit secrets
        api_key = st.secrets.get("openai_api_key", None)
        assistant_id = st.secrets.get("assistant_id", None)
        
        if not api_key:
            st.error("OpenAI API key not found in secrets. Please add it to your Streamlit secrets.toml file.")
            return False
        
        # Initialize components
        document_processor = DocumentProcessor(
            data_folder=DATA_FOLDER,
            processed_folder=PROCESSED_FOLDER,
            openai_api_key=api_key
        )
        
        vector_store_manager = VectorStoreManager(
            processed_folder=PROCESSED_FOLDER,
            openai_api_key=api_key
        )
        
        clinvar_api = ClinVarAPI(
            email="your-email@example.com",  # Optional but recommended
            tool="AortaGPT"
        )
        
        openai_client = OpenAIClient(
            api_key=api_key,
            assistant_id=assistant_id,
            model="gpt-4",  # Fallback model if assistant_id not available
            max_tokens=4000
        )
        
        km_visualizer = KMVisualizer(
            mac_data_path=os.path.join(DATA_FOLDER, "mac_data.docx")  # Update with actual path
        )
        
        # Create RAG engine
        rag_engine = RAGEngine(
            document_processor=document_processor,
            vector_store_manager=vector_store_manager,
            clinvar_api=clinvar_api,
            openai_client=openai_client,
            km_visualizer=km_visualizer
        )
        
        # Initialize system
        with st.spinner("Initializing AortaGPT..."):
            success = rag_engine.initialize_system(force_reindex=False)
            
        if success:
            st.session_state.rag_engine = rag_engine
            st.session_state.initialized = True
            return True
        else:
            st.error("Failed to initialize AortaGPT. Check logs for details.")
            return False
            
    except Exception as e:
        st.error(f"Error initializing AortaGPT: {str(e)}")
        logger.error(f"Error initializing AortaGPT: {str(e)}")
        return False

# Function to process patient data
def process_patient_data(patient_data):
    """Process patient data and generate recommendations."""
    if not st.session_state.initialized or not st.session_state.rag_engine:
        st.error("AortaGPT not initialized. Please refresh the page.")
        return None
    
    try:
        # Process query
        with st.spinner("Generating patient-specific recommendations..."):
            result = st.session_state.rag_engine.process_patient_query(patient_data)
        
        st.session_state.last_result = result
        return result
    except Exception as e:
        st.error(f"Error processing patient data: {str(e)}")
        logger.error(f"Error processing patient data: {str(e)}")
        return None

# ---- UI STYLING ----
def setup_page_styling():
    """Setup page styling with CSS."""
    st.set_page_config(page_title="AortaGPT: Clinical Decision Support Tool", 
                      layout="wide", 
                      page_icon=":anatomical_heart:")
    
    st.markdown("""
    <style>
    body { background-color: #0e1117; }
    .section-header { font-size: 20px; font-weight: bold; margin-top: 1em; color: #ffffff; }
    .highlight-box { background-color: #3b2b2b; border-radius: 12px; padding: 10px; color: white; font-weight: 500; }
    .risk-high { background-color: #3b2b2b; }
    .risk-moderate { background-color: #3b3b2b; }
    .risk-low { background-color: #2b3b2b; }
    .stSubheader { font-size: 1.2em; font-weight: 600; margin-top: 1em; }
    .citation { font-size: 0.9em; color: #9e9e9e; }
    .recommendation-section { 
        background-color: #1e2130; 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 15px;
    }
    .subheader {
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 0.5em;
        color: #4fbba9;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to create sidebar inputs
def create_sidebar_inputs():
    """Create sidebar inputs for patient data."""
    st.sidebar.title("Patient Input Parameters")

    # Demographics
    st.sidebar.subheader("Demographics")
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Other"])

    # Genetic Data
    st.sidebar.subheader("Genetic Data")
    gene = st.sidebar.selectbox("Gene", ["FBN1", "ACTA2", "TGFBR1", "TGFBR2", "SMAD3", "COL3A1", "MYH11", "Other"])
    variant = st.sidebar.text_input("Variant", placeholder="e.g., NM_001613.4:c.536G>A")

    # Aortic measurements
    st.sidebar.subheader("Aortic Measurements")
    root_diam = st.sidebar.number_input("Aortic Root Diameter (mm)", min_value=0.0, max_value=100.0, value=36.0, step=0.1)
    asc_diam = st.sidebar.number_input("Ascending Aorta Diameter (mm)", min_value=0.0, max_value=100.0, value=36.0, step=0.1)
    zscore = st.sidebar.number_input("Z-score (optional)", value=0.0, step=0.1)

    # Clinical history and meds
    st.sidebar.subheader("Clinical Information")
    history = st.sidebar.text_area("Clinical History", placeholder="e.g., Sister died of dissection at 24, no known hypertension, smokes occasionally...")
    meds = st.sidebar.text_area("Current Medications", placeholder="e.g., Beta-blocker, ARB, multivitamin...")

    # Freeform entry (optional alternative to above)
    freeform = st.sidebar.text_area("Freeform Clinical Note (Optional)", placeholder="Paste note or summary here to auto-populate...")

    # Advanced options
    st.sidebar.subheader("Advanced Options")
    force_reindex = st.sidebar.checkbox("Force Reindex Documents", value=False)
    if force_reindex:
        st.sidebar.warning("This will rebuild the vector database, which may take some time.")

    # Submit button
    submitted = st.sidebar.button("Generate Recommendations")

    # Combine inputs
    patient_data = {
        "age": age,
        "sex": sex,
        "gene": gene,
        "variant": variant,
        "root_diam": root_diam,
        "asc_diam": asc_diam,
        "zscore": zscore,
        "history": history,
        "meds": meds,
        "freeform": freeform
    }

    return patient_data, submitted, force_reindex

# Function to display recommendations
def display_recommendations(result):
    """Display recommendations from RAG engine."""
    if result is None or result.get("status") != "success":
        st.error("Failed to generate recommendations.")
        return
    
    # Extract components
    recommendations = result.get("recommendations", {})
    km_curve = result.get("km_curve", {})
    clinvar_data = result.get("clinvar_data", None)
    execution_time = result.get("execution_time", 0)
    
    # Display header
    st.markdown("# 🩵 AortaGPT: Clinical Decision Support Tool")
    st.markdown(f"<p>Recommendations generated in {execution_time:.2f} seconds</p>", unsafe_allow_html=True)
    
    # Create columns for main content
    col1, col2 = st.columns([3, 1])
    
    # Display recommendations in main column
    with col1:
        # Display each section
        sections = [
            ("Initial Workup", "initial_workup"),
            ("Risk Stratification", "risk_stratification"),
            ("Surgical Thresholds", "surgical_thresholds"),
            ("Imaging Surveillance", "imaging_surveillance"),
            ("Lifestyle & Activity Guidelines", "lifestyle_activity"),
            ("Pregnancy/Peripartum", "pregnancy"),
            ("Genetic Counseling", "genetic_counseling"),
            ("Blood Pressure Recommendations", "blood_pressure"),
            ("Medication Management", "medication"),
            ("Gene/Variant Interpretation", "variant_interpretation")
        ]
        
        for display_name, key in sections:
            content = recommendations.get(display_name, "")
            if content:
                st.markdown(f"""
                <div class="recommendation-section">
                    <div class="subheader">{display_name}</div>
                    {content}
                </div>
                """, unsafe_allow_html=True)
    
    # Display sidebar content
    with col2:
        # Display KM curve
        st.markdown("### Kaplan-Meier Curve")
        st.image(f"data:image/png;base64,{km_curve.get('base64', '')}", use_column_width=True)
        
        # Display KM curve data
        st.markdown("#### Survival Probability")
        km_data = km_curve.get("data", {})
        survival = km_data.get("survival_at_age", 0)
        st.markdown(f"**At age {km_data.get('age', 0)}:** {survival:.2%}")
        
        # Display ClinVar data if available
        if clinvar_data and clinvar_data.get("found", False):
            st.markdown("### ClinVar Data")
            for entry in clinvar_data.get("clinvar_data", []):
                st.markdown(f"""
                **Clinical Significance:** {entry.get('clinical_significance', 'Unknown')}  
                **Review Status:** {entry.get('review_status', 'Unknown')}
                """)

# Main application function
def main():
    """Main application entry point."""
    # Setup page
    setup_page_styling()
    
    # Initialize session state
    init_session_state()
    
    # Create sidebar inputs
    patient_data, submitted, force_reindex = create_sidebar_inputs()
    
    # Initialize RAG engine if not already initialized or force reindex
    if not st.session_state.initialized or force_reindex:
        if not initialize_rag_engine():
            st.stop()
    
    # Process patient data if submitted
    if submitted:
        result = process_patient_data(patient_data)
        if result:
            display_recommendations(result)
    # Display previous result if available
    elif st.session_state.last_result:
        display_recommendations(st.session_state.last_result)
    # Display welcome message if no result
    else:
        st.markdown("# 🩵 AortaGPT: Clinical Decision Support Tool")
        st.markdown("""
        ## Welcome to AortaGPT
        
        This tool provides guideline-based, gene- and variant-specific recommendations for patients with Heritable Thoracic Aortic Disease (HTAD).
        
        ### How to use:
        1. Enter patient demographics, genetic data, and clinical information in the sidebar
        2. Click "Generate Recommendations" to get personalized clinical guidance
        3. View structured recommendations across multiple categories
        
        ### Data Sources:
        - 2022 ACC/AHA Guidelines on Aortic Disease
        - GeneReviews entries
        - MAC Consortium Data
        - ClinVar/ClinGen variant databases
        - Other curated reference materials
        
        All recommendations are evidence-based and include citations to the source materials.
        """)

if __name__ == "__main__":
    main()