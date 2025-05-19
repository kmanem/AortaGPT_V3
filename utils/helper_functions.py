"""
Helper functions for AortaGPT application.
"""

import requests
import time
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_all_params(config):
    """
    Apply configuration parameters from GPT-parsed data to session state.
    """
    import streamlit as st
    
    # Apply each parameter to session state
    for key, value in config.items():
        if key != "custom_gene":  # Handle custom gene separately
            st.session_state[key] = value
    
    # Special handling for custom gene
    if config.get("gene") == "Other" and config.get("custom_gene"):
        st.session_state["gene"] = config["custom_gene"]

def fetch_clinvar_variants(gene):
    """
    Fetch variants for a specific gene from ClinVar.
    This is a simplified version that returns a basic list.
    In production, would call the actual ClinVar API.
    
    Args:
        gene: Gene symbol (e.g., FBN1)
    
    Returns:
        List of variant strings
    """
    # For demo purposes, return a small set of example variants
    # In production, this would make API calls to ClinVar
    example_variants = {
        "FBN1": [
            "NM_000138.4:c.5788+5G>A", 
            "NM_000138.4:c.7339G>A", 
            "NM_000138.4:c.1148-1G>A",
            "Enter custom variant"
        ],
        "TGFBR1": [
            "NM_004612.4:c.1459C>T", 
            "NM_004612.4:c.1460G>A", 
            "NM_004612.4:c.1460+1G>A",
            "Enter custom variant"
        ],
        "TGFBR2": [
            "NM_003242.6:c.1561T>C", 
            "NM_003242.6:c.1582C>T", 
            "NM_003242.6:c.1609C>T",
            "Enter custom variant"
        ],
        "ACTA2": [
            "NM_001613.4:c.536G>A", 
            "NM_001613.4:c.772C>T", 
            "NM_001613.4:c.115C>T",
            "Enter custom variant"
        ],
        "SMAD3": [
            "NM_005902.4:c.584A>G", 
            "NM_005902.4:c.788C>T", 
            "NM_005902.4:c.1170G>A",
            "Enter custom variant"
        ],
        "MYH11": [
            "NM_001040114.2:c.5273G>A", 
            "NM_001040114.2:c.5273G>T", 
            "NM_001040114.2:c.5274G>T",
            "Enter custom variant"
        ],
        "COL3A1": [
            "NM_000090.3:c.1662G>A", 
            "NM_000090.3:c.3221G>A", 
            "NM_000090.3:c.1714G>A",
            "Enter custom variant"
        ]
    }
    
    # Return example variants or default list
    return example_variants.get(gene, ["Enter custom variant"])

def filter_variants(variants, search_query):
    """
    Filter variants based on search query.
    
    Args:
        variants: List of variant strings
        search_query: Query string to filter by
        
    Returns:
        Filtered list of variants
    """
    if not search_query:
        return variants
    
    # Always include "Enter custom variant" option
    filtered = ["Enter custom variant"]
    
    # Add variants that match the search query
    for variant in variants:
        if variant != "Enter custom variant" and search_query.lower() in variant.lower():
            filtered.append(variant)
    
    return filtered

def fetch_variant_details(variant):
    """
    Fetch detailed information about a variant.
    This is a simplified version that returns mock data.
    In production, would call the actual ClinVar API.
    
    Args:
        variant: Variant string
        
    Returns:
        Dictionary with variant details
    """
    # For demo purposes, return mock data
    # In production, this would make API calls to ClinVar
    
    if variant == "Enter custom variant":
        return None
    
    # Generate mock data based on variant string patterns
    if "5788+5G>A" in variant:
        return {
            "clinical_significance": "Likely pathogenic",
            "review_status": "criteria provided, multiple submitters, no conflicts",
            "last_updated": "2023-06-15"
        }
    elif "1148-1G>A" in variant:
        return {
            "clinical_significance": "Pathogenic",
            "review_status": "reviewed by expert panel",
            "last_updated": "2023-01-22"
        }
    elif "536G>A" in variant:
        return {
            "clinical_significance": "Pathogenic",
            "review_status": "criteria provided, single submitter",
            "last_updated": "2022-11-30"
        }
    elif "G>A" in variant:
        return {
            "clinical_significance": "Likely pathogenic",
            "review_status": "criteria provided, single submitter",
            "last_updated": "2023-04-10"
        }
    elif "C>T" in variant:
        return {
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter",
            "last_updated": "2022-09-05"
        }
    else:
        return {
            "clinical_significance": "Uncertain significance",
            "review_status": "no assertion criteria provided",
            "last_updated": "2022-12-18"
        }

def build_patient_context(session_state, clinical_options):
    """
    Build a context string from patient data in session state.
    
    Args:
        session_state: Streamlit session state
        clinical_options: List of clinical history options
        
    Returns:
        Context string
    """
    context = """You are AortaGPT, a specialized clinical decision support assistant for Heritable Thoracic Aortic Disease (HTAD). 
Your purpose is to provide evidence-based recommendations for patients with genetic variants associated with HTAD.
Base your recommendations ONLY on curated aortic clinical data including ACC/AHA guidelines, GeneReviews, MAC registry, ClinVar/ClinGen, and related sources.
"""
    
    # Add patient information
    context += "\n## PATIENT INFORMATION:\n\n"
    
    # Demographics
    context += f"Age: {session_state.get('age', 'Unknown')} years\n"
    context += f"Sex: {session_state.get('sex', 'Unknown')}\n\n"
    
    # Genetic data
    context += "Genetic Profile:\n"
    context += f"Gene: {session_state.get('gene', 'Unknown')}\n"
    if session_state.get('variant'):
        context += f"Variant: {session_state.get('variant')}\n"
    
    # Variant information
    if session_state.get('selected_variant_info'):
        context += "\nClinVar Information:\n"
        variant_info = session_state.get('selected_variant_info', {})
        if 'clinical_significance' in variant_info:
            context += f"Clinical Significance: {variant_info['clinical_significance']}\n"
        if 'review_status' in variant_info:
            context += f"Review Status: {variant_info['review_status']}\n"
    
    # Aortic measurements
    context += "\nAortic Measurements:\n"
    context += f"Aortic Root Diameter: {session_state.get('root_diameter', 'Unknown')} mm\n"
    context += f"Ascending Aorta Diameter: {session_state.get('ascending_diameter', 'Unknown')} mm\n"
    if session_state.get('z_score'):
        context += f"Z-score: {session_state.get('z_score')}\n"
    
    # Medications
    if session_state.get('meds'):
        context += "\nCurrent Medications:\n"
        for med in session_state.get('meds', []):
            context += f"- {med}\n"
    
    # Clinical history
    clinical_history = []
    for option in clinical_options:
        if session_state.get(option, False):
            clinical_history.append(option)
    
    if clinical_history:
        context += "\nClinical History:\n"
        for item in clinical_history:
            context += f"- {item}\n"
    
    # Other relevant details
    if session_state.get('other_relevant_details'):
        context += "\nOther Relevant Details:\n"
        context += session_state.get('other_relevant_details')
    
    return context