"""
ClinVar API integration for AortaGPT.
Handles querying ClinVar for variant information.
"""

import requests
import re
import time
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinVarAPI:
    def __init__(self, email=None, tool="AortaGPT"):
        """
        Initialize the ClinVar API client.
        
        Args:
            email: Email for NCBI E-utilities (optional but recommended)
            tool: Tool name for NCBI E-utilities
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email
        self.tool = tool
        self.cache = {}  # Simple cache to avoid repeated API calls
    
    def search_variant(self, gene: str, variant_str: str) -> Dict[str, Any]:
        """
        Search for variant information in ClinVar.
        
        Args:
            gene: Gene name (e.g., FBN1)
            variant_str: Variant string (e.g., NM_001613.4:c.536G>A)
            
        Returns:
            Dictionary with variant information
        """
        # Check cache first
        cache_key = f"{gene}:{variant_str}"
        if cache_key in self.cache:
            logger.info(f"Using cached ClinVar data for {cache_key}")
            return self.cache[cache_key]
        
        try:
            # Construct search query
            search_term = f"{gene}[gene] AND {variant_str}"
            
            # Parameters for esearch
            params = {
                "db": "clinvar",
                "term": search_term,
                "retmode": "json",
                "retmax": 5
            }
            
            if self.email:
                params["email"] = self.email
                params["tool"] = self.tool
            
            # First query: get IDs
            logger.info(f"Searching ClinVar for {search_term}")
            response = requests.get(f"{self.base_url}esearch.fcgi", params=params)
            response.raise_for_status()
            search_result = response.json()
            
            id_list = search_result.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                logger.warning(f"No results found for {search_term}")
                result = {
                    "found": False,
                    "gene": gene,
                    "variant": variant_str,
                    "message": "Variant not found in ClinVar"
                }
                self.cache[cache_key] = result
                return result
            
            # Second query: get details
            fetch_params = {
                "db": "clinvar",
                "id": ",".join(id_list),
                "retmode": "json",
                "rettype": "variation"
            }
            
            if self.email:
                fetch_params["email"] = self.email
                fetch_params["tool"] = self.tool
            
            # Add delay to comply with NCBI API rules
            time.sleep(0.3)
            
            logger.info(f"Fetching details for variant IDs: {id_list}")
            fetch_response = requests.get(f"{self.base_url}esummary.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            details = fetch_response.json()
            
            # Extract relevant information
            result = {
                "found": True,
                "gene": gene,
                "variant": variant_str,
                "clinvar_data": []
            }
            
            for id in id_list:
                if id in details.get("result", {}):
                    variant_data = details["result"][id]
                    
                    # Extract clinical significance
                    clinical_sig = "Unknown"
                    if "clinical_significance" in variant_data:
                        clinical_sig = variant_data["clinical_significance"].get("description", "Unknown")
                    
                    # Extract review status
                    review_status = "Not provided"
                    if "review_status" in variant_data:
                        review_status = variant_data["review_status"]
                    
                    result["clinvar_data"].append({
                        "id": id,
                        "clinical_significance": clinical_sig,
                        "review_status": review_status,
                        "last_updated": variant_data.get("last_updated", "Unknown")
                    })
            
            # Determine overall pathogenicity
            if result["clinvar_data"]:
                pathogenicity_map = {
                    "Pathogenic": 4,
                    "Likely pathogenic": 3,
                    "Uncertain significance": 2,
                    "Likely benign": 1,
                    "Benign": 0
                }
                
                # Find highest pathogenicity rating
                highest_path = 0
                result["pathogenicity"] = "Unknown"
                
                for entry in result["clinvar_data"]:
                    sig = entry["clinical_significance"]
                    if sig in pathogenicity_map and pathogenicity_map[sig] > highest_path:
                        highest_path = pathogenicity_map[sig]
                        result["pathogenicity"] = sig
            
            self.cache[cache_key] = result
            return result
                
        except Exception as e:
            logger.error(f"Error querying ClinVar API: {str(e)}")
            return {
                "found": False,
                "gene": gene,
                "variant": variant_str,
                "error": str(e),
                "message": "Error querying ClinVar API"
            }
    
    def parse_variant_string(self, variant_str: str) -> Dict[str, str]:
        """
        Parse a variant string to extract relevant information.
        
        Args:
            variant_str: Variant string (e.g., NM_001613.4:c.536G>A)
            
        Returns:
            Dictionary with parsed variant information
        """
        try:
            # Parse transcript and variant
            match = re.match(r'(NM_\d+\.\d+):c\.(.+)', variant_str)
            if match:
                transcript, change = match.groups()
                return {
                    "transcript": transcript,
                    "change": change,
                    "full": variant_str
                }
            return {"full": variant_str}
        except:
            return {"full": variant_str}