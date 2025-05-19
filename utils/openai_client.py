"""
OpenAI API integration for AortaGPT.
Handles interactions with OpenAI's language models for generating recommendations.
"""

import logging
import time
from typing import Dict, Any, Optional, List
import json

import openai
from openai import OpenAI, APIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Manages interactions with OpenAI API for generating clinical recommendations.
    Supports both Chat Completions API and Assistants API.
    """
    
    def __init__(self, 
                 api_key: str,
                 assistant_id: Optional[str] = None,
                 model: str = "gpt-4",
                 max_tokens: int = 4000,
                 temperature: float = 0.1):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            assistant_id: Optional Assistant ID for using the Assistants API
            model: OpenAI model to use for Chat Completions
            max_tokens: Maximum tokens for completion
            temperature: Temperature for completion (lower = more deterministic)
        """
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Flag to determine which API to use
        self.use_assistant_api = assistant_id is not None
        
        # Track thread ID for Assistant API
        self.active_thread_id = None
    
    def generate_recommendations(self, context: str) -> Dict[str, Any]:
        """
        Generate clinical recommendations based on patient context.
        
        Args:
            context: Full patient context with RAG retrieval
            
        Returns:
            Dictionary with recommendations
        """
        if self.use_assistant_api:
            return self._generate_with_assistant(context)
        else:
            return self._generate_with_chat(context)
    
    def _generate_with_chat(self, context: str) -> Dict[str, Any]:
        """Use Chat Completions API to generate recommendations."""
        try:
            logger.info(f"Generating recommendations with {self.model}")
            
            # Add specific system instructions for recommendation format
            system_message = """You are AortaGPT, a specialized clinical decision support AI for Heritable Thoracic Aortic Disease (HTAD).
Generate evidence-based recommendations for the patient based on the provided guidelines and clinical details.
Your recommendations should be comprehensive, specific, and directly tied to the evidence.

Your response should be a valid JSON object with the following structure:
{
    "initial_workup": "HTML-formatted recommendations for initial diagnostic workup",
    "risk_stratification": "HTML-formatted recommendations for risk assessment",
    "surgical_thresholds": "HTML-formatted recommendations for when surgery should be considered",
    "imaging_surveillance": "HTML-formatted recommendations for imaging frequency and modality",
    "lifestyle_activity": "HTML-formatted recommendations for physical activity and lifestyle",
    "pregnancy": "HTML-formatted recommendations for pregnancy management if applicable",
    "genetic_counseling": "HTML-formatted recommendations for genetic testing and family screening",
    "blood_pressure": "HTML-formatted recommendations for blood pressure targets and management",
    "medication": "HTML-formatted recommendations for medication management",
    "variant_interpretation": "HTML-formatted assessment of genetic variant significance"
}

Each HTML section should include relevant citations to guidelines. Use HTML formatting for readability,
including <strong> for emphasis, <ul>/<li> for lists, and <p> for paragraphs.

Ensure recommendations are:
1. Based ONLY on the provided evidence, not general knowledge
2. Specific to the patient's gene, variant, age, sex, and measurements
3. Referenced to specific guidelines when available
"""
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": context}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Extract and validate response
            result = json.loads(response.choices[0].message.content)
            
            # Ensure all expected sections are present
            expected_sections = [
                "initial_workup",
                "risk_stratification",
                "surgical_thresholds",
                "imaging_surveillance",
                "lifestyle_activity",
                "pregnancy",
                "genetic_counseling",
                "blood_pressure",
                "medication",
                "variant_interpretation"
            ]
            
            for section in expected_sections:
                if section not in result:
                    result[section] = ""
            
            logger.info("Successfully generated recommendations")
            return result
            
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return self._generate_error_response("OpenAI API error occurred.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._generate_error_response("Failed to parse model response.")
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._generate_error_response("An unexpected error occurred.")
    
    def _generate_with_assistant(self, context: str) -> Dict[str, Any]:
        """Use Assistants API to generate recommendations."""
        try:
            logger.info(f"Generating recommendations with Assistant ID: {self.assistant_id}")
            
            # Create a new thread
            thread = self.client.beta.threads.create()
            self.active_thread_id = thread.id
            
            # Add a message to the thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=context
            )
            
            # Run the Assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id,
                instructions="""Generate evidence-based recommendations for the patient based on the provided guidelines and clinical details.
Your response should be a valid JSON object with sections for different clinical domains."""
            )
            
            # Wait for completion
            max_wait_time = 60  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    # Get the assistant's response
                    messages = self.client.beta.threads.messages.list(
                        thread_id=thread.id
                    )
                    
                    # Extract the latest assistant message
                    for msg in messages.data:
                        if msg.role == "assistant":
                            try:
                                content_text = msg.content[0].text.value
                                # Extract JSON from the response
                                json_text = self._extract_json(content_text)
                                result = json.loads(json_text)
                                
                                # Ensure all expected sections are present
                                expected_sections = [
                                    "initial_workup",
                                    "risk_stratification",
                                    "surgical_thresholds",
                                    "imaging_surveillance",
                                    "lifestyle_activity",
                                    "pregnancy",
                                    "genetic_counseling",
                                    "blood_pressure",
                                    "medication",
                                    "variant_interpretation"
                                ]
                                
                                for section in expected_sections:
                                    if section not in result:
                                        result[section] = ""
                                
                                logger.info("Successfully generated recommendations with Assistant")
                                return result
                            except (json.JSONDecodeError, IndexError) as e:
                                logger.error(f"Error parsing assistant response: {str(e)}")
                                return self._generate_error_response("Failed to parse assistant response.")
                    
                    # If we reach here, no valid response was found
                    return self._generate_error_response("No valid response from assistant.")
                
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    logger.error(f"Assistant run failed with status: {run_status.status}")
                    return self._generate_error_response(f"Assistant run failed with status: {run_status.status}")
                
                # Wait before checking again
                time.sleep(1)
            
            # If we reach here, timeout occurred
            logger.error("Assistant run timed out")
            return self._generate_error_response("Assistant run timed out.")
            
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return self._generate_error_response("OpenAI API error occurred.")
        except Exception as e:
            logger.error(f"Error generating recommendations with Assistant: {str(e)}")
            return self._generate_error_response("An unexpected error occurred.")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from a text string."""
        # Look for JSON between triple backticks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        import re
        match = re.search(json_pattern, text)
        
        if match:
            return match.group(1)
        
        # If no JSON in triple backticks, assume the entire text is JSON
        return text
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate a standardized error response."""
        return {
            "initial_workup": f"<p class='error'>{error_message} Please try again.</p>",
            "risk_stratification": "",
            "surgical_thresholds": "",
            "imaging_surveillance": "",
            "lifestyle_activity": "",
            "pregnancy": "",
            "genetic_counseling": "",
            "blood_pressure": "",
            "medication": "",
            "variant_interpretation": ""
        }