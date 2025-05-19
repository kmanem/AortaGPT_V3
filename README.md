AortaGPT_V2: Clinical Decision Support Tool for HTAD
AortaGPT is a specialized clinical decision support tool designed to generate guideline-based, gene- and variant-specific recommendations for patients with Heritable Thoracic Aortic Disease (HTAD). It combines a Retrieval-Augmented Generation (RAG) system with advanced Kaplan-Meier curve visualization to provide personalized clinical guidance.

Features
Narrative Input: Describe patients in natural language
Personalized Recommendations: Generate gene and variant-specific clinical recommendations
Guideline-Based: Uses only approved medical guidelines and literature
Visualization: Interactive Kaplan-Meier survival curves based on MAC Consortium data
ClinVar Integration: Real-time variant pathogenicity assessment
Interactive Chat: Discuss patient cases with the AI assistant
Structured Output: Recommendations organized into clinical categories
Installation
Clone the repository:
bash
git clone https://github.com/yourusername/AortaGPT_V2.git
cd AortaGPT_V2
Install dependencies:
bash
pip install -r requirements.txt
Add your API keys to .streamlit/secrets.toml:
toml
openai_api_key = "your-openai-api-key-here"
Add reference documents:
Place guideline PDFs, GeneReviews, and other reference materials in the data/raw/ folder
Running the Application
bash
streamlit run app.py
System Architecture
AortaGPT combines several advanced components:

Vector Store Manager: Indexes and retrieves relevant content from medical guidelines
ClinVar API: Queries variant pathogenicity from the NCBI ClinVar database
KM Visualizer: Generates personalized Kaplan-Meier survival curves
OpenAI Integration: Leverages GPT models for natural language processing
Reference Materials
The system uses the following reference materials:

2022 ACC/AHA Guidelines on Aortic Disease
GeneReviews entries for HTAD genes
MAC Consortium Data for survival curves
European Hypertension Guidelines
HTAD Diagnostic Pathway documents
Variant-specific recommendations from structured databases
Usage
Enter Patient Information:
Use the natural language input to describe the patient
Or manually enter details in the sidebar
Generate Recommendations:
Click "Generate Recommendations" to run the RAG system and create personalized guidance
Explore Results:
Review structured recommendations across multiple clinical categories
Examine the Kaplan-Meier survival curve
Use the chat interface to ask follow-up questions
License
[Add your license information here]

Acknowledgments
[Add acknowledgments here]
