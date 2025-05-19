
import os
import sys
from openai import OpenAI

# Try to get API key from environment or command line
api_key = sys.argv[1] if len(sys.argv) > 1 else None

if not api_key:
    # Try to read from .streamlit/secrets.toml
    try:
        with open('.streamlit/secrets.toml', 'r') as f:
            content = f.read()
            import re
            match = re.search(r'openai_api_key\s*=\s*"([^"]+)"', content)
            if match:
                api_key = match.group(1)
    except:
        pass

if not api_key:
    print("No API key provided. Please provide as command line argument.")
    sys.exit(1)

print(f"Testing API key: {api_key[:5]}...{api_key[-4:]}")

try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print("API key works! Response:", response.choices[0].message.content)
except Exception as e:
    print("API key error:", str(e))
