"""
Lists the exact models available to YOUR API key right now.
Run this once to get a definitive answer instead of guessing model names.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
data = response.json()

print("Models available to your API key that support generateContent:\n")
for model in data.get("models", []):
    methods = model.get("supportedGenerationMethods", [])
    if "generateContent" in methods:
        name = model["name"].replace("models/", "")
        print(f" - {name}")
