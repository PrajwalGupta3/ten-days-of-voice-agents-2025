import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key
load_dotenv(".env.local")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found. Check your .env.local file.")
else:
    print(f"✅ Found API Key. Checking available models...")
    try:
        genai.configure(api_key=api_key)
        
        print("\n--- AVAILABLE MODELS ---")
        for m in genai.list_models():
            # We only care about models that can generate content (chat)
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
        print("------------------------\n")
        
    except Exception as e:
        print(f"❌ Error connecting to Google: {e}")