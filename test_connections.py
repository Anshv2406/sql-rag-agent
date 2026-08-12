"""
Quick connectivity test — run this first to confirm:
1. Gemini API key works
2. Postgres (Chinook DB) connection works
Before we build any agent logic on top.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file

print("=" * 50)
print("STEP 1: Testing Gemini API connection...")
print("=" * 50)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )

    response = llm.invoke("Say 'Gemini connection successful' and nothing else.")
    print(f"✅ Gemini response: {response.content}")

except Exception as e:
    print(f"❌ Gemini connection FAILED: {e}")

print()
print("=" * 50)
print("STEP 2: Testing Postgres (Chinook DB) connection...")
print("=" * 50)

try:
    from sqlalchemy import create_engine, text

    db_url = "postgresql://agent_user:agent_pass@localhost:5432/chinook"
    engine = create_engine(db_url)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM album;"))
        count = result.scalar()
        print(f"✅ Postgres connected. Number of albums in Chinook DB: {count}")

except Exception as e:
    print(f"❌ Postgres connection FAILED: {e}")

print()
print("=" * 50)
print("Done. If both steps show ✅, we're ready to build the agent.")
print("=" * 50)
