# SQL + RAG Agent

A conversational AI agent that can answer questions against **any SQL database** and a **knowledge base of documents**, choosing the right source automatically. Built with LangChain, LangGraph, and Google Gemini.

Inspired by [Farzad Roozitalab's](https://github.com/Farzad-R) work on combining SQL and RAG-based agents.

## Features

- 🗄️ **Dynamic SQL agent** — works against any database (SQLite, Postgres, MySQL) via a single connection string, not hardcoded to one schema
- 📚 **RAG knowledge base** — searches local documents (policies, FAQs) using local embeddings, no API cost
- 🧠 **Conversation memory** — handles multi-turn follow-up questions without repeating context
- 🔁 **Rate-limit resilience** — automatically retries on Gemini API quota errors instead of crashing
- 🔐 **Config-driven** — model, database, and API key all set via `.env`, no code edits needed to switch

## Setup

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Configure environment

Create a `.env` file in the project root:

    DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
    GOOGLE_API_KEY=your_gemini_api_key
    GEMINI_MODEL=gemini-flash-lite-latest

Supported `DATABASE_URL` formats:

    SQLite:    sqlite:///path/to/your.db
    Postgres:  postgresql://user:pass@host:port/dbname
    MySQL:     mysql+pymysql://user:pass@host:port/dbname

### 3. Add knowledge base documents

Drop `.txt` files into the `knowledge_base/` folder, then build the vector index:

    python build_vectorstore.py

Re-run this any time you add or change documents.

### 4. Run the agent

    python sql_agent.py

## Usage

Ask questions in plain English — the agent decides whether to query the database or search the knowledge base:

    You: which country has the most customers?
    Agent: The country with the most customers is the USA, with 13 customers.

    You: what's the refund policy?
    Agent: Refunds are available within 14 days of purchase...

Type `exit` to quit.

## Switching to a different database

Change `DATABASE_URL` in `.env` — no code changes required. The agent inspects the schema at runtime, so it adapts to whatever tables exist.

## Project structure

    sql-rag-agent/
    ├── sql_agent.py          # main agent (SQL + RAG + memory)
    ├── build_vectorstore.py  # builds the knowledge base vector index
    ├── knowledge_base/       # source documents for RAG (.txt files)
    ├── chroma_db/            # generated vector store (do not edit manually)
    ├── .env                  # config (not committed — see .gitignore)
    └── requirements.txt


### Screenshots

![SQL Agent Interface 1](screenshots/sql1.png)
![SQL Agent Interface 2](screenshots/sql2.png)
![SQL Agent Interface 3](screenshots/sql3.png)
![SQL Agent Interface 4](screenshots/sql4.png)

## Known limitations

- Conversation memory is in-memory only — resets when the script restarts
- Free-tier Gemini quota is limited (5 requests/min on some models); the agent retries automatically on rate limits
