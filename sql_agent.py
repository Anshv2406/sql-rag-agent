import os
import time
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool

# --- 1. Connect to the database ---
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise ValueError(
        "No DATABASE_URL set in .env. Example formats:\n"
        "  SQLite:    sqlite:///path/to/your.db\n"
        "  Postgres:  postgresql://user:pass@host:port/dbname\n"
        "  MySQL:     mysql+pymysql://user:pass@host:port/dbname"
    )

try:
    db = SQLDatabase.from_uri(DB_URL)
except Exception as e:
    raise RuntimeError(
        f"Could not connect to database at '{DB_URL}'.\n"
        f"Check the connection string and that the DB server is running.\n"
        f"Original error: {e}"
    )

print(f"Connected to: {DB_URL.split('@')[-1] if '@' in DB_URL else DB_URL}")
print("Detected tables:", db.get_usable_table_names())
print()

# --- 1b. Load the vector store as a retrieval tool ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_knowledge_base",
    description=(
        "Search company policy and FAQ documents (refund policy, account help, "
        "playlist sharing, etc.). Use this for questions about policies or how-to "
        "topics — NOT for questions about specific data like customers, tracks, or sales."
    ),
)

# --- 2. Set up the LLM ---
model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
print("Using model:", model_name)

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# --- 3. Build the SQL + retriever toolkit and agent ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools() + [retriever_tool]

system_prompt = """You are an agent that can answer questions using two sources:
1. A SQL database — for questions about specific data (customers, tracks, sales, employees, etc.)
2. A knowledge base search tool — for questions about policies, refunds, or how-to/FAQ topics

Pick the right tool based on the question. If a query fails, rewrite it and try again.
Never run destructive SQL statements (INSERT/UPDATE/DELETE/DROP)."""

checkpointer = InMemorySaver()
agent = create_agent(llm, tools, system_prompt=system_prompt, checkpointer=checkpointer)

# --- 4. Simple interactive loop for testing ---
print("SQL + RAG Agent ready. Ask questions about the database or policies (type 'exit' to quit).\n")

config = {"configurable": {"thread_id": "session-1"}}
MAX_RETRIES = 3

while True:
    question = input("You: ")
    if question.strip().lower() == "exit":
        break

    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            break  # success, exit retry loop
        except ChatGoogleGenerativeAIError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < MAX_RETRIES:
                wait_time = 30
                print(f"\nRate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {wait_time}s...\n")
                time.sleep(wait_time)
            else:
                print(f"\nAgent error: {e}\n")
                result = None
                break

    if result is None:
        print("Couldn't get a response. Try again in a bit.\n")
        continue

    final_message = result["messages"][-1]
    content = final_message.content
    if isinstance(content, list):
        text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        text = content
    print("\nAgent:", text, "\n")