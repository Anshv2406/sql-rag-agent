import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

st.set_page_config(page_title="SQL + RAG Agent", page_icon="🤖", layout="centered")

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "sqlite:///chinook.db")


@st.cache_resource(show_spinner=False)
def load_retriever_tool():
    """Knowledge base is fixed (policy docs) regardless of which DB is connected."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return create_retriever_tool(
        retriever,
        name="search_knowledge_base",
        description=(
            "Search company policy and FAQ documents (refund policy, account help, "
            "playlist sharing, etc.). Use this for questions about policies or how-to "
            "topics — NOT for questions about specific data in the database."
        ),
    )


def build_agent(db_url: str):
    """Builds a fresh agent connected to whatever DB URL is passed in."""
    db = SQLDatabase.from_uri(db_url)

    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools() + [load_retriever_tool()]

    system_prompt = """You are an agent that can answer questions using two sources:
1. A SQL database — for questions about specific data in the connected database
2. A knowledge base search tool — for questions about policies, refunds, or how-to/FAQ topics

Pick the right tool based on the question. If a query fails, rewrite it and try again.
Never run destructive SQL statements (INSERT/UPDATE/DELETE/DROP)."""

    checkpointer = InMemorySaver()
    agent = create_agent(llm, tools, system_prompt=system_prompt, checkpointer=checkpointer)
    return agent, db.get_usable_table_names(), model_name


# --- Session state setup ---
if "db_url" not in st.session_state:
    st.session_state.db_url = DEFAULT_DB_URL
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = os.urandom(8).hex()
if "connection_error" not in st.session_state:
    st.session_state.connection_error = None

# --- Sidebar: dynamic DB connection ---
with st.sidebar:
    st.header("Database Connection")
    st.caption("Try the default Chinook sample DB, or connect your own.")

    db_url_input = st.text_input(
        "Connection string",
        value=st.session_state.db_url,
        type="password",
        help=(
            "sqlite:///path/to.db\n"
            "postgresql://user:pass@host:port/dbname\n"
            "mysql+pymysql://user:pass@host:port/dbname\n\n"
            "Tip: use a read-only DB user for safety."
        ),
    )

    if st.button("Connect"):
        with st.spinner("Connecting..."):
            try:
                agent, tables, model_name = build_agent(db_url_input)
                st.session_state.agent = agent
                st.session_state.db_url = db_url_input
                st.session_state.tables = tables
                st.session_state.model_name = model_name
                st.session_state.connection_error = None
                st.session_state.messages = []
                st.session_state.thread_id = os.urandom(8).hex()
                st.success("Connected!")
            except Exception as e:
                st.session_state.connection_error = str(e)

    if st.session_state.connection_error:
        st.error(f"Connection failed: {st.session_state.connection_error}")

    if st.session_state.agent is not None:
        db_display = (
            st.session_state.db_url.split("@")[-1]
            if "@" in st.session_state.db_url
            else st.session_state.db_url
        )
        st.write(f"**Connected to:** `{db_display}`")
        st.write(f"**Model:** `{st.session_state.model_name}`")
        st.write(f"**Tables:** {len(st.session_state.tables)}")
        with st.expander("View tables"):
            for t in st.session_state.tables:
                st.write(f"- {t}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = os.urandom(8).hex()
        st.rerun()

# --- Initial connection on first load ---
if st.session_state.agent is None and st.session_state.connection_error is None:
    with st.spinner("Connecting to default database..."):
        try:
            agent, tables, model_name = build_agent(st.session_state.db_url)
            st.session_state.agent = agent
            st.session_state.tables = tables
            st.session_state.model_name = model_name
        except Exception as e:
            st.session_state.connection_error = str(e)
            st.error(f"Could not connect to default database: {e}")

st.title("🤖 SQL + RAG Agent")
st.caption("Ask questions about the connected database or knowledge base documents.")

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if st.session_state.agent is not None:
    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": question}]},
                        config=config,
                    )
                    final_message = result["messages"][-1]
                    content = final_message.content
                    if isinstance(content, list):
                        answer = "".join(b.get("text", "") for b in content if isinstance(b, dict))
                    else:
                        answer = content
                except ChatGoogleGenerativeAIError as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        answer = "⚠️ Rate limit hit. Please wait a moment and try again."
                    else:
                        answer = f"⚠️ Error: {e}"
                except Exception as e:
                    answer = f"⚠️ Error: {e}"

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Waiting for a database connection. Use the sidebar to connect.")