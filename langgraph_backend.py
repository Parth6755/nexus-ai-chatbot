"""
NEXUS AI - LangGraph backend.

An agentic chatbot built on LangGraph with:
  * OpenAI chat models (model + temperature selectable per request)
  * Tools: DuckDuckGo web search, Alpha Vantage stock quotes,
    Open-Meteo live weather, current date/time, and a local
    Math MCP server (stdio)
  * Persistent conversation memory in SQLite (LangGraph checkpointer)
  * Thread metadata (title, last-updated) for a recency-sorted history
"""

import os
import sys
import sqlite3
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# 1. PATHS AND ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MATH_SERVER_PATH = BASE_DIR / "mcp_servers" / "math_server.py"
DB_PATH = Path(os.getenv("NEXUS_DB_PATH", BASE_DIR / "chatbot.db"))

load_dotenv(BASE_DIR / ".env")


def _load_streamlit_secrets():
    """When deployed (Streamlit Cloud etc.), keys may live in st.secrets."""
    try:
        import streamlit as st

        for key in ("OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY"):
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


_load_streamlit_secrets()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to .env (local) "
        "or to the host's environment variables / secrets (deployed)."
    )

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
DEFAULT_MODEL = os.getenv("NEXUS_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.7


# ============================================================
# 2. ASYNC EVENT LOOP FOR MCP
# ============================================================

background_loop = asyncio.new_event_loop()


def _start_background_loop():
    asyncio.set_event_loop(background_loop)
    background_loop.run_forever()


threading.Thread(target=_start_background_loop, daemon=True).start()


def run_async(coroutine, timeout: float = 60):
    future = asyncio.run_coroutine_threadsafe(coroutine, background_loop)
    return future.result(timeout=timeout)


# ============================================================
# 3. LOCAL MATH MCP TOOLS (non-fatal if unavailable)
# ============================================================

def _convert_mcp_tool(mcp_tool):
    def sync_tool(**kwargs):
        return run_async(mcp_tool.ainvoke(kwargs))

    return StructuredTool.from_function(
        func=sync_tool,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
    )


def _load_mcp_tools():
    if not MATH_SERVER_PATH.exists():
        print(f"[NEXUS] Math MCP server not found: {MATH_SERVER_PATH}")
        return [], False

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "local_math": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(MATH_SERVER_PATH)],
            }
        })
        mcp_tools = run_async(client.get_tools(), timeout=45)
        print("[NEXUS] MCP tools:", ", ".join(t.name for t in mcp_tools))
        return [_convert_mcp_tool(t) for t in mcp_tools], True
    except Exception as error:  # keep the chatbot usable without MCP
        print(f"[NEXUS] MCP unavailable, continuing without it: {error}")
        return [], False


local_math_tools, MCP_ONLINE = _load_mcp_tools()


# ============================================================
# 4. BUILT-IN TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest available stock price for a stock ticker symbol (e.g. AAPL, TSLA)."""

    if not ALPHA_VANTAGE_API_KEY:
        return {"error": "ALPHA_VANTAGE_API_KEY is not configured."}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper(),
                "apikey": ALPHA_VANTAGE_API_KEY,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        quote = data.get("Global Quote", {})

        if not quote:
            return {"error": data.get("Note", data.get("Information", "No stock data returned."))}

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "latest_trading_day": quote.get("07. latest trading day"),
        }
    except Exception as error:
        return {"error": str(error)}


WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Dense drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
    81: "Heavy rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}


@tool
def get_weather(city: str) -> dict:
    """Get the current live weather for a city (temperature, wind, humidity, conditions)."""

    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=15,
        ).json()
        results = geo.get("results")
        if not results:
            return {"error": f"City '{city}' not found."}

        place = results[0]
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "weather_code,wind_speed_10m",
            },
            timeout=15,
        ).json()
        current = weather.get("current", {})

        return {
            "location": f"{place.get('name')}, {place.get('country', '')}".strip(", "),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "conditions": WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
        }
    except Exception as error:
        return {"error": str(error)}


@tool
def get_current_datetime() -> str:
    """Get the current date and time (UTC). Use for any question about today's date or time."""
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")


tools = [search_tool, get_stock_price, get_weather, get_current_datetime, *local_math_tools]
print("[NEXUS] Tools registered:", ", ".join(t.name for t in tools))


# ============================================================
# 5. LLM + SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are NEXUS, a sharp, friendly AI assistant.

Today is {today}.

Guidelines:
- Answer clearly and concisely. Use Markdown: short paragraphs, bullet points,
  tables and fenced code blocks where they help readability.
- Use tools when they give a better answer: `duckduckgo_search` for recent or
  factual web information, `get_stock_price` for stock quotes, `get_weather`
  for live weather, `get_current_datetime` for the date/time, and the math
  tools (add, subtract, multiply, divide) for exact arithmetic.
- When you used web search, briefly mention where the information came from.
- Write math with LaTeX using $...$ (inline) or $$...$$ (block) delimiters,
  never \\( \\) or \\[ \\].
- Never invent tool results. If a tool fails, say so and offer an alternative.
"""


def _get_llm(model: str, temperature: float):
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True,
        api_key=OPENAI_API_KEY,
    ).bind_tools(tools)


# ============================================================
# 6. GRAPH
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config: RunnableConfig):
    settings = config.get("configurable", {})
    llm = _get_llm(
        settings.get("model", DEFAULT_MODEL),
        float(settings.get("temperature", DEFAULT_TEMPERATURE)),
    )
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(today=datetime.now().strftime("%A, %d %B %Y"))
    )
    response = llm.invoke([system, *state["messages"]], config=config)
    return {"messages": [response]}


workflow = StateGraph(ChatState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "chat_node")
workflow.add_conditional_edges("chat_node", tools_condition)
workflow.add_edge("tools", "chat_node")


# ============================================================
# 7. SQLITE MEMORY
# ============================================================

sqlite_connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
checkpointer = SqliteSaver(sqlite_connection)
checkpointer.setup()

_db_lock = threading.Lock()

with _db_lock:
    sqlite_connection.execute(
        """
        CREATE TABLE IF NOT EXISTS nexus_threads (
            thread_id  TEXT PRIMARY KEY,
            title      TEXT,
            updated_at TEXT
        )
        """
    )
    sqlite_connection.commit()

chatbot = workflow.compile(checkpointer=checkpointer)


# ============================================================
# 8. THREAD HELPERS
# ============================================================

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def touch_thread(thread_id: str, title: str | None = None):
    """Create/update a thread's metadata (title is only set the first time)."""
    with _db_lock:
        sqlite_connection.execute(
            """
            INSERT INTO nexus_threads (thread_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                title = COALESCE(nexus_threads.title, excluded.title)
            """,
            (thread_id, title, _now_iso()),
        )
        sqlite_connection.commit()


def rename_thread(thread_id: str, title: str):
    with _db_lock:
        sqlite_connection.execute(
            "UPDATE nexus_threads SET title = ? WHERE thread_id = ?", (title, thread_id)
        )
        sqlite_connection.commit()


def delete_thread(thread_id: str):
    checkpointer.delete_thread(thread_id)
    with _db_lock:
        sqlite_connection.execute("DELETE FROM nexus_threads WHERE thread_id = ?", (thread_id,))
        sqlite_connection.commit()


def get_chat_history(thread_id: str):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    if not state or not state.values:
        return []
    return state.values.get("messages", [])


def _first_user_text(thread_id: str):
    for message in get_chat_history(thread_id):
        if getattr(message, "type", "") == "human":
            text = message.content if isinstance(message.content, str) else str(message.content)
            return text.replace("\n", " ").strip()
    return None


def retrieve_all_threads():
    """Return [{thread_id, title, updated_at}] sorted newest first.

    Threads created before the metadata table existed are back-filled once.
    """
    with _db_lock:
        known = {
            row[0]: {"thread_id": row[0], "title": row[1], "updated_at": row[2] or ""}
            for row in sqlite_connection.execute(
                "SELECT thread_id, title, updated_at FROM nexus_threads"
            )
        }

    checkpoint_threads = {
        c.config.get("configurable", {}).get("thread_id")
        for c in checkpointer.list(None)
    }
    checkpoint_threads.discard(None)

    for thread_id in checkpoint_threads - set(known):
        title = _first_user_text(thread_id) or "New conversation"
        touch_thread(thread_id, title[:60])
        known[thread_id] = {"thread_id": thread_id, "title": title[:60], "updated_at": ""}

    return sorted(
        (t for t in known.values() if t["thread_id"] in checkpoint_threads),
        key=lambda t: t["updated_at"],
        reverse=True,
    )


TOOL_NAMES = [t.name for t in tools]

print(f"[NEXUS] Backend ready | model={DEFAULT_MODEL} | MCP={'on' if MCP_ONLINE else 'off'}")
