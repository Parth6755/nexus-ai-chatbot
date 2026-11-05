"""NEXUS AI - Streamlit frontend for the LangGraph agent."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="NEXUS AI · Agentic assistant",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Design layer (animations, glass, aurora). Colours & fonts: .streamlit/config.toml
st.html(BASE_DIR / "assets" / "nexus.css")


# ============================================================
# BACKEND (fails gracefully if keys are missing)
# ============================================================

try:
    from langgraph_backend import (
        AVAILABLE_MODELS,
        DEFAULT_MODEL,
        MCP_ONLINE,
        TOOL_NAMES,
        chatbot,
        delete_thread,
        get_chat_history,
        rename_thread,
        retrieve_all_threads,
        touch_thread,
    )
except ValueError as error:
    st.error(str(error), title="NEXUS can't start", icon=":material/key_off:")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("model", DEFAULT_MODEL)
st.session_state.setdefault("temperature", 0.7)


# ============================================================
# TOOL PRESENTATION
# ============================================================

TOOL_META = {
    "duckduckgo_search": (":material/travel_explore:", "Searching the web", "Searched the web"),
    "get_stock_price": (":material/candlestick_chart:", "Fetching stock quote", "Fetched stock quote"),
    "get_weather": (":material/partly_cloudy_day:", "Checking live weather", "Checked live weather"),
    "get_current_datetime": (":material/schedule:", "Reading the clock", "Read the clock"),
    "add": (":material/calculate:", "Calculating", "Calculated"),
    "subtract": (":material/calculate:", "Calculating", "Calculated"),
    "multiply": (":material/calculate:", "Calculating", "Calculated"),
    "divide": (":material/calculate:", "Calculating", "Calculated"),
}


def tool_meta(name):
    return TOOL_META.get(name, (":material/build:", f"Running {name}", f"Ran {name}"))


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content or "")


def render_tool_details(record):
    args = record.get("args")
    if args:
        st.caption("Input")
        st.code(json.dumps(args, indent=2, ensure_ascii=False), language="json", wrap_lines=True)
    st.caption("Result")
    output = str(record.get("output", ""))
    st.code(output[:1800] + ("…" if len(output) > 1800 else ""), language=None, wrap_lines=True)


def render_tool_timeline(records, label):
    with st.expander(label, icon=":material/bolt:", type="compact"):
        for record in records:
            icon, _, done = tool_meta(record["name"])
            with st.expander(done, icon=icon, type="step"):
                render_tool_details(record)


# ============================================================
# HISTORY <-> UI MESSAGES
# ============================================================

def load_messages(thread_id):
    """Group LangGraph messages into UI turns, keeping tool calls per answer."""
    turns, current = [], None
    pending_args = {}

    for message in get_chat_history(thread_id):
        kind = getattr(message, "type", "")
        if kind == "human":
            if current:
                turns.append(current)
                current = None
            turns.append({"role": "user", "content": text_of(message.content)})
        elif kind == "ai":
            current = current or {"role": "assistant", "content": "", "tools": []}
            for call in getattr(message, "tool_calls", []) or []:
                pending_args[call.get("id")] = call.get("args", {})
            text = text_of(message.content)
            if text:
                current["content"] += ("\n\n" if current["content"] else "") + text
        elif kind == "tool":
            current = current or {"role": "assistant", "content": "", "tools": []}
            current["tools"].append({
                "name": getattr(message, "name", "tool"),
                "args": pending_args.get(getattr(message, "tool_call_id", None), {}),
                "output": text_of(message.content),
            })

    if current:
        turns.append(current)
    return turns


def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def open_chat(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.messages = load_messages(thread_id)


def remove_chat(thread_id):
    delete_thread(thread_id)
    new_chat()


def export_markdown():
    lines = [f"# NEXUS AI conversation\n\n_Exported {datetime.now():%d %b %Y, %H:%M}_\n"]
    for message in st.session_state.messages:
        who = "**You**" if message["role"] == "user" else "**NEXUS**"
        lines.append(f"{who}\n\n{message['content']}\n")
    return "\n---\n\n".join(lines)


# ============================================================
# STREAMING AGENT RUN
# ============================================================

def run_agent(prompt, thread_id):
    """Stream the LangGraph run, rendering tool steps live and tokens as they arrive."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "model": st.session_state.model,
            "temperature": st.session_state.temperature,
        }
    }

    started = time.perf_counter()
    thinking_slot = st.empty()
    thinking_slot.html(
        '<div class="nx-thinking"><span class="nx-dots"><span></span><span></span>'
        "<span></span></span>NEXUS is thinking…</div>"
    )
    tools_box = None
    answer_slot = st.empty()

    answer = ""
    records = []
    steps = {}  # tool_call_id -> status container
    call_ids = {}  # chunk index -> tool_call_id
    call_args = {}  # tool_call_id -> raw json args
    last_paint = 0.0

    for chunk, meta in chatbot.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="messages",
    ):
        node = meta.get("langgraph_node", "")

        if node == "chat_node":
            for part in getattr(chunk, "tool_call_chunks", None) or []:
                index = part.get("index", 0)
                if part.get("id"):
                    call_ids[index] = part["id"]
                call_id = call_ids.get(index)
                if call_id is None:
                    continue
                call_args[call_id] = call_args.get(call_id, "") + (part.get("args") or "")

                if part.get("name") and call_id not in steps:
                    if tools_box is None:
                        thinking_slot.empty()
                        tools_box = thinking_slot.container().status(
                            ":shimmer[Working with tools]", type="compact", expanded=True
                        )
                    icon, running, _ = tool_meta(part["name"])
                    with tools_box:
                        steps[call_id] = st.status(f":shimmer[{running}…]", type="step")

            text = text_of(chunk.content)
            if text:
                if tools_box is None:
                    thinking_slot.empty()
                answer += text
                now = time.perf_counter()
                if now - last_paint > 0.03:
                    answer_slot.markdown(answer + " ▍")
                    last_paint = now

        elif node == "tools":
            call_id = getattr(chunk, "tool_call_id", None)
            try:
                args = json.loads(call_args.get(call_id) or "{}")
            except json.JSONDecodeError:
                args = {"raw": call_args.get(call_id)}
            record = {"name": getattr(chunk, "name", "tool"), "args": args, "output": text_of(chunk.content)}
            records.append(record)

            step = steps.get(call_id)
            if step is None and tools_box is not None:
                with tools_box:
                    step = st.status("", type="step")
            if step is not None:
                with step:
                    render_tool_details(record)
                _, _, done = tool_meta(record["name"])
                step.update(label=done, state="complete", expanded=False)

    elapsed = time.perf_counter() - started
    if tools_box is None:
        thinking_slot.empty()
    answer_slot.markdown(answer or "_No response._")

    if tools_box is not None:
        tools_box.update(
            label=f"Used {len(records)} tool{'s' if len(records) != 1 else ''} · {elapsed:.1f}s",
            state="complete",
            expanded=False,
        )

    return {"role": "assistant", "content": answer, "tools": records, "elapsed": elapsed}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.html(
        """
        <div class="nx-brand">
            <div class="nx-logo"></div>
            <div>
                <div class="nx-brand-name">NEXUS</div>
                <div class="nx-brand-tag">AGENTIC AI ASSISTANT</div>
            </div>
        </div>
        """
    )

    st.button(
        "New conversation",
        icon=":material/add_comment:",
        type="primary",
        width="stretch",
        on_click=new_chat,
    )

    st.html('<div class="nx-section">Conversations</div>')
    query = st.text_input(
        "Search conversations",
        placeholder="Search chats…",
        icon=":material/search:",
        label_visibility="collapsed",
    )

    try:
        threads = retrieve_all_threads()
    except Exception as error:
        threads = []
        st.caption(f"History unavailable: {error}")

    if query:
        threads = [t for t in threads if query.lower() in (t["title"] or "").lower()]

    with st.container(height=300 if len(threads) > 7 else "content", border=False):
        if not threads:
            st.caption("No conversations yet — say hi! 👋")
        for thread in threads[:60]:
            active = thread["thread_id"] == st.session_state.thread_id
            title = thread["title"] or "New conversation"
            st.button(
                title if len(title) <= 34 else title[:33] + "…",
                key=f"thread_{thread['thread_id']}",
                icon=":material/chat_bubble:" if active else ":material/chat_bubble_outline:",
                type="secondary" if active else "tertiary",
                width="stretch",
                on_click=open_chat,
                args=(thread["thread_id"],),
            )

    if st.session_state.messages:
        st.html('<div class="nx-section">This conversation</div>')
        with st.container(horizontal=True, gap="small"):
            with st.popover("Rename", icon=":material/edit:", width="stretch"):
                new_title = st.text_input("New title", max_chars=60)
                if st.button("Save", type="primary", disabled=not new_title):
                    rename_thread(st.session_state.thread_id, new_title)
                    st.rerun()
            st.download_button(
                "Export",
                data=export_markdown(),
                file_name="nexus-conversation.md",
                mime="text/markdown",
                icon=":material/download:",
                on_click="ignore",
                width="stretch",
            )
        st.button(
            "Delete conversation",
            icon=":material/delete:",
            width="stretch",
            on_click=remove_chat,
            args=(st.session_state.thread_id,),
        )

    st.html('<div class="nx-section">Model</div>')
    st.selectbox("Model", AVAILABLE_MODELS, key="model", label_visibility="collapsed")
    st.slider(
        "Creativity",
        0.0,
        1.2,
        key="temperature",
        step=0.1,
        help="Lower = precise and factual, higher = more creative.",
    )

    st.html(
        f"""
        <div class="nx-section">System</div>
        <div class="nx-monitor">
            <div class="nx-monitor-row"><span>LangGraph agent</span><span class="nx-led on">READY</span></div>
            <div class="nx-monitor-row"><span>SQLite memory</span><span class="nx-led on">ONLINE</span></div>
            <div class="nx-monitor-row"><span>Math MCP server</span>
                <span class="nx-led {'on' if MCP_ONLINE else 'off'}">{'ONLINE' if MCP_ONLINE else 'OFFLINE'}</span></div>
            <div class="nx-monitor-row"><span>Tools loaded</span><span class="nx-led on">{len(TOOL_NAMES)}</span></div>
        </div>
        """
    )


# ============================================================
# WELCOME SCREEN
# ============================================================

SUGGESTIONS = {
    ":violet[:material/travel_explore:] Latest AI news": "What are the most important AI news stories this week?",
    ":blue[:material/partly_cloudy_day:] Weather in Berlin": "What's the weather like in Berlin right now?",
    ":green[:material/candlestick_chart:] NVIDIA stock": "What is NVIDIA's current stock price and today's change?",
    ":orange[:material/calculate:] Quick maths": "What is (1847 * 23) divided by 7?",
    ":red[:material/code:] Python help": "Write a Python function that checks whether a string is a palindrome, with tests.",
}

prompt = st.chat_input("Message NEXUS…", submit_mode="stop")

if not st.session_state.messages:
    st.html(
        """
        <section class="nx-hero">
            <div class="nx-orb-wrap">
                <div class="nx-orbit"></div>
                <div class="nx-orb-ring"></div>
                <div class="nx-orb"></div>
            </div>
            <div class="nx-eyebrow">SYSTEM ONLINE</div>
            <h1 class="nx-title">Hello, I'm <span class="nx-gradient-text">NEXUS</span></h1>
            <div class="nx-typing">Search the web, crunch numbers, check stocks and weather.</div>
        </section>
        <div class="nx-grid">
            <div class="nx-card"><div class="nx-card-icon">🧠</div><div class="nx-card-title">Agentic reasoning</div>
                <div class="nx-card-text">LangGraph decides when to think and when to reach for a tool.</div></div>
            <div class="nx-card"><div class="nx-card-icon">🌐</div><div class="nx-card-title">Live web search</div>
                <div class="nx-card-text">Fresh answers from DuckDuckGo, with sources.</div></div>
            <div class="nx-card"><div class="nx-card-icon">📈</div><div class="nx-card-title">Markets</div>
                <div class="nx-card-text">Real-time stock quotes via Alpha Vantage.</div></div>
            <div class="nx-card"><div class="nx-card-icon">⛅</div><div class="nx-card-title">Weather</div>
                <div class="nx-card-text">Current conditions for any city on Earth.</div></div>
            <div class="nx-card"><div class="nx-card-icon">🧮</div><div class="nx-card-title">MCP math server</div>
                <div class="nx-card-text">Exact arithmetic through a Model Context Protocol server.</div></div>
            <div class="nx-card"><div class="nx-card-icon">💾</div><div class="nx-card-title">Persistent memory</div>
                <div class="nx-card-text">Every chat is saved — pick up right where you left off.</div></div>
        </div>
        <div class="nx-divider">— TRY ONE OF THESE —</div>
        """
    )
    suggestion_slot = st.empty()
    picked = suggestion_slot.pills(
        "Suggestions",
        list(SUGGESTIONS),
        label_visibility="collapsed",
        key=f"suggest_{st.session_state.thread_id}",
    )
    if picked and not prompt:
        prompt = SUGGESTIONS[picked]
    if prompt:
        suggestion_slot.empty()


# ============================================================
# CONVERSATION
# ============================================================

AVATARS = {"user": "user", "assistant": ":material/auto_awesome:"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        if message.get("tools"):
            n = len(message["tools"])
            render_tool_timeline(message["tools"], f"Used {n} tool{'s' if n != 1 else ''}")
        st.markdown(message["content"])

if prompt:
    thread_id = st.session_state.thread_id
    is_first = not st.session_state.messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    touch_thread(thread_id, prompt.replace("\n", " ").strip()[:60] if is_first else None)

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            reply = run_agent(prompt, thread_id)
            st.session_state.messages.append(reply)
            touch_thread(thread_id)
        except Exception as error:
            st.error(f"{error}", title="NEXUS hit a problem", icon=":material/error:")
            st.stop()

    st.rerun()


st.html('<div class="nx-footer">POWERED BY OPENAI · LANGGRAPH · MCP · STREAMLIT</div>')
