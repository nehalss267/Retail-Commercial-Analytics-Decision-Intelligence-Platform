"""AI Copilot page."""
import streamlit as st

from src.ai.agent import process_query


def render():
    st.title("AI Business Copilot")

    st.caption("Ask questions about your retail data, models, or methodology.")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "source" in msg:
                st.caption(f"Source: {msg['source']}")

    # User input
    if question := st.chat_input("Ask a business question..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = process_query(question)
                answer = result.get("answer", "I couldn't find an answer.")
                source = result.get("source", "unknown")

                st.write(answer)
                st.caption(f"Source: {source}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "source": source,
        })
