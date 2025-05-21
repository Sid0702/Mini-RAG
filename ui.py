import streamlit as st
import json
import os
import time
import app as rag

# --- Configuration ---
rag.TOP_K_RESULTS = 2
MAX_FILES = 5
SUPPORTED_TYPES = ["txt", "md","pdf"]

def main():
    st.set_page_config(page_title="RAG Assignment", layout="centered")

    st.title("🧠 Mini RAG - Simplified Q&A System")

    # --- Upload Section ---
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload up to 5 text/markdown files", type=SUPPORTED_TYPES, accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        if len(uploaded_files) > MAX_FILES:
            st.error(f"Please upload at most {MAX_FILES} files.")
        else:
            with st.spinner("Processing files..."):
                time.sleep(1)
                result = rag.ingest_documents(uploaded_files)
                st.success(result)

    # --- Question Section ---
    st.header("2. Ask a Question")
    question = st.text_input("Type your question below:")

    show_sources = st.checkbox("Show source chunks used", value=True)

    if st.button("Get Answer"):
        if not (hasattr(rag, 'chunks') and rag.chunks):
            st.error("Please upload and process documents first.")
        elif not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer, sources = rag.answer_question(question)
                st.markdown("### ✅ Answer")
                st.markdown(answer)

                if show_sources and sources:
                    st.markdown("#### 📚 Source Chunks Used")
                    for src in sources:
                        with st.expander(src["source"]):
                            st.markdown(src["text"])

    # --- Interaction Log Section ---
    st.header("3. Interaction History")
    if os.path.exists(rag.LOGS_FILE):
        with open(rag.LOGS_FILE, 'r') as f:
            logs = json.load(f)
            if logs:
                for log in reversed(logs[-5:]):  # Show last 5
                    with st.expander(f"Q: {log['question']}"):
                        st.write(f"🕒 {log['timestamp']}")
                        st.write(f"📝 **Answer:** {log['answer'][:300]}{'...' if len(log['answer']) > 300 else ''}")
                        st.write(f"📄 **Sources:** {', '.join(log['sources'])}")
            else:
                st.info("No logs yet.")
    else:
        st.info("No logs file found.")

    st.markdown("<hr style='margin-top:30px'>", unsafe_allow_html=True)
    st.caption("Built for RAG Assignment | Streamlit UI")

if __name__ == "__main__":
    main()
