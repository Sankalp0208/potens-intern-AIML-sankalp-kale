import streamlit as st
import requests

# ==========================================================
# Configuration
# ==========================================================

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Company Policy & Document Q&A with Citations",
    page_icon="📄",
    layout="wide",
)

# ==========================================================
# API Helper Functions
# ==========================================================

def get_health():
    """
    Calls FastAPI /health endpoint.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health")

        if response.status_code == 200:
            return response.json()

        return None

    except Exception:
        return None


def ask_question(question: str):
    """
    Calls FastAPI /ask endpoint.
    """
    payload = {
        "question": question
    }

    response = requests.post(
        f"{API_BASE_URL}/ask",
        json=payload,
    )

    response.raise_for_status()

    return response.json()


def compare_documents(
    doc1: str,
    doc2: str,
    topic: str,
):
    """
    Calls FastAPI /contradict endpoint.
    """

    payload = {
        "doc1": doc1,
        "doc2": doc2,
        "topic": topic,
    }

    response = requests.post(
        f"{API_BASE_URL}/contradict",
        json=payload,
    )

    response.raise_for_status()

    return response.json()


# ==========================================================
# Header
# ==========================================================

st.title("📄 Document Q&A with Citations")

st.caption(
    "Groq • LangChain • ChromaDB • FastAPI • Streamlit"
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.header("⚙️ System Information")

    health = get_health()

    if health:

        st.success("Backend Connected")

        st.metric(
            "Documents Indexed",
            health["documents_indexed"],
        )

    else:

        st.error("Backend Offline")

    st.divider()

    st.subheader("🌐 Supported Languages")

    st.write("🇬🇧 English")
    st.write("🇮🇳 Hindi")
    st.write("🇮🇳 Marathi")

    st.divider()

    st.subheader("🤖 Model")

    st.info("llama-3.3-70b-versatile")

    st.subheader("💾 Vector Store")

    st.info("ChromaDB")

# ==========================================================
# Tabs
# ==========================================================

tab1, tab2 = st.tabs(
    [
        "💬 Ask Documents",
        "⚖️ Compare Documents",
    ]
)

#part2

# ==========================================================
# Ask Documents Tab
# ==========================================================

with tab1:

    st.subheader("💬 Ask Questions About Your Documents")

    question = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Example: What is the leave policy?",
    )

    ask_button = st.button(
        "🔍 Ask",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching documents..."):

                try:

                    result = ask_question(question)

                    # --------------------------------------------------
                    # Answer
                    # --------------------------------------------------

                    st.success("Answer Generated")

                    st.markdown("## 📖 Answer")

                    st.write(result["answer"])

                    st.divider()

                    # --------------------------------------------------
                    # Confidence
                    # --------------------------------------------------

                    st.markdown("## 📊 Confidence")

                    confidence = result["confidence"]

                    st.progress(confidence)

                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence:.2f}",
                    )

                    st.divider()

                    # --------------------------------------------------
                    # Citations
                    # --------------------------------------------------

                    st.markdown("## 📚 Citations")

                    citations = result["citations"]

                    if not citations:

                        st.info("No citations available.")

                    else:

                        for index, citation in enumerate(
                            citations,
                            start=1,
                        ):

                            with st.expander(
                                f"📄 Citation {index}"
                            ):

                                st.write(
                                    f"**Source:** {citation['source']}"
                                )

                                st.write(
                                    f"**Page:** {citation['page']}"
                                )

                                st.write(
                                    f"**Chunk ID:** {citation['chunk_id']}"
                                )

                                st.write(
                                    f"**Similarity Score:** {citation['score']:.4f}"
                                )

                                st.markdown("**Snippet**")

                                st.info(
                                    citation["snippet"]
                                )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to the FastAPI backend."
                    )

                except Exception as e:

                    st.error(str(e))
                    
#part3
# ==========================================================
# Compare Documents Tab
# ==========================================================

with tab2:

    st.subheader("⚖️ Compare Two Documents")

    col1, col2 = st.columns(2)

    with col1:

        doc1 = st.text_input(
            "Document 1 ID",
            placeholder="Example: leave_policy",
        )

    with col2:

        doc2 = st.text_input(
            "Document 2 ID",
            placeholder="Example: employee_handbook",
        )

    topic = st.text_input(
        "Comparison Topic",
        placeholder="Example: Annual Leave",
    )

    compare_button = st.button(
        "⚖️ Compare Documents",
        use_container_width=True,
    )

    if compare_button:

        if (
            not doc1.strip()
            or not doc2.strip()
            or not topic.strip()
        ):

            st.warning(
                "Please fill in all the fields."
            )

        else:

            with st.spinner("Comparing documents..."):

                try:

                    result = compare_documents(
                        doc1=doc1,
                        doc2=doc2,
                        topic=topic,
                    )

                    st.divider()

                    st.markdown("## 📋 Comparison Result")

                    if result["conflict"]:

                        st.error("⚠️ Conflict Detected")

                    else:

                        st.success("✅ No Conflict Detected")

                    st.divider()

                    st.markdown("### 📝 Reason")

                    st.write(result["reason"])

                    st.divider()

                    st.markdown("### 📄 Evidence")

                    st.info(result["evidence"])

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to the FastAPI backend."
                    )

                except Exception as e:

                    st.error(str(e))