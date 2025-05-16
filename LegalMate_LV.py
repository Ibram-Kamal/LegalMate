
# streamlit_rag_legalmate.py
import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# —————————————————————————————————————————
# Configuration
# —————————————————————————————————————————
API_KEY = os.getenv("GENAI_API_KEY", "PUT_HERE_YOUR_API")
MODEL_ID = "gemini-2.5-flash-preview-04-17"

if not API_KEY or API_KEY.startswith("YOUR_"):
    st.error("Please set your GENAI_API_KEY environment variable.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# —————————————————————————————————————————
# Helper: extract text from uploaded file
# —————————————————————————————————————————
def extract_text(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    return ""

# —————————————————————————————————————————
# Streamlit App
# —————————————————————————————————————————
st.set_page_config(page_title="Legal Mate", layout="wide")
st.title("📜 Legal Mate")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Arabic contract (PDF or TXT)", type=["pdf","txt"], key="file_uploader")

# If a new file is uploaded (different from session), reset everything
if uploaded_file and uploaded_file != st.session_state.get("current_file"):
    # reset states for new contract
    st.session_state.current_file = uploaded_file
    st.session_state.contract_text = extract_text(uploaded_file)
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.index = None
    st.session_state.chunk_map = {}
    st.session_state.embed_model = None
    # clear history of Q&A as needed or keep
    # st.session_state.history = []  # uncomment to reset chat history on new file

if not uploaded_file:
    st.info("Please upload a contract file to begin.")
    st.stop()

# 2. Initialize Session State for contract processing (after file check)
if "contract_text" not in st.session_state:
    # in case session restored without new upload
    st.session_state.contract_text = extract_text(uploaded_file)
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.index = None
    st.session_state.chunk_map = {}
    st.session_state.embed_model = None
if "contract_text" not in st.session_state:
    st.session_state.contract_text = extract_text(uploaded_file)
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.index = None
    st.session_state.chunk_map = {}
    st.session_state.embed_model = None

# 3. Split into Chunks
if st.session_state.chunks is None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    st.session_state.chunks = splitter.split_text(st.session_state.contract_text)
    st.success(f"Contract split into {len(st.session_state.chunks)} chunks.")

# 4. Initialize Chat & System Prompt (once per file)
if "chat" not in st.session_state:
    # Create new chat session and send system prompt
    st.session_state.chat = client.chats.create(model=MODEL_ID)
    # build & save system prompt
    system_prompt = f"""
أنتَ “ليجال ميت”، مساعدٌ ذكِيٌّ مختصٌّ في العقود القانونيّة باللغة العربيّة.

1. قسّم النصّ العقديّ إلى عناوين وبنود مرقّمة.
2. بالنسبة لكلّ بند:
   a) ألخِصه بلغةٍ عربيّةٍ فصحىٍ واضحة وبسيطة ودقيقة.
   b) حدّد النقاط الضعيفة أو المبهمة.
   c) صنّف خطورة كلّ نقطة: (منخفضة / متوسّطة / عالية).
   d) قدّم توصيةً أو توصيتين عمليّتين.

التنسيق بدقة كما يلي:

البند 1:
• الملخص: …
• نقاط الضعف والمخاطر:

1. … — الخطورة: (منخفضة/متوسّطة/عالية)
2. … — الخطورة: (منخفضة/متوسّطة/عالية)
• التوصيات:
3. …
4. …

## نص العقد:
---
{st.session_state.contract_text}
---
"""
    st.session_state.chat.send_message(system_prompt)
    st.session_state.system_prompt = system_prompt
    system_prompt = f"""
أنتَ “ليجال ميت”، مساعدٌ ذكِيٌّ مختصٌّ في العقود القانونيّة باللغة العربيّة.

1. قسّم النصّ العقديّ إلى عناوين وبنود مرقّمة.
2. بالنسبة لكلّ بند:
   a) ألخِصه بلغةٍ عربيّةٍ فصحىٍ واضحة وبسيطة ودقيقة.
   b) حدّد النقاط الضعيفة أو المبهمة.
   c) صنّف خطورة كلّ نقطة: (منخفضة / متوسّطة / عالية).
   d) قدّم توصيةً أو توصيتين عمليّتين.

التنسيق بدقة كما يلي:

البند 1:
• الملخص: …
• نقاط الضعف والمخاطر:

1. … — الخطورة: (منخفضة/متوسّطة/عالية)
2. … — الخطورة: (منخفضة/متوسّطة/عالية)
• التوصيات:
3. …
4. …

## نص العقد:
---
{st.session_state.contract_text}
---
"""
    st.session_state.chat.send_message(system_prompt)  # send system prompt (no author arg)

# 5. Compute Embeddings & Build FAISS Index Compute Embeddings & Build FAISS Index
if st.session_state.index is None:
    st.session_state.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    st.session_state.embeddings = st.session_state.embed_model.encode(
        st.session_state.chunks, show_progress_bar=True
    )
    dim = st.session_state.embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(st.session_state.embeddings))
    st.session_state.index = index
    st.session_state.chunk_map = {i: c for i, c in enumerate(st.session_state.chunks)}
    st.success("Embeddings generated and FAISS index built.")

# Display existing chat history
for entry in st.session_state.history:
    st.chat_message(entry["role"]).write(entry["content"])

# 5. Action Buttons & Chat Input
# Define buttons and record selection in session_state
col1, col2, col3, col4 = st.columns(4)
if col1.button("🔍 تحليل مبسط للعقد", key="btn_simple"):
    st.session_state.selected_action = "simple"
if col2.button("👥 أطراف العقد", key="btn_parties"):
    st.session_state.selected_action = "parties"
if col3.button("📄 نوع العقد", key="btn_type"):
    st.session_state.selected_action = "type"
if col4.button("🧐 تحليل تفصيلي للعقد", key="btn_detailed"):
    st.session_state.selected_action = "detailed"

# Determine the prompt based on selected action
generate_prompt = None
action = st.session_state.get("selected_action")
if action == "simple":
    generate_prompt = "حلل لي هذا العقد بلغة عربية فصحى بسيطة وسهلة الفهم."
elif action == "parties":
    generate_prompt = "ما هي الأطراف المشاركة في هذا العقد وما أدوارهم؟"
elif action == "type":
    generate_prompt = "ما نوع هذا العقد (مثلاً: بيع، إيجار، خدمات) ولماذا؟"
elif action == "detailed":
    # use the full system prompt for detailed analysis
    generate_prompt = "اريد تحليل تفصيلي لهذا العقد"
    generate_prompt = st.session_state.system_prompt

# Always provide a chat input for custom questions
custom_q = st.chat_input("📩 يمكنك كتابة سؤالك الخاص هنا…", key="chat_input")

# Final user_question: prefer button action, else custom input
user_question = generate_prompt if generate_prompt else custom_q

if user_question:
    # Do not echo the detailed analysis system_prompt in chat
    if action != "detailed":
        # Record and display user message
        st.session_state.history.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
    else:
        # For detailed analysis, only record to history without GUI echo
        st.session_state.history.append({"role": "user", "content": "<تحليل تفصيلي للعقد>"})

    # 6. Retrieve all chunks in order Retrieve all chunks in order Retrieve all chunks in order
    total_chunks = len(st.session_state.chunks)
    q_vec = st.session_state.embed_model.encode([user_question])
    distances, indices = st.session_state.index.search(
        np.array(q_vec), k=total_chunks
    )
    all_indices = sorted(indices[0])
    retrieved = [st.session_state.chunk_map[idx] for idx in all_indices]
    context = "".join(retrieved)

    # 7. Build System Prompt + Context
    system_prompter = (
        "أنتَ ليجال ميت مساعد قانوني ذكي مختصّ بالعقود باللغة العربية. "
        "استعن بالسياق التالي للإجابة بدقة وإيجاز."
    )
    prompt = f"""
{system_prompter}

## مقتطفات من العقد:
{context}

## السؤال:
{user_question}
"""

    # 8. Send to Gemini with streaming and display
    # Send the user question only; system prompt is preloaded
    assistant_chat = st.session_state.chat
    reply_content = ""
    with st.chat_message("assistant"):
      placeholder = st.empty()
       
      for chunk in assistant_chat.send_message_stream(prompt):
    # Guard against None
        text_piece = chunk.text or ""
        reply_content += text_piece
        placeholder.write(reply_content)

    
    # Save assistant reply to history
    st.session_state.history.append({"role": "assistant", "content": reply_content})
    st.session_state.selected_action = None  # 🔁 Reset action
