import os
import re
import time
import requests
import streamlit as st
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from dotenv import load_dotenv

# ---------------------- Config & ENV ----------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

POPPLER_BIN_PATH = os.getenv("POPPLER_PATH", r"C:\poppler\poppler\Library\bin" if os.name == "nt" else "/usr/bin")

# ---------------------- UI Setup ----------------------
st.set_page_config(page_title="AI PDF Auto-Filler & Q/A", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #3a0ca3 0%, #9d4edd 50%, #c77dff 100%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, .stTextInput label, .stTextArea label, .stDownloadButton {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI PDF Auto-Filler with Q/A")

if not GROQ_API_KEY:
    st.warning("⚠️ `GROQ_API_KEY` missing. .env / secrets me set karein, warna AI steps fail honge.")

# ---------------------- Helpers ----------------------
def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def pdf_to_text(pdf_file) -> str:
    try:
        images = convert_from_bytes(pdf_file.read(), poppler_path=POPPLER_BIN_PATH)
    except Exception as e:
        st.error(f"Poppler/convert error: {e}")
        return ""
    full_text = ""
    for img in images:
        page_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
        full_text += clean_text(page_text) + "\n\n"
    return full_text

def call_groq(prompt: str, temperature: float, max_tokens: int):
    if not GROQ_API_KEY:
        return None, "Missing GROQ_API_KEY"

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=data, timeout=30
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content'].strip()
                return content, None
            else:
                st.error(f"[Groq Error {resp.status_code}] {resp.text}")
                time.sleep(2)
        except Exception as e:
            st.error(f"[Groq Exception] {e}")
            time.sleep(2)

    return None, "Failed after retries"

def call_groq_chunk(chunk):
    prompt = f"""
You are an expert form assistant. The scanned form text below has missing values like 'N/A', 'nan', or '---'.
Please fill missing values realistically, preserving format.

--- FORM START ---
{chunk}
--- FORM END ---
"""
    content, err = call_groq(prompt, temperature=0.2, max_tokens=1300)
    return content if content else "AI filling failed."

def groq_fill_missing(text: str) -> str:
    if len(text) > 8000:
        st.warning("OCR text too large, splitting into parts...")
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        filled_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            st.info(f"Filling chunk {idx}/{len(chunks)}")
            filled = call_groq_chunk(chunk)
            filled_chunks.append(filled)
        return "\n".join(filled_chunks)
    else:
        return call_groq_chunk(text)

def groq_answer_question(filled_text: str, question: str) -> str:
    prompt = f"""
You are a document information extractor.
Answer the user's question based ONLY on the given form text.
If the answer is not explicitly present, reply "Not available in the form."

--- FORM START ---
{filled_text}
--- FORM END ---

Question: {question}
Answer:
"""
    content, err = call_groq(prompt, temperature=0.0, max_tokens=400)
    return content if content else "AI failed to answer."

def generate_pdf(filled_text: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Final Clean Form (AI-filled)")
    y -= 30
    c.setFont("Helvetica", 10)
    for line in filled_text.split("\n"):
        if not line.strip():
            y -= 12
            continue
        while len(line) > 110:
            c.drawString(40, y, line[:110])
            line = line[110:]
            y -= 12
        c.drawString(40, y, line.strip())
        y -= 12
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------- Session ----------------------
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "filled_text" not in st.session_state:
    st.session_state.filled_text = ""
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# ---------------------- UI Flow ----------------------
uploaded_file = st.file_uploader("Upload scanned PDF form", type="pdf")

if uploaded_file:
    if not st.session_state.ocr_text:
        with st.spinner("Performing OCR..."):
            st.session_state.ocr_text = pdf_to_text(uploaded_file)

    st.subheader("OCR Extracted Text")
    st.text_area("OCR Text", st.session_state.ocr_text, height=250)

    if st.button("AI: Fill Missing Fields", disabled=not st.session_state.ocr_text):
        with st.spinner("AI is filling missing values..."):
            st.session_state.filled_text = groq_fill_missing(st.session_state.ocr_text)

    if st.session_state.filled_text:
        st.subheader("AI-Filled Form Text")
        st.text_area("Filled Text", st.session_state.filled_text, height=250)

        pdf_buffer = generate_pdf(st.session_state.filled_text)
        st.download_button("Download Clean Filled PDF", pdf_buffer, file_name="AI_Filled_Form.pdf", mime="application/pdf")

        st.markdown("#### Ask questions about the filled form:")
        question = st.text_input("Your question:")

        if question:
            with st.spinner("Getting answer..."):
                answer = groq_answer_question(st.session_state.filled_text, question)
                st.session_state.qa_history.append({"question": question, "answer": answer})

        if st.session_state.qa_history:
            st.subheader("Q/A History")
            for qa in reversed(st.session_state.qa_history):
                st.markdown(f"**Q:** {qa['question']}")
                st.info(f"**A:** {qa['answer']}")