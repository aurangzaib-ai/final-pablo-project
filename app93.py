import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
import requests
import time
from dotenv import load_dotenv
import os

# --- Load .env ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

# --- UI Setup ---
st.set_page_config(page_title="AI PDF Auto-Filler & Q/A", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #3a0ca3 0%, #9d4edd 50%, #c77dff 100%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, .stTextInput label, .stTextArea label {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI PDF Auto-Filler with Q/A")
st.write("Upload scanned PDF, AI fills missing values, download clean PDF, and ask questions.")

# --- Functions ---

def pdf_to_text(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    full_text = ""
    for img in images:
        full_text += pytesseract.image_to_string(img) + "\n"
    return full_text

def groq_fill_missing(text):
    prompt = f"""
You are an expert form assistant. The scanned form text below has missing values like 'N/A', 'nan', or '---'.
Please fill missing values realistically, preserving format.

--- FORM START ---
{text}
--- FORM END ---
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 1300}

    for _ in range(3):
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            time.sleep(2)
        except:
            time.sleep(2)
    return "AI filling failed."

def generate_pdf(filled_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Final Clean Form (AI-filled)")
    y -= 30
    c.setFont("Helvetica", 10)
    lines = filled_text.split("\n")
    for line in lines:
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

def groq_answer_question(filled_text, question):
    prompt = f"""
You are an AI assistant reading a filled scanned form.
Form content:
{filled_text}
Answer this question accurately:
{question}
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.4, "max_tokens": 600}

    for _ in range(3):
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            time.sleep(2)
        except:
            time.sleep(2)
    return "AI failed to answer."

# --- Streamlit Session State ---
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "filled_text" not in st.session_state:
    st.session_state.filled_text = ""
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- UI Flow ---
uploaded_file = st.file_uploader("Upload scanned PDF form", type="pdf")

if uploaded_file:
    if not st.session_state.ocr_text:
        with st.spinner("Performing OCR..."):
            st.session_state.ocr_text = pdf_to_text(uploaded_file)

    st.subheader("OCR Extracted Text")
    st.text_area("OCR Text", st.session_state.ocr_text, height=250)

    if st.button("AI: Fill Missing Fields"):
        with st.spinner("AI is filling missing values..."):
            st.session_state.filled_text = groq_fill_missing(st.session_state.ocr_text)

    if st.session_state.filled_text:
        st.subheader("AI-Filled Form Text")
        st.text_area("Filled Text", st.session_state.filled_text, height=250)

        pdf_buffer = generate_pdf(st.session_state.filled_text)
        st.download_button("Download Clean Filled PDF", pdf_buffer, file_name="AI_Filled_Form.pdf", mime="application/pdf")

        st.markdown("Ask questions about the filled form:")
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
