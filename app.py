import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2
from gtts import gTTS
from io import BytesIO
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CONFIG & PROFESSIONAL STYLING (Person 1)

st.set_page_config(page_title="MedGraph AI Pro", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #007bff; color: white; border: none; height: 3em; }
    .stButton>button:hover { background-color: #0056b3; }
    </style>
    """, unsafe_allow_html=True)


# 2. SECURITY & SECRETS (The Fix!)

# 🚨 NEVER hardcode keys in a real app. For this prototype, paste them here just before running.
GEMINI_API_KEY = ""
PINECONE_API_KEY = ""

genai.configure(api_key=GEMINI_API_KEY)


# 3. LOAD THE HEAVY TOOLS (Cached so it's fast)

@st.cache_resource
def load_tools():
    # The Brain
    llm = genai.GenerativeModel('gemini-2.5-flash')
    # The Library
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("medgraph")
    # The Translator
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, index, embedder

llm, index, embedder = load_tools()


# 4. THE LANGGRAPH BRAIN (Person 3)

class AgentState(TypedDict):
    question: str
    retrieved_context: str
    draft_answer: str
    final_verification: str
    attempts: int

def researcher_agent(state: AgentState):
    query_vector = embedder.encode(state['question']).tolist()
    
    search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    context = "\n\n".join([match['metadata']['text'] for match in search_results['matches']])
    
    prompt = f"""
    You are answering a medical question. 
    Question: '{state['question']}'
    Read this text extracted from a medical PDF: '{context}'
    Write a brief, accurate answer based ONLY on the text.
    """
    response = llm.generate_content(prompt)
    return {"draft_answer": response.text, "retrieved_context": context}

def verifier_agent(state: AgentState):
    prompt = f"""
    Question: '{state['question']}'
    Draft Answer: '{state['draft_answer']}'
    Original PDF Text: '{state['retrieved_context']}'
    
    Is the draft answer 100% supported by the text?
    If yes, reply with exactly: "APPROVED: [paste the answer here]"
    If no, reply with exactly: "REJECTED"
    """
    response = llm.generate_content(prompt)
    return {"final_verification": response.text, "attempts": state['attempts'] + 1}

def router(state: AgentState):
    if state["attempts"] >= 3: return "end"
    if "APPROVED" in state["final_verification"]: return "end"
    return "try_again"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("verifier", verifier_agent)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "verifier")
workflow.add_conditional_edges("verifier", router, {"end": END, "try_again": "researcher"})
medgraph_brain = workflow.compile()


# 5. VOICE ASSISTANT HELPER (Person 1)

def speak_text(text):
    clean_text = text.replace('APPROVED:', '').replace('*', '').strip()
    tts = gTTS(text=clean_text, lang='en', tld='co.uk') # Changed to English for standard medical text
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp

# 6. THE "AUTOMATIC JUICER" (Person 2 Integration)
def process_new_pdf(file):
    # 1. Extract text
    reader = PyPDF2.PdfReader(file)
    raw_text = "".join([p.extract_text() for p in reader.pages])
    
    # 2. Chop it up
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    
    # 3. Wipe the old patient data and upload the new patient data
    index.delete(delete_all=True) # Clear old memory
    
    vectors_to_upload = []
    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        vectors_to_upload.append({"id": f"chunk_{i}", "values": vector, "metadata": {"text": chunk}})
        
    index.upsert(vectors=vectors_to_upload)
    return raw_text

# 7. THE USER INTERFACE (Streamlit Website)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2869/2869811.png", width=80)
    st.title("Control Panel")
    
    # The moment a doctor drops a file here, it triggers the "Automatic Juicer"
    pdf = st.file_uploader("Upload New Patient Report", type="pdf")
    
    if pdf:
        with st.spinner("Analyzing and updating database..."):
            text = process_new_pdf(pdf)
            st.session_state.ready = True
            st.success("Database Updated! Ready for questions. ✅")

st.title("👨‍⚕️ MedGraph AI Voice Assistant")
st.caption("Hallucination-Free Medical Analysis Powered by LangGraph & Pinecone")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask about the uploaded report...")

if query:
    if not st.session_state.get("ready"):
        st.warning("⚠️ Please upload a patient report in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.write(query)

        with st.chat_message("assistant", avatar="👨‍⚕️"):
            with st.status("🧠 MedGraph is analyzing...", expanded=True) as status:
                st.write("🕵️‍♂️ Researcher is scanning the PDF...")
                
                # Pass the question to our custom Brain instead of raw Gemini!
                result = medgraph_brain.invoke({"question": query, "attempts": 1})
                
                st.write("🧐 Verifier is double-checking for hallucinations...")
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
            # Clean up the output string (Notice how this is aligned with 'with st.status')
            final_answer = result["final_verification"]
            display_answer = final_answer.replace("APPROVED:", "").strip()
                
            if "REJECTED" in final_answer or "Maximum attempts" in final_answer:
                    display_answer = "I'm sorry, I cannot verify the answer to that question based strictly on the uploaded document."
                
            st.write(display_answer)
                
                # Generate the Voice!
            audio_fp = speak_text(display_answer)
            st.audio(audio_fp, format='audio/mp3')
            
            st.session_state.messages.append({"role": "assistant", "content": display_answer})
            
