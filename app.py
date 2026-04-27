from langchain_chroma import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import re
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
import streamlit as st 
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)
with open("data\\100 Most Frequently Asked Machine L.txt", encoding="utf-8") as f:
    text = f.read()

parts = re.split(r"(?=\n\d+\.\s)", text)
qa_chunks = []
for part in parts:
    part.strip()
    if len(part)<50:
        continue
    qa_chunks.append(
        Document(
                page_content=part,
                 metadata ={
                        "source":"ml_interview_qa",
                        "type":"question_answer"
                           }
                ))
print(len(qa_chunks))

with open("data\Machine Learning Cheatsheet (Compan.txt", encoding="utf-8") as f:
    text = f.read()

sections = re.split(r"(?=(?:^|\n)\d+\.\s)", text)
cheatsheet_chunks = []
for sec in sections:
    sec = sec.strip()
    if len(sec)<30:
        continue
    cheatsheet_chunks.append(Document(
                page_content=sec,
                metadata = {
                            "source":"ml_cheatsheet",
                            "type":"section"
                           }
    ))
print(len(cheatsheet_chunks))
all_chunks = qa_chunks + cheatsheet_chunks
embeddings = HuggingFaceEmbeddings(
             model_name = "BAAI/bge-base-en-v1.5"
)
db = Chroma(
     persist_directory = "./chroma_db",
     embedding_function=embeddings
)
if db._collection.count() == 0:
    db.add_documents(
        all_chunks
    )
    print("Knowledge Base Loaded!")

    
load_dotenv("keys.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
      model="llama-3.1-8b-instant",
      api_key=api_key
)
memory = ConversationBufferWindowMemory(
         k=3,
         memory_key="chat_history",
         return_messages=True

)

# chroma retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":5,
        "fetch_k":10
    }
)
# UI

st.title("AI Interview Prep Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = [] 

with st.sidebar:
    st.header("Loaded Documents")
    st.write("ML Interview Questions")
    st.write("ML Cheatsheet")
    if "loaded_docs" not in st.session_state:
        st.session_state.loaded_docs = []
    
    uploaded_file = st.file_uploader(
        "Add Document",
        type=["txt","pdf" ]
        )
if uploaded_file is not None:
    if uploaded_file.name in st.session_state.loaded_docs:

            st.sidebar.warning(
                "Document already loaded."
            )
    else:

        path = f"data/{uploaded_file.name}"

        with open(path,"wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        if path.endswith(".txt"):
            docs = TextLoader(path).load()

        elif path.endswith(".pdf"):
            docs = PyPDFLoader(path).load()

        new_chunks = splitter.split_documents(
            docs
        )

        db.add_documents(
            new_chunks
        )
        retriever = db.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                    "k":5,
                    "fetch_k":10
                    }
        )
        print(len(new_chunks))
        test = db.similarity_search(
            "What is Label Smoothing?",
             k=3
        )

        
        st.sidebar.success(
            "Document added to vector DB."
            )
        
if st.button("Clear Chat"):
    st.session_state.messages=[]
            #memory.clear()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Say something")
if query:
    st.session_state.messages.append(
                {"role":"user", "content":query}
            )

    docs = retriever.invoke(query)
    context = "\n\n".join(
                            i.page_content for i in docs
                )
    history = memory.load_memory_variables({})[
                            "chat_history"
                ]
    prompt = f""" 
                            You are an AI interview assistant. Provide interview style answers.

                            Use chat history for conversational context.
                            You must answer ONLY using the retrieved context below.

                            If the answer is not explicitly stated in the context,
                            reply EXACTLY:

                            INSUFFICIENT_CONTEXT

                            Do not use prior knowledge.
                            Do not guess.
                            Do not answer from memory.
                            Answer ONLY using retrieved context.
                            If not in context, say insufficient context.

                            Chat_history:{history}
                            Context:{context}
                            Question:{query}
                            
                    """
    
    response = llm.invoke(prompt)
    answer = response.content
    memory.save_context(
                    {"input": query},
                    {"output": answer}
                )
    st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":answer
                }
            )
    st.rerun()
