import streamlit as st
from openai import OpenAI
import asyncio
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_utils import VectorDbWithBM25, LangchainLlms, RagFusion
# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below for GPT to reference and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
uploaded_file = st.file_uploader(
        "Upload a document (.pdf)", type=("pdf")
    )
if not openai_api_key or not uploaded_file:
    st.info("Please add your OpenAI API key and file to continue.", icon="🗝️")
    
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    # uploaded_file = st.file_uploader(
    #     "Upload a document (.txt or .md or .pdf)", type=("txt", "md", "pdf")
    # )

    message_type = st.selectbox("Select the type of message:", ["first-time connecting", "thank you notes for interview", "invitation for coffee chat"])
    relationship = st.selectbox("Select the type of message:", ["stranger", "UCLA alumni", "UCB alumni"])
    subject = st.text_input("Enter the customer name (max 20 characters):")[:20]
    @st.cache_resource
    def initialize_rag_agent():
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
        # Initialize vector database
        vector_db = FAISS(
            embedding_function=embeddings,
            index=IndexFlatL2(1536),
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            #file_name = uploaded_file.name
        doc_loader = PyPDFLoader(temp_file)
        pages = doc_loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        vector_db.add_documents(docs)
        bm25_corpus = [doc.page_content for doc in docs]
        vector_db_with_bm25 = VectorDbWithBM25(vector_db = vector_db, bm25_corpus = bm25_corpus)
        langchain_llm = LangchainLlms()
        rag = RagFusion(vector_store=vector_db_with_bm25, 
                        llm=langchain_llm.get_llm("OpenAI", 
                        openai_api_key=openai_api_key, 
                        model_name="gpt-4o-mini").llm)
        return rag
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        rag_agent = initialize_rag_agent()
        st.write(rag_agent)

        template = 'Provide LinkedIn message template suitable for {message_type} with a {relationship} , and fill in my name as Nei Fang and the subject name as {subject}'
        recommended_query = template.format(message_type = message_type, relationship = relationship, subject= subject)
        user_query = st.text_area("Modify the recommended query if needed:", value=recommended_query)
        if st.button("Confirm"):
            async def run_query():
                return await rag_agent.arun(user_query, rewrite_original_query=False)
            try:
                result = asyncio.run(run_query())
                st.write("Generated Message:")
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.write("Please upload a file to initialize the RAG agent.")
        rag_agent = initialize_rag_agent()
