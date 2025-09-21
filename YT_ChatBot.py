import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="YouTube RAG", page_icon="🎥")

# Main heading
st.title("🎥 YouTube RAG Chat")

# Input boxes
video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g., Gfr50f6ZBvo")
question = st.text_input("Enter your question:", placeholder="e.g., Can you summarize the video?")

# Button
if st.button("Get Answer"):
    if video_id and question:
        try:
            with st.spinner("Processing..."):
                # Your exact code - Step 1a: Document Ingestion
                try:
                    ytt_api = YouTubeTranscriptApi()
                    transcript_list = ytt_api.fetch(video_id, languages=["en"])
                    transcript = " ".join(chunk.text for chunk in transcript_list)
                except TranscriptsDisabled:
                    st.error("No captions available for this video.")
                    st.stop()

                # Your exact code - Step 1b: Text Splitting
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                # Your exact code - Step 1c & 1d: Embedding Generation and Vector Store
                embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embedding)

                # Your exact code - Step 2: Retrieval
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # Your exact code - Step 3: Augmentation
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                prompt = PromptTemplate(
                    template="""
                      You are a helpful assistant.
                      Answer ONLY from the provided transcript context.
                      If the context is insufficient, just say you don't know.

                      {context}
                      Question: {question}
                    """,
                    input_variables = ['context', 'question']
                )

                # Your exact code - Building a Chain
                def format_docs(retrieved_docs):
                    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    return context_text

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser

                # Your exact code - Step 4: Generation
                answer = main_chain.invoke(question)

            # Display answer
            st.success("Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter both video ID and question.")