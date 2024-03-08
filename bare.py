from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

import os
import io

# Set your NVIDIA API key
nvapi_key = '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'
os.environ["NVIDIA_API_KEY"] = nvapi_key

# Function to store document embeddings
def storeDocEmbeds(file):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(corpus)
    embedding = NVIDIAEmbeddings(model="nvolveqa_40k", model_type='passage')

    vectors = FAISS.from_texts(chunks, embedding)
    return vectors

# Function for conversational chat
def conversational_chat(query, qa, history):
    result = qa({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Main function to run the chatbot
def main():
    history = []
    print("PDFChat :")

    file_path = input("Enter the path to your PDF file: ")
    
    with open(file_path, 'rb') as file:
        vectors = storeDocEmbeds(file)
        qa = ConversationalRetrievalChain.from_llm(
            ChatNVIDIA(model="mixtral_8x7b"),
            retriever=vectors.as_retriever(),
            return_source_documents=True
        )

    print("Welcome! You can now ask any questions.")

    while True:
        user_input = input("Query: ")
        if user_input.lower() == 'exit':
            print("Exiting the chat.")
            break
        output = conversational_chat(user_input, qa, history)
        print("Bot:", output)

if __name__ == "__main__":
    main()
