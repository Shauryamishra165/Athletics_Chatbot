from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain.retrievers.multi_query import MultiQueryRetriever
from pdf2image import convert_from_path
import pytesseract

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)

    # Iterate through all the pages and extract text
    extracted_text = ''
    for page_number, page_data in enumerate(pages):
        # Perform OCR on the image
        text = pytesseract.image_to_string(page_data)
        extracted_text += f"Page {page_number + 1}:\n{text}\n"
    return extracted_text

def extract_text_from_excel(file_path):
    # Read the Excel file
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
        all_text = ""
        
        # Iterate through each sheet
        for sheet_name, sheet_data in excel_data.items():
            all_text += f"Sheet: {sheet_name}\n"
            
            # Iterate through each row and column in the sheet
            for _, row in sheet_data.iterrows():
                all_text += " | ".join(str(cell) for cell in row) + "\n"
        
        return all_text

    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, batch_size=100):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_embeddings = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        text_embeddings.extend(zip(batch, batch_embeddings))
    
    vector_store = FAISS.from_embeddings(text_embeddings, embedding=embeddings)
    vector_store.save_local("faiss_index_Athletics")
    return vector_store


def user_input(user_question):
    prompt_template = """
    You are IIT Jodhpur's Athletic society chatbot which requires to answer the questions based on the context provided. Do not answer from outside the context and answer respectfully.
    Context:\n{context}?\n
    Question:\n{question} + Explain in detail.\n
    Answer:
    """
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    new_db = FAISS.load_local("faiss_index_Athletics", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db

    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )

    # docs = mq_retriever.get_relevant_documents(query=user_question)
   
    docs = new_db.similarity_search(query=user_question, k=10)  # Get similar text from the database with the query
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # save_permanent_answer(user_question, response)
    return response , docs


def load_in_db():
    url_text_chunks = []
    Folder_name = 'Data'
    # Get a list of all PDF files in the folder
    files_athletics = [f for f in os.listdir(Folder_name) if f.endswith('.pdf')]

    # Process each PDF file
    for pdf in files_athletics:
        pdf_path = os.path.join(Folder_name, pdf)
        article_text = extract_text_from_pdf(pdf_path)
        text_chunks = get_text_chunks(article_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"Pdf Link : {pdf}\n{chunk}")
    excel_sheets = [f for f in os.listdir(Folder_name) if f.endswith('.xlsx')]
    for url in excel_sheets :
        article_text = extract_text_from_excel(url)
        text_chunks = get_text_chunks(article_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"Excel sheet : {url}\n{chunk}")
    get_vector_store(url_text_chunks)

def main():
    load_in_db()

if __name__ == "__main__":
    main()

