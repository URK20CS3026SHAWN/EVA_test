#Sample PDFs to be loaded
#pdf ='RAG/pdfs/ACM.pdf'
#pdf ='RAG/pdfs/G20.pdf'
#pdf ='RAG/pdfs/GenCPro.pdf'
pdf ='Input_pdfs/KITS2024AdBro.pdf'

#Load the PDF using PyPDFloader
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader(pdf)
data = loader.load()

#Use Langchain's RecursiveCharacterTextSplitter to split the document into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
all_splits = text_splitter.split_documents(data)
#print(all_splits)  #View the Chunks for a better understanding

#Use a custom encased SentenceTransformerEmbedding Function for inferencing on Mac.
from CustomEmbedding import SentenceTransformerEmbeddingFunction # type: ignore
embedding = SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/msmarco-bert-base-dot-v5',device="mps")

# Adding the documents to the vectorstore
from langchain.vectorstores.chroma import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding,
                                    collection_name="edustore",
                                    persist_directory="RAG/Vectorstores/.chromaVS")