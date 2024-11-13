import argparse
import os
import pdfplumber
import re
import logging
import getpass
import time
import pandas as pd

from typing import Callable, List, Tuple, Dict
from dotenv import load_dotenv, find_dotenv
from PyPDF4 import PdfFileReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def extract_matadata_from_pdf(pdf_file) -> dict:

    reader = PdfFileReader(pdf_file)
    metadata = reader.getDocumentInfo()

    return {
        "title": metadata.get("/Title", "").strip(),
        "author": metadata.get("/Author", "").strip(),
        "creation_date": metadata.get("/CreationDate", "").strip()
    }


def extract_pages_from_pdf(pdf_file) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.
    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """

    with pdfplumber.open(pdf_file) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


def parse_pdf(pdf_file) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.
    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if pdf_file is None:
        raise FileNotFoundError("File not found")

    metadata = extract_matadata_from_pdf(pdf_file)
    pages = extract_pages_from_pdf(pdf_file)

    return pages, metadata


# clean text
def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


cleaning_functions = [
    merge_hyphenated_words,
    fix_newlines,
    remove_multiple_newlines
]


def clean_text(pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    doc_chunks = []
    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       separators=[
                                                           "\n\n", "\n", ".", "!", "?", ",", " ", ""],
                                                       chunk_overlap=200)
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


def load_csv(path):
    dataframe = pd.read_csv(path)
    dataframe['stuff'] = dataframe['product details'] + \
        "Product link: " + dataframe['product link']
    texts = dataframe['stuff'].tolist()

    return texts


def main():

    _ = load_dotenv(find_dotenv())

    logging.basicConfig(filename='./logs/ingest.log',
                        encoding='utf-8', level=logging.INFO)

    parser = argparse.ArgumentParser()

    # load file path
    parser.add_argument("--file",
                        type=str,
                        default="",
                        help="Enter the file path")

    args = parser.parse_args()
    #
    # logging.info("Extracting pages and metadata from the pdf")
    # raw_pages, metadata = parse_pdf(os.path.join(os.getcwd(),
    #                                              args.pdf))
    # logging.info("Cleaning extracted pages")
    # cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    # logging.info("Loading documents in chunks from the cleaned pages")
    # document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    logging.info("Initializing hunggingface embedding model configuration")
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)

    # initialize pinecone
    logging.info("Initializing pinecone vectorstore")
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass(
            "Enter your Pinecone API key: ")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = os.environ.get("PINECONE_INDEX")  # change if desired

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    if args.file is None:
        raise Exception("file not found!")
    else:
        texts = load_csv(args.file)
    if index_name not in existing_indexes:
        PineconeVectorStore.from_texts(
            texts, embeddings, index_name=os.environ.get("PINECONE_INDEX"))
    else:
        PineconeVectorStore.from_existing_index(
            os.environ.get("PINECONE_INDEX"), embedding=embeddings)


if __name__ == "__main__":
    main()
