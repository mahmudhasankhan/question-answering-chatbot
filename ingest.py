import argparse
import os
import pdfplumber
import re
import pinecone

from typing import Callable, List, Tuple, Dict
from dotenv import load_dotenv, find_dotenv
from PyPDF4 import PdfFileReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone


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


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser()

    # load file path
    parser.add_argument("--pdf",
                        type=str,
                        default="",
                        help="Enter the pdf file name")
    parser.add_argument("--db",
                        type=str,
                        default=None,
                        help="Enter pinecone index name")

    args = parser.parse_args()

    if args.db is None:
        raise Exception("Pinecone index name must not be none!")

    raw_pages, metadata = parse_pdf(os.path.join(os.getcwd(),
                                                 args.pdf))
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # HuggingFaceEmbeddings

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)

    # initialize pinecone
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
                  environment=os.environ['PINECONE_ENV'])

    print(f"Creating new index {args.db}")
    pinecone.create_index(name=args.db,
                          dimension=embeddings.client[1].word_embedding_dimension,
                          metric='cosine',
                          pods=1,
                          replicas=1,
                          pod_type="p1")
    Pinecone.from_documents(document_chunks, embeddings, index_name=args.db)
