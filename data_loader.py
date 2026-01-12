from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

# Load all supported files from the given directory and convert them into LangChain Document objects.
# data_dir : str
        #Path to the folder containing documents
# RETURNS: List[Any]
        #List of LangChain Document objects

def load_all_documents(data_dir: str) -> List[Any]:

    # Convert relative path (e.g., "data") to absolute path
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    # This list will store ALL loaded documents
    documents = []

    # To Load Text Files
    # Use for notes, logs, scraped text, or plain documentation.
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")

    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file),encoding="utf-8")
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")
    
    # To Load Pdf Files
    # Use : manuals, resumes, or reports in PDF format.
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded_docs = loader.load()  # Each page = one Document
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # To Load CSV Files
    # # Use when your data is tabular (datasets, logs, reports).
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # To Load Excel File
    # Use for business reports, financial data, or structured sheets. Unstructured loader handles mixed layouts.
    xlsx_files = list(data_path.glob("**/*.xlsx"))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files")

    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # To Load Word File
    # Use for resumes, project docs, policies, agreements.
    docx_files = list(data_path.glob("**/*.docx"))
    print(f"[DEBUG] Found {len(docx_files)} Word files")

    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # To Load JSON File
    # Use for API responses, config files, logs, or structured text.
    json_files = list(data_path.glob("**/*.json"))
    print(f"[DEBUG] Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")
    
    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_all_documents("data")

    print(f"\nLoaded {len(docs)} documents successfully.")

    # Print first document for verification
    if docs:
        print("\nSample Document:")
        print(docs[0])
