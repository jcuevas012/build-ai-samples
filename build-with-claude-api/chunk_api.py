import re
from embedding_apy import generate_embedding
from vector_db_api import VectorIndex
from bm25_api import BM25Index

# Chunk by a set number of charactesr
def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)

        start_idx = (
            end_idx - chunk_overlap if end_idx < len(text) else len(text)
        )

    return chunks



def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))

        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))

        start_idx += max_sentences_per_chunk - overlap_sentences

        if start_idx < 0:
            start_idx = 0

    return chunks


# Chunk by section
def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)



def setup_chunk_to_vector_store():

    store = VectorIndex()

    with open("./report.md", "r") as f:
        text = f.read()

    chunks = chunk_by_section(text)
    embeddings = [generate_embedding(chunk) for chunk in chunks]

    for chunk, embedding in zip(chunks, embeddings):
        store.add_vector(embedding, {"content": chunk})
        
    return store



def setup_chunk_to_vector_store_with_bms25():

    store = BM25Index()

    with open("./report.md", "r") as f:
        text = f.read()

    chunks = chunk_by_section(text)

    for chunk in chunks:
        store.add_document({"content": chunk})
        
    return store