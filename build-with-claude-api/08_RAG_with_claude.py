from embedding_apy import generate_embedding_fn
from vector_db_api import VectorIndex
from bm25_api import BM25Index
from retriever_api import Retriever
from chunk_api import chunk_by_section

vector_index = VectorIndex(embedding_fn=generate_embedding_fn)
bm25_index = BM25Index()

retriever = Retriever(vector_index, bm25_index)


with open("./report.md", "r") as f:
    text = f.read()

chunks = chunk_by_section(text)

retriever.add_documents([{"content": chunk} for chunk in chunks])


results = retriever.search("What happened with INC-2023-Q4-011?", k=3)

for doc, score in results:
    print(f"Score: {score:.4f}, Content: {doc['content'][0:200]}...")  # Print the first 200 characters of the content

# store = setup_chunk_to_vector_store()
# store_bm25 = setup_chunk_to_vector_store_with_bms25()

# # Search for the 2 most relevant documents based on the user query embedding
# results = store.search(user_embedding_query, 2)

# for doc, score in results:
#     print(f"Score: {score:.4f}, Content: {doc['content']}")

# print("\n========= BM25 Search Results =========\n")
# bm25_results = store_bm25.search("What happened with INC-2023-Q4-011?", 2)

# for doc, score in bm25_results:
#     print(f"Score: {score:.4f}, Content: {doc['content']}")
