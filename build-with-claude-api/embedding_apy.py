from dotenv import load_dotenv
import voyageai

load_dotenv()

client = voyageai.Client()

def generate_embedding(text, model="voyage-3-large", input_type="query"):
    response = client.embed(
        [text],
        model=model,
    )
    return response.embeddings[0]


# Embedding Generation
def generate_embedding_fn(chunks, model="voyage-3-large", input_type="query"):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]
    result = client.embed(input, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]