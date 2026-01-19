# embedding test

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")
embeddings = model.encode(["Hello, world!"])

print(embeddings)
