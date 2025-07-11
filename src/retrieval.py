from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.load_and_preprocess_data import preprocess_data
import warnings
warnings.simplefilter("ignore")

CHROMA_DB_DIR = "chromadb_store"

df = preprocess_data()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

def populate_chromadb():
    if chroma._collection.count() == 0:
        BATCH_SIZE = 5000
        documents = df["clean_statement"].tolist()
        metadatas = [{"statement": row["statement"], "label": row["label"]} for _, row in df.iterrows()]
        print("Populating ChromaDB with embeddings...")

        for i in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[i:i + BATCH_SIZE]
            batch_meta = metadatas[i:i + BATCH_SIZE]
            chroma.add_texts(texts=batch_docs, metadatas=batch_meta)

        print("ChromaDB population complete!")

populate_chromadb()

def retrieve_similar_claim(query, threshold):
    results = chroma.similarity_search_with_score(query, k=5) #checking for the closest claims (lowest cosine distance)
    
    filtered_results = [claim for claim, score in results if score < threshold]
    
    if filtered_results:
        return filtered_results[0].metadata
    
    return None
