import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain.schema import Document
from chroma_vector_aggregator import aggregate_embeddings

class TestEmbeddings(FakeEmbeddings):
    def __init__(self, size: int = 10):
        super().__init__(size=size)

    def embed_documents(self, texts):
        return [np.random.rand(self.size).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(self.size).tolist()

def create_test_chroma_collection():
    embeddings = TestEmbeddings(size=10)
    documents = [
        Document(page_content="Test document 1", metadata={"id": "group1"}),
        Document(page_content="Test document 2", metadata={"id": "group1"}),
        Document(page_content="Test document 3", metadata={"id": "group2"}),
        Document(page_content="Test document 4", metadata={"id": "group2"}),
    ]
    return Chroma.from_documents(documents, embeddings)

def test_chroma_aggregator():
    print("Creating test Chroma collection...")
    chroma_collection = create_test_chroma_collection()
    
    print("Initial collection size:", len(chroma_collection._collection.get(include=['embeddings'])['ids']))
    
    print("Aggregating embeddings...")
    aggregated_collection = aggregate_embeddings(
        chroma_collection=chroma_collection,
        column_name="id",
        method="average"
    )
    
    all_data = aggregated_collection._collection.get(include=['embeddings', 'metadatas', 'documents'])
    print("Aggregated collection size:", len(all_data['ids']))
    
    # Print detailed information about the aggregated collection
    print("Detailed information about aggregated collection:")
    print(f"Number of documents: {len(all_data['documents'])}")
    print(f"Number of embeddings: {len(all_data['embeddings'])}")
    print(f"Number of metadatas: {len(all_data['metadatas'])}")
    print("Metadatas:")
    for metadata in all_data['metadatas']:
        print(metadata)
    
    # Verify that the aggregation reduced the number of documents
    assert len(all_data['ids']) == 2, "Aggregation should result in 2 documents"
    
    print("Testing similarity search...")
    results = aggregated_collection.similarity_search("Test query", k=2)
    assert len(results) == 2, "Similarity search should return 2 results"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_chroma_aggregator()