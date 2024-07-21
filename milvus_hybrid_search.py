from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, WeightedRanker, connections
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

model = ChatGroq(model_name="Llama3-8b-8192")
embd = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

def setup_milvus_retriever(all_texts, embd, dense_dim):
    CONNECTION_URI = "database/milvus_hybrid_search.db"
    connections.connect(uri=CONNECTION_URI)

    sparse_embedding_func = BM25SparseEmbedding(corpus=all_texts)
    sparse_embedding_func.embed_query(all_texts[1])

    collection = Collection(name="text_embeddings")
    
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "IP", "params": {}}
    
    retriever = MilvusCollectionHybridSearchRetriever(
        collection=collection,
        rerank=WeightedRanker(0.5, 0.5),
        anns_fields=["embeddings", "sparse_vector"],
        field_embeddings=[embd, sparse_embedding_func],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=3,
        text_field="text",
    )
    
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=model, retriever=compression_retriever
    )
    
    return chain
