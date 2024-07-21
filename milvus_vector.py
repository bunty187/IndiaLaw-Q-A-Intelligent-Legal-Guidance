from langchain_milvus.vectorstores import Milvus

from langchain_huggingface import HuggingFaceEmbeddings

embd = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

def setup_milvus_database(all_texts, embd, dense_dim):
    URI = "/content/drive/MyDrive/rag_with_raptor/database/milvus_rag.db"
    vector_db = Milvus.from_texts(
        texts=all_texts,
        embedding=embd,
        connection_args={"uri": URI},
    )
    return vector_db
