# IndiaLaw-Q-A-Intelligent-Legal-Guidance

IndiaLaw Q&A is a project focused on building an intelligent chatbot for providing legal guidance on the Indian legal system. This repository contains the code and resources required to develop, deploy, and interact with the chatbot.

# Introduction

IndiaLaw Q&A uses powerful AI technology to develop a conversational interface that can initiate and maintain conversations with users, delivering smooth and intuitive legal advice. IndiaLaw Q&A provides exact and contextually relevant responses to legal questions by leveraging the power of Retrieval-Augmented Generation (RAG) with the Recursive Abstractive Processing for Tree Organized Retrieval (RAPTOR) indexing approach and the Milvus vector database. RAPTOR is an innovative and powerful indexing and retrieval strategy for LLMs that takes a bottom-up approach, grouping and summarizing text segments (chunks) to create a hierarchical tree structure. This intelligent chatbot is designed with Streamlit to provide a user-friendly experience for a variety of use cases in the Indian legal system.

# Getting Started
1. Clone the repository:
   ```git clone https://github.com/your-username/IndiaLaw-Q-A-Intelligent-Legal-Guidance.git```
2. Install the libraries:
   ``` pip install the requirements.txt```
3. Load the data:
Here, we used legal books as input data for our analysis. Each PDF file is over 300 pages. The PDF links for these books are provided below.:
* [Family Law](https://lawfaculty.du.ac.in/userfiles/downloads/LLBCM/Ist%20Term_Family%20Law-%20I_LB105_2023.pdf)
* [Administrative Law](https://lawfaculty.du.ac.in/userfiles/downloads/LLBCM/IVth%20Term_Administrative%20Law_LB%20402_2023.pdf)
* [Labour Law](https://www.icsi.edu/media/webmodules/Labour_Laws&_Practice.pdf)

4. Text Cleaning Steps:
   * Convert Text to Lowercase
   * Remove Punctuation and Special Characters
   * Tokenize Text into Words
   * Remove Stopwords
   * Lemmatize Words
5. Create reference document chunks:
Typically for RAG, large texts are broken down into smaller chunks at ingest time. Given a user query, only the most relevant chunks are retrieved, to pass on as context to the LLM. So as a next step, we will chunk up our reference texts before embedding and ingesting them into.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split text by tokens using the tiktoken tokenizer
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", keep_separator=False, chunk_size=400,   chunk_overlap=30
)

def split_texts(texts):
    chunked_texts = []
    for text in texts:
        chunks = text_splitter.create_documents([text])
        chunked_texts.extend([chunk.page_content for chunk in chunks])
    return chunked_texts
```
We use the from_tiktoken_encoder method of the RecursiveCharacterTextSplitter class in LangChain. This way, the texts are split by character and recursively merged into tokens by the tokenizer as long as the chunk size (in terms of number of tokens) is less than the specified chunk size (chunk_size). Some overlap between chunks has been shown to improve retrieval, so we set an overlap of 30 characters in the chunk_overlap parameter. 

6. Define the LLM Model:
   * Here we use the Groq API to access the open-source LLaMA3 model.
   * The Groq API, combined with the powerful capabilities of Llama 3, offers an innovative approach to building and deploying machine learning models.
   * Groq, known for its high-performance AI accelerators, provides an efficient and scalable platform for running complex AI workloads.
   * Llama 3, a state-of-the-art language model, leverages these capabilities to deliver robust natural language processing (NLP) solutions.
  ```
    os.environ['GROQ_API_KEY'] = 'GROQ_API_KEY'
    llm_model = ChatGroq(model_name="Llama3-8b-8192")
```
7. For embedding models, I use SBERT for embeddings:
   * Find the paper for Sentence BERT here: [SBERT](https://arxiv.org/pdf/1908.10084.pdf).
   * BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS). However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT. The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering.
   * Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT.
     ```
     embd = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
     ```
   
8. Define every RAPTOR phase.
   1. Global Clustering with UMAP:
      Reduces the dimensionality of the input embeddings globally using UMAP (Uniform Manifold Approximation and Projection).Returns a numpy array of the embeddings reduced to the specified dimensionality.
   2. Local Clustering with UMAP:
      Performs local dimensionality reduction on the embeddings using UMAP after global clustering. Returns a numpy array of the embeddings reduced to the specified dimensionality.
   3. Determine Optimal Number of Clusters:
      Determines the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model. Returns an integer representing the optimal number of clusters found.
   4. Gaussian Mixture Model Clustering:
      Clusters embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold. Returns a tuple containing the cluster labels and the number of clusters determined.
   5. Perform Clustering:
      Performs clustering by first reducing dimensionality globally, clustering with GMM, and then performing local clustering within each global cluster. Returns a list of numpy arrays, where each array contains the cluster IDs for each embedding.
   6. Generate Embeddings for Texts:
      Generates embeddings for a list of text documents. Returns a numpy array of embeddings for the given text documents.
   7. Embed and Cluster Texts:
      Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels. Returns a DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
   8. Format Texts for Summarization:
      Formats the text documents in a DataFrame into a single string. Returns a single string where all text documents are joined by a specific delimiter.
   9. Embed, Cluster, and Summarize Texts:
      Embeds, clusters, and summarizes a list of texts, returning two DataFrames: one with clusters and one with summaries. Returns a tuple containing two DataFrames: one with clusters and one with summaries.
   10. Recursive Embed, Cluster, and Summarize Texts:
       Recursively embeds, clusters, and summarizes texts up to a specified level or until the number of unique clusters becomes 1. Returns a dictionary where keys are the recursion levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.

8. Build Tree:
   ```
   leaf_texts = docs_texts
   results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
   ```
9. Generate final summaries
    1. Tree Traversal Retrieval
       Tree traversal starts at the root level of the tree and retrieves the top k documents of a node based on the cosine similarity of the vector embedding. So, at each level it retrieves top k documents from the child node.
    2. Collapsed Tree Retrieval
       Collapsed Tree retrieval is a much simpler method. It collapses all the trees into a single layer and retrieves nodes until a threshold number of tokens is reached based on the cosine similarity of the query vector.

   In our code, we will extract the dataframe text, cluster text, and text from final summaries and combine them to form a single huge list of texts that includes both the root documents and the summaries. This text is then placed in the vector store.

   ```
   all_texts = leaf_texts.copy()
   # Iterate through the results to extract summaries from each level and add them to all_texts
   for level in sorted(results.keys()):
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)
   ```

10. Load the texts into vectorstore:
    1. To store the vectors, we use the Milvus database.[Milvus](https://milvus.io/docs)
    2. Milvus is a strong vector database designed specifically for processing and querying large amounts of vector data.
    3. It stands out for its exceptional performance and scalability, making it ideal for machine learning, deep learning, similarity search jobs, and recommendation systems.
       ```
       from langchain_milvus.vectorstores import Milvus
       URI = "/content/drive/MyDrive/rag_with_raptor/database/milvus_rag.db"
       vector_db = Milvus.from_texts(
        texts= all_texts,
        embedding=embd,
        connection_args={"uri": URI},
        # metadatas= chunks.metadatas
       )
       ```
11. Retrieval Techniques:
    1. Milvus Hybrid Search retriever
       Milvus Hybrid Search retriever combines the advantages of dense and sparse vector searches.
    2. BM25 Retriever
       (BM stands for best matching) is a ranking mechanism used by search engines to determine the relevance of pages to a particular search query.
    3. Dense Passage Retrieval (DPR) - is a set of tools and models for state-of-the-art open-domain Q&A research. It is based on the following paper:[DPR](https://arxiv.org/pdf/2004.04906)

12. Re-ranker Techinques:
    * A re-ranker takes the user's inquiry and all of the initially retrieved documents as input and re-ranks them depending on how well they match the question.
    * Here we use Cohere Reranker
    ```
    os.environ['COHERE_API_KEY'] = 'COHERE_API_KEY'
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )
    ```
13. 
    




       
   
