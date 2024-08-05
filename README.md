# IndiaLaw-Q-A-Intelligent-Legal-Guidance

IndiaLaw Q&A is a project focused on building an intelligent chatbot for providing legal guidance on the Indian legal system. This repository contains the code and resources required to develop, deploy, and interact with the chatbot.

[DEMO OF CHATBOT](https://github.com/user-attachments/assets/7da5f9ca-7698-48ea-8eca-63ef411d24d5)

# Introduction

IndiaLaw Q&A uses powerful AI technology to develop a conversational interface that can initiate and maintain conversations with users, delivering smooth and intuitive legal advice. IndiaLaw Q&A provides exact and contextually relevant responses to legal questions by leveraging the power of Retrieval-Augmented Generation (RAG) with the Recursive Abstractive Processing for Tree Organized Retrieval (RAPTOR) indexing approach and the Milvus vector database. RAPTOR is an innovative and powerful indexing and retrieval strategy for LLMs that takes a bottom-up approach, grouping and summarizing text segments (chunks) to create a hierarchical tree structure. This intelligent chatbot is designed with Streamlit to provide a user-friendly experience for a variety of use cases in the Indian legal system.

RAPTOR introduces a novel approach to retrieval-augmented language models by constructing a recursive tree structure from documents. RAPTOR takes an innovative method to retrieval-augmented language models by creating a recursive tree structure from texts. This enables more efficient and context-aware information retrieval from huge texts, solving major constraints in traditional language models.

The RAPTOR study proposes an innovative approach for indexing and retrieval of documents.

* The leaves are a collection of starter documents.
* Leaves are embedded and crowded.
* Clusters are then combined into higher-level (more abstract) consolidations of information from related documents.
* This is done recursively, resulting in a "tree" of raw documents (leaves) that lead to more abstract summaries.


This tree structure is critical to the RAPTOR function because it captures both high-level and detailed aspects of text, which is especially beneficial for complex theme questions and multi-step reasoning in questioning and answering activities.

Documents are segmented into shorter texts known as chunks, which are then embedded using an embedding model. A clustering method is then used to group these embeddings together. After clusters are formed, the text linked with each cluster is summarized using an LLM.

The summaries are created as nodes in a tree, with higher-level nodes delivering more abstract summaries.
![RAPTOR](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*tDFZ-oHJJM4ww5w_S-ZLNg.png)



Here, we used legal books as input data for our analysis. Each PDF file is over 300 pages. The PDF links for these books are provided below.:
* [Family Law](https://lawfaculty.du.ac.in/userfiles/downloads/LLBCM/Ist%20Term_Family%20Law-%20I_LB105_2023.pdf)
* [Administrative Law](https://lawfaculty.du.ac.in/userfiles/downloads/LLBCM/IVth%20Term_Administrative%20Law_LB%20402_2023.pdf)
* [Labour Law](https://www.icsi.edu/media/webmodules/Labour_Laws&_Practice.pdf)

Steps Involved:
* Load the PDF:
   *Import necessary libraries for PDF processing.
   * Load the PDF document into your Python environment.
   * Perform Cleaning on the PDF:

* Extract text from the PDF.
   * Remove unwanted characters, whitespace, and noise.
* Standardize text format (e.g., lowercase conversion).

* Chunk the PDF:
   * Divide the cleaned text into smaller, manageable chunks.
   * Ensure chunks are of appropriate size for processing.

* Apply Raptor Indexing Techniques:
   * Use Raptor indexing to convert chunks into meaningful embeddings.
   * Create sparse and dense embeddings for each chunk.
   * Store in Milvus Vector Database:

* Set up a Milvus database connection.
  * Create a schema for storing text and embeddings.
  * Insert chunks and their embeddings into the Milvus collection.

* Hybrid Search using BM25 and Semantic Dense Retrieval:
   * Implement BM25 for sparse retrieval.
   * Implement semantic dense retrieval for dense embeddings.
   * Combine results from both methods for hybrid search.

* Re-rank with Cohere Re-rank:
   * Use Cohere’s re-ranking API to re-rank the search results.
   * Integrate the re-ranking step to improve result relevance.
  
* Use Streamlit App to Make User Interface:
   * Create a Streamlit app to provide a user-friendly interface.
   * Implement functionalities for PDF upload, search, and display results.
   * Integrate hybrid search and re-ranking into the Streamlit app.

# Getting Started
1. Clone the repository:
   ```
   git clone https://github.com/your-username/IndiaLaw-Q-A-Intelligent-Legal-Guidance.git
   cd IndiaLaw-Q-A-Intelligent-Legal-Guidance
   ```
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Configuration
   1. Set Environment Variables:
      Open config.env or set environment variables directly in your terminal.
      ```
      export GROQ_API_KEY='your_groq_api_key'
      export COHERE_API_KEY='your_cohere_api_key'
      
      ```

4. Running the Application
   1. Launch Streamlit:
      ```
      streamlit run app.py
      ```
   2. Use the Chatbot:
      * Open your web browser and go to http://localhost:8501 (default Streamlit port).
      * You will see the chat interface titled "💬IndiaLaw-Q-A-Intelligent-Legal-Guidance


5. Interacting with the Chatbot:
   * Start interacting with the chatbot by typing your questions or context in the input field provided.
   * Each interaction alternates between user inputs (Human) and responses generated by the RAG system (AI).
   * The chat history is maintained and displayed in the Streamlit interface.
  
6. Application Structure
Components Used
* Streamlit: Frontend interface for the chatbot application.
* LangChain and Dependencies:
   * **langchain_huggingface** for Hugging Face model embeddings.
   * **langchain_groq** for integrating with the Groq API.      
   * **langchain_milvus** for storing and retrieving vectors using Milvus.
   * **langchain_cohere* for contextual compression reranking.
   *Other necessary libraries and modules for text processing and interaction.

7. Code Explanation
* The application uses Python with Streamlit for the user interface.
* Text preprocessing, vector embeddings, and retrieval mechanisms are handled using LangChain and associated modules.
* Milvus is employed for efficient vector storage and retrieval.
* The RAG system is implemented using a pipeline that combines contextual compression and model-based responses.

8. Additional Notes
* Ensure all required environment variables are set before running the application.
* Adjust configurations and parameters in app.py as per specific requirements or API keys.
