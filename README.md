# Question Answering Chatbot

This repo is an implementation of a chatbot that leverages [OpenAI](https://openai.com/)'s LLMs 🧠 and specializes on question answering over your own custom knowledge-base.

Built with [Langchain](https://www.langchain.com/), [FastAPI](https://fastapi.tiangolo.com/), [Pinecone](https://www.pinecone.io/) and [Docker](https://www.docker.com/) for deployment 🚀

## Setup

Clone the repo
```
https://github.com/mahmudhasankhan/question-answering-chatbot.git
```
You need to create a .env file to setup necessary environment variables.

```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_ENV=
```
## ✅ Running Locally

### Environment Setup

Use any of the two package managers you prefer.
- pip:
    - Create a virtual environment with pip.
    - Activate the virtual environment.
    - Run command `pip install --no-cache-dir --upgrade -r requirements.txt`
- conda:
    - Create a new environment from the environment.yml file.
    - Run command `conda env create -f environment.yml`

### Run your Chatbot 
1. Run `python ingest.py --file filepath` with your pdf file path to ingest pdf doc chunks ino the Pinecone vectorstore (only needs to be done once). Note: If you want to test out my chatbot to see how it works, you can avoid this step.
2. Start the app by running `python main.py` 
3. Open [localhost:8000](localhost:8000) in your browser


## 🐳 Run with Docker (Recommended)

1. You need to have docker desktop installed and running on your machine.

2. Run this command to run your chatbot in an isolated docker container
    ```
    docker compose up
    ```

3. It may take a while to download and install all the dependencies, so be patient! 🙏

4. Open [localhost:8000](localhost:8000) in your browser.

5. To stop the container, run 
    ```
    docker compose down
    ```

## 🚀 Deployement
I will finish this section once I have done deploying the app with docker.

## 📚 Technical Description: 
We want to use OpenAI's large language models like, GPT-3.5 turbo or GPT-4 and combine with our own data i.e. our own knowledge-base. But LLMs (Large Language Models) can only inspect a few thousand words at a time.
Therefore, if our data for the LLMS is fairly large in size then how can we get the LLM to read all of our data and answer the questions about our data.

## That is where, embeddings and vector databases come into play:

First let's talk about embeddings

### Embeddings: 
Embeddings create numerical represenation of pieces of texts. These numerical representations captures the semantic meaning of the piece of text that has been passed through an embedding. Texts with similar content will have similar vectors. **This let's us compare pieces of texts in the vector space.** 

This is really useful when retrieving relevant texts that will be useful to the LLM to formulate an answer for a specific question.

### Vector Database:
A vector database is a way to store these vector representations. The way we store data into this database is by chunks of texts coming from our data. First, we convert our data into "Documents" with the help of **langchain** python library. These documents can be split into "chunks". 

Splitting documents into **chunks** is particularly important because our LLM would not be able to process a large document, instead it is efficient to give the most relevant (usually top most similar 4 chunks) chunks of information that would help the LLM to formulate an answer.

Finally, these similar chunks are then passed through an embedding model to create embeddings of these chunks. Then, store these embeddings in a vector database like, pinecone or redis.

## How it works 
There are two components: ingestion and question-answering.

Ingestion has the following steps:
1. Parse the pdf 
2. Split documents with RecursiveCharacterTextSplitter
3. Create a vectorstore of embeddings, using Pinecone (with Huggingface's embeddings)

Question-Answering has the following steps:
1. Given the chat history and new user question, generate a new standalone question using GPT-3.5-instruct
2. Given that standalone question, look up top 4 similar document chunks from the Pinecone vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.


