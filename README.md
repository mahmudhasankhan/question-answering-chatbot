# Question Answering Chatbot

This repo is an implementation of a chatbot that leverages [OpenAI](https://openai.com/)'s LLMs 🧠 and specializes on question answering over your own custom knowledge-base.

Built with [Langchain](https://www.langchain.com/), [FastAPI](https://fastapi.tiangolo.com/), [Pinecone](https://www.pinecone.io/) and [Docker](https://www.docker.com/) for deployment 🚀

## ✅ Running Locally

Clone the repo
```
https://github.com/mahmudhasankhan/question-answering-chatbot.git
```

### Environment Setup
Use any of the two package managers you prefer.
- pip:
    - Create a virtual environment with pip.
    - Activate the virtual environment.
    - Run command `pip install --no-cache-dir --upgrade -r requirements.txt`
- conda:
    - Create a new environment from the environment.yml file.
    - Run command `conda env create -f environment.yml`

You need to create a .env file to setup necessary environment variables.

```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX=
PINECONE_ENV=
```

## 📚 General Idea: 
We want to use OpenAI's large language models like, GPT-3.5 turbo or GPT-4 and combine with our own data i.e. our own knowledge-base. But there's a key issue LLMs( Large Language Models) can only inspect a few thousand words at a time. Therefore, if we have large data, how can we get the LLM(large language model) to read all of our data and answer the questions about our data.

## That is where, embeddings and vector databases come into play:

First let's talk about embeddings

### Embeddings: 
Embeddings create numerical represenation of pieces of texts. These numerical representations captures the semantic meaning of the piece of text that has been passed through an embedding. Texts with similar content will have similar vectors. **This let's us compare pieces of texts in the vector space.** 

This is really useful when retrieving relevant texts that will be useful to the LLM to formulate an answer for a specific question.

### Vector Database:
A vector database is a way to store this vector representations. The way we store data into this database is by chunks of texts coming from our data. First, we convert our data into "Documents" with the help of **langchain** python library. These documents can be split into "chunks". 

Splitting documents into **chunks** is particularly important because our LLM would not be able to process a large document, instead it is efficient to give the most relevant (usually top most similar 4 chunks) chunks of information that would help the LLM to formulate an answer.

Finally, these similar chunks are then passed through an embedding model to create embeddings of these chunks. Then, store these embeddings in a vector database like, pinecone or redis.
