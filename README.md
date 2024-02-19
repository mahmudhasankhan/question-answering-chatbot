# E-commerce Chatbot

General Idea: We want to use OpenAI's large language models like, GPT-3.5 turbo or GPT-4 and combine with our own data. But there's a key issue LLMs( Large Language Models) can only inspect a few thousand words at a time. Therefore, if we have large data, howcan we get the LLM(large language model) to read all of our data and answer the questions about our data.

## That is where, embeddings and vector databases come into play:

First let's talk about embeddings

### Embeddings: 
Embeddings create numerical represenation of pieces of texts. These numerical representations captures the semantic meaning of the piece of text that has been passed through an embedding. Texts with similar content will have similar vectors. **This let's us compare pieces of texts in the vector space.** 

This is really useful when retrieving relevant texts that will be useful to the LLM to formulate an answer for a specific question.

### Vector Database:
A vector database is a way to store this vector representations. The way we store data into this database is by chunks of texts coming from our data. First, we convert our data into "Documents" with the help of **langchain** python library. These documents can be split into "chunks". 

Splitting documents into **chunks** is particularly important because our LLM would not be able to process a large document, instead it is efficient to give the most relevant (usually top most similar 4 chunks) chunks of information that would help the LLM to formulate an answer.

Finally, these similar chunks are then passed through an embedding model to create embeddings of these chunks. Then, store these embeddings in a vector database like, pinecone or redis.
