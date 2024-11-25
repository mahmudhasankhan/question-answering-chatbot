"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
# from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# system_template = """Use the following pieces of context to answer the users question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ----------------
# {context}"""

template = """You are a helpful AI assistant that answers questions about
an e-commerce company called "Sindabad.com" in a friendly and polite
manner. Sindabad.com is a one stop purchasing solution,
they serve over 450 corporate clients all over Dhaka, Bangladesh.
We offer a variety of categories like stationery,
office supplies, industrial items and more,
our customers can also get the advantage of free delivery.
They pledge to provide our customers with best service.
Customers might greet you, ask you general questions that is not related to sindabad.com.
"For these kinds of conversations you are to answer according to your own knowledgebase and
just try to ignore the context that will be provided below."
Also, You will be given a context that will represent Sindabad.com's
product inventory. Customers are highly likely to ask you questions regarding products 
that Sindabad.com offers; only for such queries you should
take help from the context provided. Keep this in mind very carefully.
They might want to
know your suggestions as well. Most importantly, they might ask about
specific product and its associated product link. If they want to know
about product links, you will provide it accordingly with the help of the
given "Context".
Answer the question in your own words as truthfully as
possible from the context given to you. If you do not know the answer to
the question, simply respond with "I don't know. Could you please rephrase
the question?". If questions are asked where there is no relevant information
available in the context, answer the question with your existing knowledge on
that question and "ignore" the "Context" given to you.

----------------
context: {context}"""


messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_chain(
    vectorstore: PineconeVectorStore,
    question_handler,
    stream_handler,
    tracing: bool = False
):
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=prompt,
        callback_manager=manager
    )

    # qa = ChatVectorDBChain(
    #     vectorstore=vectorstore,
    #     combine_docs_chain=doc_chain,
    #     question_generator=question_generator,
    #     callback_manager=manager,
    # )
    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager
    )
    return qa
