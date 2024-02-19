import logging
import os
import pinecone

from typing import Optional
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone

app = FastAPI()
templates = Jinja2Templates(directory="templates\\")
vectorestore: Optional[Pinecone] = None
pinecone_index = "sindabad"


@app.on_event("startup")
async def startup_event():

    logging.info("Initializing vectorestore")

    _ = load_dotenv(find_dotenv())

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
                  environment=os.environ['PINECONE_ENV'])
    global vectorstore
    vectorstore = Pinecone.from_existing_index(pinecone_index, embeddings)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    chain = get_chain(
        vectorstore=vectorstore,
        question_handler=question_handler,
        stream_handler=stream_handler
    )
    while True:
        try:
            # Recieve and send back clent message
            question = await websocket.receive_text()
            resp = ChatResponse(
                sender="you",
                message=question,
                type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(
                sender="bot",
                message="",
                type="start"
            )
            await websocket.send_json(start_resp.dict())

            result = await chain.acall(
                {"question": question, "chat_history": chat_history}
            )

            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(
                sender="bot",
                message="",
                type="end"
            )
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, Something went wrong",
                type="error"
            )
            await websocket.send_json(resp.dict())


# @app.post("/chat")
# async def get_chat_response(question: ChatResponse) -> dict:
#
#     chain = make_chain("sindabad")
#
#     try:
#         response = chain({"question": question.message})
#         answer = ChatResponse(sender="bot", message=response["answer"])
#         user_source = response["source_documents"]
#         source_dict = {}
#         for document in user_source:
#             source_dict[f"Page = {document.metadata['page_number']}"] = f"Text chunk: {document.page_content[:160]}...\n"
#         print(source_dict)
#         return answer.dict()
#     except Exception as e:
#         print(f"Exception {e} has ocurred")
#         error_resp = ChatResponse(
#             sender="bot",
#             message="Sorry, something went wrong, Try Again"
#         )
#         return error_resp


def main():
    import uvicorn
    logging.basicConfig(filename='./logs/test.log',
                        encoding='utf-8', level=logging.DEBUG)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
