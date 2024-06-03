from fastapi import FastAPI, HTTPException, Request
from typing import List, Dict, Any, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import os
import time
from openai import OpenAI
from sse_starlette.sse import EventSourceResponse
import json
import replicate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Verify the environment variable
api_key = os.getenv("OPENAI_API_KEY")
replicate_api_key = os.getenv("REPLICATE_API_TOKEN")

if api_key and replicate_api_key :
    logger.info(f"OpenAI API and Replicate Api Key loaded")
else:
    logger.error("OpenAI API key or Replicate Api Key not found")

openAIClient = OpenAI(api_key=api_key)
replicateClient = replicate.Client(api_token=replicate_api_key)

# Global variables for chat history and vector store
chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
vector_store = None

app = FastAPI()

class UrlModel(BaseModel):
    url: str

@app.post("/api/scrape")
async def get_vectorstore_from_url(item: UrlModel):
    """
        Request Example:
        {
            "url": "https://www.example.com"
        }
    """
    url = item.url
    global vector_store
    logger.info(f"Received request to scrape URL: {url}")

    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        
        if not api_key:
            raise ValueError("OpenAI API key not found")

        max_retries = 5
        retry_delay = 2  # initial delay in seconds
        for attempt in range(max_retries):
            try:
                vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(openai_api_key=api_key))
                logger.info("Vector store initialized")
                return {"message": "Vector store initialized"}
            # except OpenAI.RateLimitError as e:
            #     logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            #     time.sleep(retry_delay)
            #     retry_delay *= 2  # Exponential backoff
            # except OpenAI.OpenAIError as e:
            #     if e.http_status == 429:
            #         logger.error(f"Quota exceeded: {e}")
            #         raise HTTPException(status_code=429, detail="Quota exceeded. Please check your OpenAI plan and usage.")
            #     else:
            #         raise e
            except Exception as e:
                  time.sleep(retry_delay)
                  retry_delay *= 2  # Exponential backoff
                  if attempt == max_retries - 1:
                      raise e

        raise HTTPException(status_code=429, detail="Rate limit exceeded after multiple attempts.")
    except Exception as e:
        logger.error(f"Error in get_vectorstore_from_url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    global chat_history, vector_store
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Vector store not found")

    try:
        user_message = HumanMessage(content=request.message)

        retriever_chain = get_context_retriever_chain(vector_store)
        context = retriever_chain.invoke({"input": request.message, "chat_history": chat_history})
        logger.info(f"Context: {context}")

        # Assuming context is a list of documents
        context_text = " ".join(doc.page_content for doc in context)

        logger.info(f"User message: {user_message.content}")
        logger.info(f"Chat history: {chat_history}")

        responses = {}

        # Call OpenAI models
        models_openai = ["gpt-3.5-turbo", "gpt-4"]
        for model in models_openai:
            response = call_openai_model(model, request.message)
            responses[model] = response
            logger.info(f"Response from {model}: {response}")

        models_replicate = ["meta/llama-2-70b-chat","joehoover/falcon-40b"]
        for model in models_replicate:
            response = call_replicate_model(model, request.message)
            responses[model] = response
            logger.info(f"Response from {model}: {response}")

        best_model, best_response = get_best_response(responses, context_text)
        logger.info(f"Best response from {best_model}: {best_response}")

        chat_history.append(user_message)
        ai_message = AIMessage(content=best_response)
        chat_history.append(ai_message)

        return {"answer": best_response, "model": best_model}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/stream")
async def chat_stream(request: Request):
    global chat_history, vector_store
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Vector store not found")

    message = request.query_params.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' query parameter")

    user_message = HumanMessage(content=message)
    retriever_chain = get_context_retriever_chain(vector_store)
    context = retriever_chain.invoke({"input": message, "chat_history": chat_history})
    logger.info(f"Context: {context}")

    # Assuming context is a list of documents
    context_text = " ".join(doc.page_content for doc in context)

    async def event_generator():
        try:
            if await request.is_disconnected():
                return

            responses = {}
            models_openai = ["gpt-3.5-turbo", "gpt-4"]
            for model in models_openai:
                response = call_openai_model(model, message)
                responses[model] = response
                logger.info(f"Response from {model}: {response}")

            models_replicate = ["meta/llama-2-70b-chat","joehoover/falcon-40b"]
            for model in models_replicate:
                response = call_replicate_model(model, message)
                responses[model] = response
                logger.info(f"Response from {model}: {response}")

            best_model, best_response = get_best_response(responses, context_text)
            logger.info(f"Best response from {best_model}: {best_response}")

            chat_history.append(user_message)
            ai_message = AIMessage(content=best_response)
            chat_history.append(ai_message)
            yield {"data": json.dumps({"role": "bot", "content": best_response, "model": best_model})}
        except Exception as e:
            logger.error(f"Error in chat_stream: {e}", exc_info=True)
            yield {"data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())

def call_openai_model(model: str, message: str) -> str:
    response = openAIClient.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

def call_replicate_model(model: str, prompt: str) -> str:
    try:
        logger.info(f"Calling Replicate model: {model}")
        if model == "joehoover/falcon-40b-instruct": 
            output = replicateClient.run(
                "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
                input={"prompt": prompt}
            )
            result = "".join([event for event in output])
            return result

        elif model == "meta/llama-2-70b-chat":
            output = replicateClient.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={"prompt": prompt}
            )
            result = "".join([event for event in output])
            return result
        logger.error(f"Model {model} not found")
        return "Error !!!!"
    except Exception as e:
        logger.error(f"Error calling Replicate model: {e}")
        return ""


def evaluate_response_accuracy(response: str, context: str) -> int:
    # Simple heuristic to evaluate response accuracy based on context
    score = 0
    for word in context.split():
        if word.lower() in response.lower():
            score += 1
    return score

def get_best_response(responses: Dict[str, str], context: str) -> Tuple[str, str]:
    # Evaluate the accuracy of each response based on the context and select the best one
    best_model = max(responses, key=lambda model: evaluate_response_accuracy(responses[model], context))
    return best_model, responses[best_model]


def get_context_retriever_chain(vector_store):
    logger.info("Creating context retriever chain")
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

