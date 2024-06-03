from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
import replicate
import os
from pydantic import BaseModel

from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

os.environ["REPLICATE_API_TOKEN"] = "r8_fRahVtnVezAXEBzukjUat4ohiWskD5x0ZaEYf"
replicateClient = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

@app.get("/llma2")
async def falcon_generate(prompt: str):
    try:
        output = replicateClient.run(
            # "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            "meta/llama-2-70b-chat",
                input={"prompt": prompt}
        )

        result = "".join([event for event in output])

        return {"response": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/falkon")
async def llma_generate(prompt: str):
    try:
        output = replicateClient.run(
        "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
            input={"prompt": prompt}
        )

        result = "".join([event for event in output])

        return {"response": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4544)

class CombinedModel(BaseModel):
    model: str
    prompt: str

@app.post("/combined_models")
async def combined_models(req: CombinedModel):
    try:
        result = call_replicate_model(req.model, req.prompt)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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