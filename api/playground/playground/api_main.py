from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
import replicate

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

@app.get("/generate_haiku")
async def generate_haiku():
    try:
        events = replicate.stream(
            "meta/llama-2-70b-chat",
            input={"prompt": "Please write a haiku about llamas."}
        )
        
        haiku = ""
        for event in events:
            haiku += str(event)
        
        return {"response": haiku}


    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4500)
