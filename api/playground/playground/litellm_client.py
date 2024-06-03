from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
import replicate
from litellm import completion


from dotenv import load_dotenv


app = FastAPI()

load_dotenv()

@app.get("/generate_haiku")
async def generate_haiku():

    messages = [{ "content": "Hello, how are you?", "role": "user"}]

    try:
        response = completion(
                model = "replicate/meta/llama-1-70b-chat",
                messages = messages, 
        )

        response_str = response["choices"][0]["message"]["content"]
        
        return {"response": response_str}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4500)

