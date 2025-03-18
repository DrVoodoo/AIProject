from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from nutrition_agent import agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
  title="Search food items with the help of data from Livsmedelsverkets",
  description="Encapsulates livsmedelsverkets data to make it searchable, food items are in per 100g",
  version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=['http://localhost:5173', 'https://ai-project-f#.netlify.app'],
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
  message: str
  thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
  response: str
  thread_id: str
  
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  """
  Chat about food items from livsmedelsverkets data.
  
  Parameters:
  - message: The user's message
  - thread_id: Optional thread ID for maintaining conversation context (defaults to "default")
  
  Returns:
  - response: Information found in livsmedelsverkets data
  - thread_id: The thread ID used for the conversation
  """
  try:
   
    result = agent.invoke(
        {"input": request.message},
        config={"configurable": {"thread_id": request.thread_id}}
      )
    
    if result["output"]:
      response = result["output"]
      return ChatResponse(
        response=response,
        thread_id=request.thread_id
      )
    
    raise HTTPException(status_code=500, detail="No response generated")
  
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
  """Welcome endpoint that returns basic information about the API."""
  return {
    "message": "Welcome to query in livsmedelsverkets food item data",
    "description": "Use the /chat endpoint to query about food items",
    "version": "1.0.0"
  }

# todo fix this so it works both in dev and prod
# it is commented out in prod because the command is run there directly but should change so this file can run with the port as a parameter
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)