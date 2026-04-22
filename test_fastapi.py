from fastapi import FastAPI; from pydantic import BaseModel; import uvicorn
class ResetRequest(BaseModel):
    task_id: str = 'task_1_priority'
    seed: int = 42
app = FastAPI()
@app.post('/reset')
def reset(req: ResetRequest): return req.model_dump()
if __name__ == '__main__': uvicorn.run(app, port=8888)
