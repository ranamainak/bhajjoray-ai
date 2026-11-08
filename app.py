from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import re
from llama_cpp import Llama

app = FastAPI()

llm = Llama(
    model_path="/app/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

FESTIVAL_DB = {
    "2026-11-07": "Pradosh Vrat",
    "2026-11-08": "Diwali / Laxmi Puja",
}

class QueryRequest(BaseModel):
    user_text: str
    language: str

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_status": "ready"}

@app.post("/ask-ai")
def ask_ai(payload: QueryRequest):
    today_str = datetime.date.today().isoformat()
    current_festival = FESTIVAL_DB.get(today_str, "Standard Day")

    prompt = (
        f"<|im_start|>system\nYou are a spiritual guide for Bhajjoray. "
        f"Today is {today_str} ({current_festival}). Respond concisely in {payload.language}.<|im_end|>\n"
        f"<|im_start|>user\n{payload.user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\n"
    )

    try:
        output = llm(
            prompt,
            max_tokens=80,
            temperature=0.7,
            stop=["<|im_end|>"]
        )
        raw_text = output["choices"][0]["text"].strip()
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        return {
            "status": "success",
            "festival_today": current_festival,
            "ai_answer": clean_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
