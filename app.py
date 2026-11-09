import datetime as dt
import logging
import os
import re
import resource
import threading
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bhajjoray-ai")

APP_TIMEZONE = ZoneInfo("Asia/Kolkata")
MODEL_NAME = "Qwen3.5-4B Q4_K_M"
MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH", "/app/models/Qwen3.5-4B-Q4_K_M.gguf"
)
MODEL_CONTEXT = int(os.getenv("LLM_N_CTX", "4096"))
MODEL_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
MODEL_BATCH = int(os.getenv("LLM_N_BATCH", "128"))

# Keep this list only as the existing app's small festival override table.
# Move to a maintained festival data source before relying on it year-round.
FESTIVAL_DB = {
    "2026-11-07": "Pradosh Vrat",
    "2026-11-08": "Diwali / Laxmi Puja",
}

app = FastAPI(title="Bhajjoray AI", version="2.0.0")

# The model is deliberately shared and all inference is serialised. This keeps
# two requests from allocating separate generation/KV-cache work concurrently
# inside the existing 4-vCPU / 8-GB container.
MODEL_LOCK = threading.Lock()


def _configure_non_thinking_chat(model: Llama) -> None:
    """Use the GGUF's own chat template, passing enable_thinking=False.

    llama-cpp-python 0.3.36's high-level create_chat_completion API does not
    expose chat_template_kwargs as a public argument. Its GGUF-derived Jinja
    handler accepts extra template variables, so wrap that handler rather than
    manually recreating Qwen's special-token/chat format.
    """
    handler = model._chat_handlers.get(model.chat_format)
    if handler is None:
        raise RuntimeError(
            "The GGUF chat template was not registered by llama-cpp-python. "
            f"Detected chat format: {model.chat_format!r}."
        )

    def non_thinking_handler(**kwargs):
        kwargs["enable_thinking"] = False
        return handler(**kwargs)

    model.chat_handler = non_thinking_handler


def _load_model() -> Llama:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Check the Docker image build."
        )

    logger.info(
        "Loading %s from %s (ctx=%s, threads=%s, batch=%s)",
        MODEL_NAME,
        MODEL_PATH,
        MODEL_CONTEXT,
        MODEL_THREADS,
        MODEL_BATCH,
    )
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=MODEL_CONTEXT,
        n_threads=MODEL_THREADS,
        n_threads_batch=MODEL_THREADS,
        n_batch=MODEL_BATCH,
        n_ubatch=MODEL_BATCH,
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    _configure_non_thinking_chat(model)
    logger.info("%s loaded successfully", MODEL_NAME)
    return model


# Preserve the current deployment's startup-loading behaviour: Azure should
# only mark this container healthy after the model has loaded successfully.
llm = _load_model()


class QueryRequest(BaseModel):
    user_text: str = Field(min_length=1, max_length=10000)
    language: str = Field(min_length=2, max_length=20)


class PodcastSourceRequest(BaseModel):
    shloka_reference: str = Field(min_length=1, max_length=120)
    shloka_text: str = Field(min_length=1, max_length=3000)
    translation: str = Field(default="", max_length=4000)
    meaning: str = Field(min_length=1, max_length=8000)
    title: Optional[str] = Field(default=None, max_length=180)


class PodcastRequest(PodcastSourceRequest):
    language: str = Field(min_length=2, max_length=20)



def _today_ist() -> str:
    return dt.datetime.now(APP_TIMEZONE).date().isoformat()


def _normalise_language(value: str) -> str:
    value = value.strip().lower()
    if value in {"hindi", "hi", "हिंदी"}:
        return "Hindi"
    if value in {"english", "en"}:
        return "English"
    raise HTTPException(
        status_code=422,
        detail="language must be Hindi or English",
    )


def _clean_model_text(text: str) -> str:
    text = (text or "").strip()
    # Defensive cleanup in case a model/template build still emits reasoning
    # markers despite non-thinking mode being requested.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<\|im_(start|end)\|>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _generate(messages: list[dict], *, max_tokens: int, temperature: float) -> str:
    try:
        with MODEL_LOCK:
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.05,
                presence_penalty=0.2,
                stream=False,
            )
        message = result["choices"][0]["message"]
        content = message.get("content") or ""
        clean_text = _clean_model_text(content)
        if not clean_text:
            raise RuntimeError("The model returned an empty final answer.")
        return clean_text
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Model generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Model generation failed: {type(exc).__name__}",
        ) from exc


def _memory_stats() -> dict:
    stats = {}
    try:
        # ru_maxrss is KiB on Linux.
        stats["process_peak_rss_mb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        )
    except Exception:
        pass

    cgroup_paths = (
        ("/sys/fs/cgroup/memory.current", "container_current_mb"),
        ("/sys/fs/cgroup/memory.max", "container_limit_mb"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "container_current_mb"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "container_limit_mb"),
    )
    for current_path, key in cgroup_paths:
        if key in stats:
            continue
        try:
            with open(current_path, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
            if value != "max":
                numeric_value = int(value)
                # Some cgroup v1 hosts report an effectively unlimited value.
                if numeric_value < (1 << 60):
                    stats[key] = round(numeric_value / (1024 * 1024), 1)
        except (OSError, ValueError):
            continue
    return stats


def _podcast_system_prompt(language: str) -> str:
    if language == "Hindi":
        language_instruction = (
            "Write natural, clear spoken Hindi in Devanagari. Avoid Hinglish and "
            "unnecessary Sanskritised vocabulary."
        )
        word_target = "350–420 Hindi words"
        pace = "about 110–115 words per minute"
    else:
        language_instruction = (
            "Write natural, clear spoken English suitable for an Indian devotional audience."
        )
        word_target = "380–460 English words"
        pace = "about 120–125 words per minute"

    return (
        "You are the devotional podcast writer for the Bhajjoray app. "
        "Create a thoughtful, accurate 3–4 minute spoken podcast based only on "
        "the verified source material supplied by the user. "
        f"{language_instruction} "
        f"Aim for {word_target}, suitable for {pace}. "
        "Explain the central teaching in accessible language, connect it to "
        "ordinary daily life using one relevant example, and end with a calm "
        "practical reflection. "
        "Do not invent scripture, chapter/verse references, quotations, or "
        "historical claims. Do not change or recreate the Sanskrit shloka. "
        "Clearly distinguish the supplied translation/meaning from your "
        "practical reflection. Do not claim that your interpretation is the "
        "only interpretation. "
        "Return only the spoken narration. Do not include analysis, chain of "
        "thought, bullet points, Markdown, production directions, sound cues, "
        "or comments about the task or the AI."
    )


def _podcast_user_prompt(payload: PodcastSourceRequest, language: str) -> str:
    title = payload.title.strip() if payload.title else "Daily Gita Reflection"
    translation = payload.translation.strip() or "Not supplied"
    return (
        f"Language: {language}\n"
        f"Podcast title: {title}\n"
        f"Verified shloka reference: {payload.shloka_reference}\n\n"
        "Original Sanskrit shloka (quote only if needed; preserve exactly):\n"
        f"{payload.shloka_text}\n\n"
        "Verified translation:\n"
        f"{translation}\n\n"
        "Verified meaning/explanation:\n"
        f"{payload.meaning}\n\n"
        "Write the complete 3–4 minute devotional podcast narration now."
    )


def _generate_podcast(payload: PodcastSourceRequest, language: str) -> dict:
    script = _generate(
        [
            {"role": "system", "content": _podcast_system_prompt(language)},
            {"role": "user", "content": _podcast_user_prompt(payload, language)},
        ],
        max_tokens=1300,
        temperature=0.7,
    )
    word_count = len(script.split())
    words_per_minute = 112 if language == "Hindi" else 123
    estimated_seconds = round((word_count / words_per_minute) * 60)
    return {
        "language": language,
        "title": payload.title or "Daily Gita Reflection",
        "shloka_reference": payload.shloka_reference,
        "script": script,
        "word_count": word_count,
        "estimated_duration_seconds": estimated_seconds,
        "estimated_duration_minutes": round(estimated_seconds / 60, 2),
        "duration_target_met": 180 <= estimated_seconds <= 240,
    }


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_status": "ready",
        "model_name": MODEL_NAME,
        "model_file": os.path.basename(MODEL_PATH),
        "runtime": "llama.cpp via llama-cpp-python",
        "context_size": MODEL_CONTEXT,
        "threads": MODEL_THREADS,
        "memory": _memory_stats(),
    }


@app.post("/ask-ai")
def ask_ai(payload: QueryRequest):
    language = _normalise_language(payload.language)
    today_str = _today_ist()
    current_festival = FESTIVAL_DB.get(today_str, "Standard Day")

    system_prompt = (
        "You are a helpful and respectful spiritual guide for the Bhajjoray app. "
        f"Today is {today_str} ({current_festival}). "
        f"Respond in {language}. Answer the user's actual question directly. "
        "Use clear language and avoid describing your reasoning or the task. "
        "Keep the answer concise unless detail is requested. Do not invent "
        "scriptural quotations or references."
    )
    clean_text = _generate(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.user_text.strip()},
        ],
        max_tokens=280,
        temperature=0.45,
    )
    return {
        "status": "success",
        "festival_today": current_festival,
        "ai_answer": clean_text,
    }


@app.post("/generate-podcast")
def generate_podcast(payload: PodcastRequest):
    language = _normalise_language(payload.language)
    result = _generate_podcast(payload, language)
    return {"status": "success", "model": MODEL_NAME, **result}


@app.post("/generate-daily-podcasts")
def generate_daily_podcasts(payload: PodcastSourceRequest):
    # The same model instance generates Hindi first and English second. The
    # lock in _generate() also serialises this against /ask-ai requests.
    hindi = _generate_podcast(payload, "Hindi")
    english = _generate_podcast(payload, "English")
    return {
        "status": "success",
        "model": MODEL_NAME,
        "shloka_reference": payload.shloka_reference,
        "scripts": {"Hindi": hindi, "English": english},
    }
