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
MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/app/models/Qwen3.5-4B-Q4_K_M.gguf")
MODEL_CONTEXT = int(os.getenv("LLM_N_CTX", "4096"))
MODEL_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
MODEL_BATCH = int(os.getenv("LLM_N_BATCH", "128"))

# Retain the existing festival overrides. This is not a complete festival calendar.
FESTIVAL_DB = {
    "2026-11-07": "Pradosh Vrat",
    "2026-11-08": "Diwali / Laxmi Puja",
}

app = FastAPI(title="Bhajjoray AI", version="2.1.0")

# One shared model instance and serial inference protect the 4-vCPU / 8-GB container.
MODEL_LOCK = threading.Lock()


def _configure_non_thinking_chat(model: Llama) -> None:
    """Pass enable_thinking=False through the GGUF's registered Jinja template."""
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
        MODEL_NAME, MODEL_PATH, MODEL_CONTEXT, MODEL_THREADS, MODEL_BATCH,
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


# Startup only becomes healthy after the model has loaded.
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
    raise HTTPException(status_code=422, detail="language must be Hindi or English")


def _clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<\|im_(start|end)\|>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _generate(
    messages: list[dict], *, max_tokens: int, temperature: float
) -> tuple[str, str]:
    """Return cleaned content and finish_reason; never silently hide token truncation."""
    try:
        with MODEL_LOCK:
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.08,
                presence_penalty=0.0,
                stream=False,
            )
        choice = result["choices"][0]
        message = choice.get("message") or {}
        content = _clean_model_text(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "unknown")
        if not content:
            raise RuntimeError("The model returned an empty final answer.")
        return content, finish_reason
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
        # Linux reports ru_maxrss in KiB. This is the process lifetime peak.
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
                if numeric_value < (1 << 60):
                    stats[key] = round(numeric_value / (1024 * 1024), 1)
        except (OSError, ValueError):
            continue
    return stats


def _podcast_system_prompt(language: str) -> str:
    if language == "Hindi":
        language_instruction = (
            "Write natural, grammatically correct spoken Hindi in Devanagari. "
            "Use familiar Indian Hindi, not word-for-word English translation. "
            "Avoid Hinglish and unnecessary highly Sanskritised vocabulary."
        )
        word_target = "350 to 420 Hindi words"
        pace = "about 110 to 115 spoken words per minute"
    else:
        language_instruction = (
            "Write fluent, natural spoken English for a general Indian devotional audience."
        )
        word_target = "380 to 460 English words"
        pace = "about 120 to 125 spoken words per minute"

    return (
        "You are the careful devotional podcast writer for the Bhajjoray app. "
        "Create a complete, thoughtful 3-to-4-minute spoken podcast using the "
        "verified source material in the user message. "
        f"{language_instruction} "
        f"Aim for {word_target}, suitable for {pace}. "
        "CONTENT AND SCRIPTURAL ACCURACY: Explain Bhagavad Gita 2.47 in a balanced "
        "way when that is the supplied reference. Cover all three connected ideas "
        "in the supplied verse: a person's responsibility is to perform their "
        "duty; they should not act only from attachment to the fruits or reward; "
        "and they should not become attached to inaction. Do not turn non-attachment "
        "into indifference, carelessness, lack of planning, or refusal to accept "
        "responsibility. Explain that we can make careful choices and influence "
        "outcomes, but cannot completely control every factor or guarantee a result. "
        "Do not say that results belong to God, another person, or fate unless that "
        "claim is explicitly supported by the supplied meaning. Do not tell listeners "
        "that they cannot receive the results of their work. "
        "SOURCE FIDELITY: Treat the supplied translation and meaning as the source "
        "of truth. Do not add unsupported theological claims, historical claims, "
        "stories, quotations, or teachings from other verses. If a detail is not "
        "provided, do not invent it. Never change the verse reference. "
        "The app provides the verified Sanskrit shloka separately as its own audio. "
        "Do not reproduce, paraphrase, or transliterate the Sanskrit shloka in this "
        "podcast script; refer to its chapter and verse instead. "
        "STRUCTURE: Begin with a warm, brief introduction; explain the meaning; "
        "give one realistic and relevant everyday example; explain how the listener "
        "can apply the teaching today; and end with a calm, concise reflection. "
        "Do not repeat the same point to fill time. "
        "LANGUAGE PURITY: Use only the requested language. Never insert Chinese, "
        "Japanese, Korean, or other unrelated scripts or words. "
        "OUTPUT: Return only the complete spoken narration. Do not add headings, "
        "Markdown, bullet points, stage directions, sound cues, analysis, chain of "
        "thought, or comments about the task or the AI. Finish with a complete "
        "sentence and a natural closing."
    )


def _podcast_user_prompt(payload: PodcastSourceRequest, language: str) -> str:
    title = payload.title.strip() if payload.title else "Daily Gita Reflection"
    translation = payload.translation.strip() or "Not supplied"
    return (
        f"Requested language: {language}\n"
        f"Podcast title: {title}\n"
        f"Verified scripture reference: {payload.shloka_reference}\n\n"
        "Verified original Sanskrit shloka (source reference only; do not repeat it):\n"
        f"{payload.shloka_text}\n\n"
        "Verified translation:\n"
        f"{translation}\n\n"
        "Verified meaning/explanation:\n"
        f"{payload.meaning}\n\n"
        "Write the complete 3-to-4-minute spoken narration in the requested language. "
        "Explain the supplied teaching accurately and accessibly. Do not invent "
        "additional scripture or doctrine."
    )


def _language_validation_issues(script: str, language: str) -> list[str]:
    issues = []
    # Catch Chinese/Japanese/Korean characters such as the unexpected text found
    # in the earlier Hindi sample. Devanagari is intentionally not included.
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", script):
        issues.append("unrelated_foreign_script")

    devanagari_count = len(re.findall(r"[\u0900-\u097f]", script))
    latin_count = len(re.findall(r"[A-Za-z]", script))
    if language == "Hindi" and devanagari_count < max(80, latin_count):
        issues.append("not_predominantly_devanagari")
    if language == "English" and latin_count < max(150, devanagari_count * 2):
        issues.append("not_predominantly_english")

    if not re.search(r"[.!?।॥][\"'”’)]?$", script.strip()):
        issues.append("possibly_incomplete_final_sentence")
    return issues


def _generate_podcast(payload: PodcastSourceRequest, language: str) -> dict:
    messages = [
        {"role": "system", "content": _podcast_system_prompt(language)},
        {"role": "user", "content": _podcast_user_prompt(payload, language)},
    ]

    # A single corrective regeneration handles obvious script contamination or
    # a missing final sentence. Token-limit truncation is not retried blindly.
    script = ""
    finish_reason = "unknown"
    issues: list[str] = []
    for attempt in range(2):
        script, finish_reason = _generate(
            messages,
            max_tokens=1800,
            temperature=0.65 if attempt == 0 else 0.35,
        )
        if finish_reason == "length":
            raise HTTPException(
                status_code=502,
                detail=(
                    f"The {language} podcast reached the model token limit and may "
                    "be incomplete. Do not publish this result; shorten source "
                    "material or adjust generation settings."
                ),
            )
        issues = _language_validation_issues(script, language)
        if not issues:
            break
        if attempt == 0:
            messages.extend([
                {"role": "assistant", "content": script},
                {
                    "role": "user",
                    "content": (
                        "Regenerate the entire narration from the original source. "
                        "Correct these output issues: " + ", ".join(issues) + ". "
                        "Use only the requested language, do not reproduce the Sanskrit "
                        "verse, and finish every sentence. Preserve source accuracy."
                    ),
                },
            ])

    if issues:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Podcast output failed language/completion validation; do not publish it.",
                "validation_issues": issues,
            },
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
        "finish_reason": finish_reason,
        "quality_checks": {
            "language_validation": "passed",
            "unexpected_foreign_script": "not_detected",
            "final_sentence_termination": "passed",
            "generation_finish_reason": finish_reason,
            "source_reference_in_metadata": payload.shloka_reference,
        },
    }


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_version": "2.1.0",
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
        f"Today is {today_str} ({current_festival}). Respond in {language}. "
        "Answer the user's actual question directly. Use clear, natural language. "
        "Do not invent scriptural quotations or references, and do not describe "
        "your reasoning or the task. Finish the answer with a complete sentence."
    )
    clean_text, finish_reason = _generate(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.user_text.strip()},
        ],
        max_tokens=512,
        temperature=0.4,
    )
    if finish_reason == "length":
        raise HTTPException(
            status_code=502,
            detail="The answer reached the generation limit and may be incomplete. Please shorten the question and retry.",
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
    # Hindi then English, sequentially, using the same model instance.
    hindi = _generate_podcast(payload, "Hindi")
    english = _generate_podcast(payload, "English")
    return {
        "status": "success",
        "model": MODEL_NAME,
        "shloka_reference": payload.shloka_reference,
        "scripts": {"Hindi": hindi, "English": english},
    }
