import datetime as dt
import logging
import os
import re
import resource
import threading
import time
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

app = FastAPI(title="Bhajjoray AI", version="2.2.0")

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
    request_started = time.perf_counter()
    lock_wait_seconds = 0.0
    inference_seconds = 0.0
    try:
        lock_wait_started = time.perf_counter()
        with MODEL_LOCK:
            lock_wait_seconds = time.perf_counter() - lock_wait_started
            inference_started = time.perf_counter()
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
            inference_seconds = time.perf_counter() - inference_started
        choice = result["choices"][0]
        message = choice.get("message") or {}
        content = _clean_model_text(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "unknown")
        if not content:
            raise RuntimeError("The model returned an empty final answer.")
        usage = result.get("usage") or {}
        logger.info(
            "llm_inference_completed lock_wait_seconds=%.2f inference_seconds=%.2f "
            "total_seconds=%.2f prompt_tokens=%s completion_tokens=%s finish_reason=%s",
            lock_wait_seconds,
            inference_seconds,
            time.perf_counter() - request_started,
            usage.get("prompt_tokens", "unknown"),
            usage.get("completion_tokens", "unknown"),
            finish_reason,
        )
        return content, finish_reason
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Model generation failed after %.2f seconds (lock wait %.2f seconds, "
            "inference %.2f seconds)",
            time.perf_counter() - request_started,
            lock_wait_seconds,
            inference_seconds,
        )
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


def _is_gita_2_47(reference: str) -> bool:
    """Identify Bhagavad Gita 2.47 without imposing its meaning on other verses."""
    normalized = re.sub(r"\s+", " ", reference.strip().lower())
    return bool(re.search(r"(?<!\d)2\s*[.:/-]\s*47(?!\d)", normalized))


def _podcast_word_limits(language: str) -> tuple[int, int, int]:
    """Hard acceptance limits and estimated speaking rate."""
    if language == "Hindi":
        return 360, 420, 112
    return 340, 400, 123


def _podcast_target_words(language: str) -> tuple[int, int]:
    """Narrow generation target that leaves headroom inside hard acceptance limits."""
    if language == "Hindi":
        return 380, 400
    return 360, 380


def _podcast_system_prompt(language: str, reference: str) -> str:
    minimum, maximum, wpm = _podcast_word_limits(language)
    target_minimum, target_maximum = _podcast_target_words(language)
    min_seconds = round(minimum / wpm * 60)
    max_seconds = round(maximum / wpm * 60)
    duration_guidance = (
        f"{min_seconds // 60} minutes {min_seconds % 60:02d} seconds to "
        f"{max_seconds // 60} minutes {max_seconds % 60:02d} seconds"
    )
    if language == "Hindi":
        language_instruction = (
            "Write natural, grammatically correct spoken Hindi in Devanagari. "
            "Use familiar Indian Hindi, not literal English-to-Hindi translation. "
            "Avoid Hinglish, awkward Sanskritised expressions, and uncommon "
            "technical terms when a simple Hindi phrase is available."
        )
    else:
        language_instruction = (
            "Write fluent, natural spoken English for a general Indian devotional audience. "
            "Use clear, conversational sentences and avoid overly grand or repetitive prose."
        )

    if _is_gita_2_47(reference):
        verse_guidance = (
            "This source is Bhagavad Gita 2.47. Explain its connected teachings "
            "accurately: a person should perform their duty; should not act only "
            "from attachment to rewards or fruits; and should not become attached "
            "to inaction. Explain that non-attachment is not carelessness, lack of "
            "planning, indifference to quality, or avoidance of responsibility. "
            "Do not say that a person can never receive the results of work. "
            "Do not claim that results belong solely to God, fate, or another person. "
            "Acknowledge that actions influence outcomes while not every factor is controllable."
        )
    else:
        verse_guidance = (
            "Explain only the teaching supported by this verse's supplied Sanskrit, "
            "translation, and verified meaning. Do not import the three-part teaching "
            "of Bhagavad Gita 2.47 unless this source is actually that verse. "
            "Do not add doctrines, conclusions, or claims from other verses."
        )

    return (
        "You are the careful devotional podcast writer for the Bhajjoray app. "
        "Create a complete spoken podcast from the verified source material in "
        "the user message. Follow the word-count requirement as the strict priority. "
        f"Write between {minimum} and {maximum} whitespace-separated words, aiming "
        f"for approximately {round((minimum + maximum) / 2)} words. "
        f"At about {wpm} spoken words per minute, this corresponds to approximately "
        f"{duration_guidance} of narration. Never add words just to lengthen the audio. "
        f"{language_instruction} "
        f"{verse_guidance} "
        "SOURCE FIDELITY: Treat the supplied Sanskrit, translation, and meaning as "
        "the source of truth. Do not change the shloka, invent another quotation, "
        "invent a chapter or verse number, introduce unsupported theological or "
        "historical claims, or describe an interpretation as the only possible one. "
        "If the supplied meaning is limited, explain it clearly without pretending "
        "that missing commentary was supplied. "
        "REFERENCE SAFETY: The exact chapter and verse reference is returned in API "
        "metadata. Do not say a chapter number or verse number aloud in the narration. "
        "You may say 'today's shloka' or 'this verse'. This prevents incorrect spoken "
        "ordinal numbers. "
        "AUDIO STRUCTURE: Use five natural sections in connected paragraphs: a brief "
        "welcome and topic; the source-grounded meaning; the key practical insight; "
        "one relevant everyday example; and one practical application followed by "
        "a calm closing reflection. Use only one example. Do not repeat the same "
        "idea in multiple paragraphs. Keep the introduction and conclusion short. "
        "Do not mention the app or brand name in the spoken narration; branding is "
        "handled by the player interface. Do not use time-specific greetings such as "
        "good morning or good night. "
        "LANGUAGE PURITY: Use only the requested language. For Hindi, use Devanagari "
        "and natural Hindi grammar. For English, use English. Do not insert Chinese, "
        "Japanese, Korean, Arabic, Cyrillic, or unrelated Indian-language scripts. "
        "Do not use Hinglish in Hindi narration. "
        "OUTPUT: Return only the final spoken narration as plain text. Do not add "
        "headings, Markdown, bullets, stage directions, sound cues, analysis, chain "
        "of thought, or comments about the task or the AI. Do not reproduce the "
        "Sanskrit shloka because the app supplies it separately as its own audio. "
        "Finish with a complete sentence and a natural closing. Stay within the "
        f"{minimum}-{maximum} word range."
    )

def _podcast_user_prompt(payload: PodcastSourceRequest, language: str) -> str:
    title = payload.title.strip() if payload.title else "Daily Gita Reflection"
    translation = payload.translation.strip() or "Not supplied"
    minimum, maximum, _ = _podcast_word_limits(language)
    target_minimum, target_maximum = _podcast_target_words(language)
    return (
        f"Requested language: {language}\n"
        f"Podcast title: {title}\n"
        f"Verified scripture reference (metadata only; do not speak the number): "
        f"{payload.shloka_reference}\n\n"
        "Verified original Sanskrit shloka (do not reproduce it in the narration):\n"
        f"{payload.shloka_text}\n\n"
        "Verified translation:\n"
        f"{translation}\n\n"
        "Verified meaning/explanation:\n"
        f"{payload.meaning}\n\n"
        f"Aim for {target_minimum} to {target_maximum} words in {language}; the hard "
        f"acceptance range is {minimum} to {maximum}. "
        "Use exactly one everyday example. Explain only what is supported by the supplied source. Do not "
        "invent verse references, unsupported theology, or extra scriptural claims. "
        "Return complete, natural spoken narration only."
    )

def _foreign_script_present(text: str) -> bool:
    # Devanagari and Latin are the expected scripts for Hindi and English here.
    # These ranges catch common accidental insertions such as the Chinese phrase
    # that appeared in the first Hindi podcast sample.
    foreign_script_pattern = (
        r"[\u0400-\u052f\u0590-\u05ff\u0600-\u06ff"
        r"\u0750-\u077f\u08a0-\u08ff\u0e00-\u0e7f"
        r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
        r"\uac00-\ud7af\u0a80-\u0aff\u0b00-\u0b7f"
        r"\u0b80-\u0bff\u0c00-\u0c7f\u0c80-\u0cff"
        r"\u0d00-\u0d7f]"
    )
    return bool(re.search(foreign_script_pattern, text))


def _podcast_quality_issues(
    script: str,
    language: str,
    payload: PodcastSourceRequest,
    finish_reason: str,
) -> list[str]:
    issues: list[str] = []
    minimum, maximum, wpm = _podcast_word_limits(language)
    word_count = len(script.split())
    estimated_seconds = round((word_count / wpm) * 60) if word_count else 0

    if word_count < minimum:
        issues.append(f"too_short_{word_count}_words_minimum_{minimum}")
    elif word_count > maximum:
        issues.append(f"too_long_{word_count}_words_maximum_{maximum}")

    if finish_reason == "length":
        issues.append("generation_token_limit_reached")
    elif finish_reason not in {"stop", "unknown"}:
        issues.append(f"unexpected_finish_reason_{finish_reason}")

    if _foreign_script_present(script):
        issues.append("unexpected_foreign_script")

    letters = [char for char in script if char.isalpha()]
    devanagari_count = len(re.findall(r"[\u0900-\u097f]", script))
    latin_count = len(re.findall(r"[A-Za-z]", script))
    letter_count = len(letters)
    if language == "Hindi":
        if letter_count == 0 or devanagari_count / letter_count < 0.65:
            issues.append("not_predominantly_hindi_devanagari")
        # Chapter/verse identifiers are supplied in metadata. Catch common
        # ordinal-number mistakes such as 'दूसरे अध्याय के चौथें श्लोक'.
        hindi_number_words = (
            "पहले|दूसरे|तीसरे|चौथे|चौथें|पाँचवें|पांचवें|छठे|सातवें|"
            "आठवें|नौवें|दसवें|ग्यारहवें|बारहवें|सैंतालीसवें|सैंतालीसवाँ"
        )
        if re.search(rf"(?:{hindi_number_words})\s+अध्याय", script) or re.search(
            rf"अध्याय.{0,45}(?:{hindi_number_words})\s+श्लोक", script
        ):
            issues.append("spoken_chapter_verse_number_may_be_inaccurate")
        if re.search(r"अध्याय\s*[0-9०-९]+|श्लोक\s*[0-9०-९]+", script):
            issues.append("spoken_numeric_reference_not_allowed")
    else:
        if letter_count == 0 or latin_count / letter_count < 0.75:
            issues.append("not_predominantly_english")
        if devanagari_count:
            issues.append("devanagari_text_not_allowed_in_english_narration")
        if re.search(r"\b(?:chapter|verse)\s*\d+", script, flags=re.IGNORECASE):
            issues.append("spoken_numeric_reference_not_allowed")

    if not re.search(r"[.!?।॥][\"'”’)]?\s*$", script.strip()):
        issues.append("possibly_incomplete_final_sentence")

    # The verified Sanskrit shloka is played separately in the app. Do not allow
    # the narration to silently alter or duplicate it as generated text.
    source_shloka = re.sub(r"\s+", "", payload.shloka_text)
    script_compact = re.sub(r"\s+", "", script)
    if source_shloka and source_shloka in script_compact:
        issues.append("source_shloka_repeated_in_podcast")

    minimum_seconds = round(minimum / wpm * 60)
    maximum_seconds = round(maximum / wpm * 60)
    if not (minimum_seconds <= estimated_seconds <= maximum_seconds):
        issues.append(
            f"estimated_duration_outside_word_count_range_{estimated_seconds}_seconds"
        )

    return issues


def _generate_podcast(payload: PodcastSourceRequest, language: str) -> dict:
    base_messages = [
        {
            "role": "system",
            "content": _podcast_system_prompt(language, payload.shloka_reference),
        },
        {"role": "user", "content": _podcast_user_prompt(payload, language)},
    ]
    messages = list(base_messages)
    script = ""
    finish_reason = "unknown"
    issues: list[str] = []
    attempts = 0
    minimum, maximum, words_per_minute = _podcast_word_limits(language)
    target_minimum, target_maximum = _podcast_target_words(language)
    attempt_history: list[dict] = []
    min_seconds = round(minimum / words_per_minute * 60)
    max_seconds = round(maximum / words_per_minute * 60)
    duration_guidance = (
        f"{min_seconds // 60} minutes {min_seconds % 60:02d} seconds to "
        f"{max_seconds // 60} minutes {max_seconds % 60:02d} seconds"
    )

    for attempt in range(2):
        attempts = attempt + 1
        script, finish_reason = _generate(
            messages,
            max_tokens=1500 if attempt == 0 else 1250,
            temperature=0.55 if attempt == 0 else 0.30,
        )
        issues = _podcast_quality_issues(script, language, payload, finish_reason)
        attempt_history.append({
            "attempt": attempts,
            "word_count": len(script.split()),
            "finish_reason": finish_reason,
            "validation_issues": list(issues),
        })
        logger.info(
            "podcast_attempt_finished language=%s attempt=%s words=%s finish_reason=%s issues=%s",
            language, attempts, len(script.split()), finish_reason, issues,
        )
        if not issues:
            break

        if attempt == 0:
            current_word_count = len(script.split())
            if current_word_count > maximum:
                length_direction = (
                    "Shorten the narration substantially. Remove repetition, long "
                    "introductions, repeated conclusions, and unnecessary transitions. "
                    "Do not remove the central teaching, the single everyday example, "
                    "or the practical application. Do not add new ideas. "
                )
            elif current_word_count < minimum:
                length_direction = (
                    "Expand the narration with concise, source-grounded explanation "
                    "and practical detail. Do not pad it with repetition or invent "
                    "additional scriptural claims. Keep exactly one everyday example. "
                )
            else:
                length_direction = (
                    "Keep the narration concise and within the target range while "
                    "correcting the listed validation issues. Do not add repetition. "
                )
            correction = (
                "The previous narration failed validation. Rewrite the COMPLETE "
                "narration as a finished final script. This is an editing task, not "
                "a request to explain the failure. "
                f"Validation issues to fix: {'; '.join(issues)}. "
                f"Previous word count: {current_word_count}. "
                f"Your new target is {target_minimum} to {target_maximum} words, "
                f"aiming near {((target_minimum + target_maximum) // 2)}. "
                f"{length_direction}"
                f"The absolute permitted range is {minimum} to {maximum} words. "
                "Preserve the verified source meaning. Do not mention the numeric "
                "chapter/verse reference aloud. Do not repeat the Sanskrit shloka. "
                f"Use only the requested language ({language}), and finish with a "
                f"complete sentence. The narration MUST be between {minimum} and "
                f"{maximum} words. Return only the revised narration, with no "
                "headings or explanation."
            )
            messages = base_messages + [
                {"role": "assistant", "content": script},
                {"role": "user", "content": correction},
            ]

    if issues:
        word_count = len(script.split())
        estimated_seconds = round((word_count / words_per_minute) * 60) if word_count else 0
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Podcast generation was rejected after one corrective retry. "
                    "Do not publish this script."
                ),
                "language": language,
                "attempts": attempts,
                "attempt_history": attempt_history,
                "word_count": word_count,
                "estimated_duration_seconds": estimated_seconds,
                "validation_issues": issues,
            },
        )

    word_count = len(script.split())
    estimated_seconds = round((word_count / words_per_minute) * 60)
    return {
        "language": language,
        "title": payload.title or "Daily Gita Reflection",
        "shloka_reference": payload.shloka_reference,
        "script": script,
        "word_count": word_count,
        "estimated_duration_seconds": estimated_seconds,
        "estimated_duration_minutes": round(estimated_seconds / 60, 2),
        "duration_target_met": True,
        "finish_reason": finish_reason,
        "generation_attempts": attempts,
        "attempt_history": attempt_history,
        "quality_checks": {
            "word_count": "passed",
            "estimated_duration": "passed",
            "language_validation": "passed",
            "language_script_check": "passed",
            "unexpected_foreign_script": "not_detected",
            "final_sentence_termination": "passed",
            "source_shloka_not_repeated": "passed",
            "scriptural_accuracy": "requires_editorial_review",
            "source_reference_in_metadata": payload.shloka_reference,
        },
    }


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_version": "2.2.0",
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
    logger.info("podcast_request_started endpoint=generate-podcast language=%s", language)
    result = _generate_podcast(payload, language)
    logger.info(
        "podcast_request_completed endpoint=generate-podcast language=%s words=%s attempts=%s",
        language, result.get("word_count"), result.get("generation_attempts"),
    )
    return {"status": "success", "model": MODEL_NAME, **result}


@app.post("/generate-daily-podcasts")
def generate_daily_podcasts(payload: PodcastSourceRequest):
    # Hindi then English, sequentially, using the same model instance.
    logger.info("podcast_request_started endpoint=generate-daily-podcasts languages=Hindi,English")
    hindi = _generate_podcast(payload, "Hindi")
    english = _generate_podcast(payload, "English")
    logger.info("podcast_request_completed endpoint=generate-daily-podcasts")
    return {
        "status": "success",
        "model": MODEL_NAME,
        "shloka_reference": payload.shloka_reference,
        "scripts": {"Hindi": hindi, "English": english},
    }
