"""Live smoke tests for a running Bhajjoray Qwen3.5 test deployment.

Run after starting the container:
    python -m unittest discover -s tests -v

For an Azure test revision, set BHJ_API_BASE_URL to its HTTPS endpoint.
"""

import json
import os
import re
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("BHJ_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SAMPLE_SOURCE = {
    "shloka_reference": "Bhagavad Gita 2.47",
    "shloka_text": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
    "translation": "You have a right to perform your prescribed duty, but you are not entitled to the fruits of action.",
    "meaning": (
        "The verse teaches that one should focus on sincere and responsible action "
        "rather than becoming attached to the result. It does not advise avoiding action."
    ),
    "title": "Focus on Right Action",
}


def get_json(path):
    req = Request(f"{BASE_URL}{path}", headers={"Accept": "application/json"})
    with urlopen(req, timeout=600) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(path, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=900) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


class Qwen35SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            status, data = get_json("/health")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise unittest.SkipTest(
                f"Bhajjoray API is not reachable at {BASE_URL}: {exc}"
            )
        if status != 200 or data.get("model_status") != "ready":
            raise unittest.SkipTest(f"Model is not ready: {data}")
        cls.health = data

    def test_model_health_and_memory_metrics(self):
        self.assertIn("Qwen3.5-4B", self.health.get("model_name", ""))
        self.assertEqual(self.health.get("context_size"), 4096)
        self.assertIn("memory", self.health)

    def test_daily_hindi_and_english_podcast_generation(self):
        status, data = post_json("/generate-daily-podcasts", SAMPLE_SOURCE)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "success")
        scripts = data.get("scripts", {})
        self.assertIn("Hindi", scripts)
        self.assertIn("English", scripts)

        for language in ("Hindi", "English"):
            with self.subTest(language=language):
                item = scripts[language]
                self.assertTrue(item.get("script", "").strip())
                self.assertGreater(item.get("word_count", 0), 250)
                self.assertTrue(
                    item.get("duration_target_met"),
                    f"{language} script is outside the estimated 3–4 minute range: "
                    f"{item.get('estimated_duration_seconds')} seconds",
                )
                self.assertNotIn("<think>", item["script"].lower())
                if language == "Hindi":
                    self.assertRegex(item["script"], re.compile(r"[\u0900-\u097F]"))
                else:
                    self.assertRegex(item["script"], re.compile(r"[A-Za-z]"))

    def test_z_container_memory_after_generation(self):
        _, data = get_json("/health")
        memory = data.get("memory", {})
        current_mb = memory.get("container_current_mb")
        limit_mb = memory.get("container_limit_mb")
        if current_mb is not None and limit_mb is not None:
            self.assertLess(
                current_mb,
                limit_mb * 0.94,
                f"Container memory has too little safety headroom: {memory}",
            )
        elif memory.get("process_peak_rss_mb") is not None:
            self.assertLess(memory["process_peak_rss_mb"], 7600, memory)

    def test_ask_ai_remains_compatible(self):
        status, data = post_json(
            "/ask-ai",
            {
                "user_text": "Explain Bhagavad Gita 2.47 in simple language.",
                "language": "English",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "success")
        self.assertTrue(data.get("ai_answer", "").strip())


if __name__ == "__main__":
    unittest.main()
