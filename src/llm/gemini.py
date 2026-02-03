from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai


@dataclass
class LLMConfig:
    model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None


class GeminiClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
        genai.configure(api_key=api_key)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        model = genai.GenerativeModel(
            self.config.model,
            system_instruction=system_prompt or None,
        )
        generation_config = {
            "temperature": float(self.config.temperature),
        }
        if self.config.max_tokens is not None:
            generation_config["max_output_tokens"] = int(self.config.max_tokens)

        response = model.generate_content(
            user_prompt,
            generation_config=generation_config,
        )
        text = getattr(response, "text", "") or ""
        return text.strip()
