import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

SYSTEM_PROMPT_OPENAI = "You are a helpful assistant."


class OpenAIModel:
    def __init__(self, provider: str = "openai"):
        """
        Initialize the OpenAIModel with the specified provider.
        """
        self.provider = provider.lower()
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not api_key:
            raise ValueError(
                f"API key for {provider} is not set in environment variables."
            )

        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.system_prompt = SYSTEM_PROMPT_OPENAI

    @retry(wait=wait_random_exponential(min=0.1, max=60), stop=stop_after_attempt(6))
    def get_response(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        n: int = 1,
        model: str = "gpt-3.5-turbo",
    ) -> List[dict]:
        if system_prompt is None:
            system_prompt = self.system_prompt

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            n=n,
        )

        usage = getattr(response, "usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        results = []
        for choice in response.choices:
            result = {
                "completion": choice.message.content.strip(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            results.append(result)

        return results


def get_responses(
    user_message: str or List[dict],
    model: str,
    system_prompt: str = "You are a helpful AI assistant.",
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 1.0,
    n: int = 1,
    stop: Optional[List[str]] = None,
    api_host: str = "127.0.0.1",
    port: int = -1,
) -> List[dict]:
    if "axolotl" in model or "Qwen" in model:
        if isinstance(user_message, str):
            user_message = [{"role": "user", "content": user_message}]
        messages = [{"role": "system", "content": system_prompt}] + user_message
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    url = f"http://{api_host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": stop,
    }

    max_retries = 5
    retry_count = 0
    results = []

    while retry_count < max_retries:
        try:
            response_obj = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=6000,
            )
            response_obj.raise_for_status()
            data = response_obj.json()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            results = [
                {
                    "completion": choice["message"]["content"].strip(),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
                for choice in data.get("choices", [])
            ]
            if results:
                break
            retry_count += 1
        except requests.exceptions.RequestException:
            retry_count += 1
            if retry_count >= max_retries:
                results = []
                break

    return results
