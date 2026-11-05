"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-3.8-flash": {"input": 0.075, "output": 0.300},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    text = response.choices[0].message.content or ""
    usage_obj = response.usage
    usage = {
        "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
    }
    return text, float(latency), usage
    


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.8 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 3.8 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-3.8-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature, top_p=top_p, max_output_tokens=max_tokens
    )
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
    latency = time.perf_counter() - start
    metadata = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(metadata, "candidates_token_count", 0) or 0),
    }
    return getattr(response, "text", None) or "", float(latency), usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start
    text = "".join(
        block.text for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", "text") == "text" and getattr(block, "text", None) is not None
    )
    usage_obj = getattr(response, "usage", None)
    usage = {
        "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
    }
    return text, float(latency), usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 3.8 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 3.8 Flash (gemini-3.8-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    def pack(model_name: str, result: tuple[str, float, dict]) -> dict:
        response, latency, usage = result
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        rates = PRICING_1M_TOKENS[model_name]
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        return {
            "response": response,
            "latency": float(latency),
            "cost": float(cost),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    return {
        "gpt4o": pack("gpt-4o", call_openai(prompt)),
        "gpt4o_mini": pack("gpt-4o-mini", call_openai(prompt, model=OPENAI_MINI_MODEL)),
        "gemini_flash": pack("gemini-3.8-flash", call_gemini(prompt)),
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 3.8 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 3.8.

    Behaviour:
        - Streams response tokens from Gemini 3.8 as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict] = []
    while True:
        try:
            prompt = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKết thúc chatbot.")
            break
        if prompt.lower() in {"quit", "exit"}:
            print("Đã kết thúc cuộc trò chuyện.")
            break
        if not prompt:
            continue

        history.append({"role": "user", "parts": [{"text": prompt}]})
        print("Gemini: ", end="", flush=True)
        answer_parts = []
        for attempt in range(4):
            try:
                stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL, contents=history
                )
                for chunk in stream:
                    chunk_text = getattr(chunk, "text", None)
                    if chunk_text:
                        print(chunk_text, end="", flush=True)
                        answer_parts.append(chunk_text)
                break
            except Exception as exc:
                status_code = getattr(exc, "code", None)
                if callable(status_code):
                    status_code = status_code()
                if status_code is None:
                    status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)

                retryable = status_code in {429, 500, 502, 503, 504}
                # Do not retry after text has been shown; that could duplicate the answer.
                if not retryable or answer_parts or attempt == 3:
                    history.pop()
                    print("\nGemini streaming failed.")
                    raise

                delay = 2 ** attempt
                print(f"\nGemini is busy (HTTP {status_code}); retrying in {delay}s...", flush=True)
                time.sleep(delay)

        print()
        history.append({"role": "model", "parts": [{"text": "".join(answer_parts)}]})
        history = history[-6:]  # 3 user/assistant turns


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if base_delay < 0:
        raise ValueError("base_delay must be >= 0")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        results.append({"prompt": prompt, **compare_models(prompt)})
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    headers = ["Prompt", "Model", "Response", "Latency (s)", "Tokens (In/Out)", "Cost (USD)"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    model_labels = [("gpt4o", "GPT-4o"), ("gpt4o_mini", "GPT-4o-mini"), ("gemini_flash", "Gemini-Flash")]

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    for item in results:
        prompt = cell(item.get("prompt", ""))
        for key, label in model_labels:
            stats = item.get(key)
            if not isinstance(stats, dict):
                continue
            response = cell(stats.get("response", ""))
            if len(response) > 50:
                response = response[:47] + "..."
            row = [
                prompt,
                label,
                response,
                f"{float(stats.get('latency', 0) or 0):.3f}",
                f"{int(stats.get('input_tokens', 0) or 0)}/{int(stats.get('output_tokens', 0) or 0)}",
                f"{float(stats.get('cost', 0) or 0):.8f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 3.8 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
