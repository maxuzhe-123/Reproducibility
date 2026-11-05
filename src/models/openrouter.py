import os
import requests


def build_reasoning_payload(thinking_cfg):
    """Translate the `thinking` switch in the config into OpenRouter's `reasoning` field.

    Applies to Gemini 2.5 (pro/flash) and Claude (Sonnet 4.6, etc.) — OpenRouter uses
    the same `reasoning` field to control thinking for both providers:
      - `reasoning.max_tokens` → Gemini thinkingBudget / Claude budget_tokens
      - `reasoning.effort`     → thinking effort level (low/medium/high/max)
      - `reasoning.enabled:false` / max_tokens:0 → disable thinking

    Config format:
        model:
          thinking:
            enabled: true     # whether to think; false → disabled
            mode: budget      # budget | effort — whichever is chosen is sent
            budget: 4096      # effective when mode=budget; omitted = dynamic thinking
                              #   note: Gemini can go down to 512/0; Claude minimum is 1024, 0 = disabled
            effort: high      # effective when mode=effort: low | medium | high | max
            exclude: true     # do not return thinking text (tokens still count toward the budget; default true)

    Returns (reasoning_dict or None, thinking_budget: int).
    - reasoning_dict=None means thinking is not configured → nothing is sent and the model uses its default behavior.
    - thinking_budget is added to max_tokens as a numeric budget (only predictable when mode=budget;
      in effort/dynamic modes the budget is unknown and 0 is returned, so max_new_tokens should be set larger to avoid truncation).
    """
    if not thinking_cfg:
        return None, 0
    enabled = thinking_cfg.get("enabled", True)
    exclude = bool(thinking_cfg.get("exclude", True))

    # Disable thinking (flash/claude: budget 0; gemini-pro is clamped by the provider to its minimum)
    if not enabled:
        return {"max_tokens": 0, "exclude": exclude}, 0

    mode = str(thinking_cfg.get("mode") or "budget").lower()

    # Effort level mode (native to Anthropic; converted by OpenRouter for Gemini)
    if mode == "effort":
        effort = thinking_cfg.get("effort")
        if not effort:
            return {"enabled": True, "exclude": exclude}, 0  # Dynamic thinking
        return {"effort": str(effort), "exclude": exclude}, 0

    # Numeric budget mode (default)
    budget = thinking_cfg.get("budget", None)
    if budget == 0:
        return {"max_tokens": 0, "exclude": exclude}, 0
    if budget is None:
        return {"enabled": True, "exclude": exclude}, 0  # Dynamic thinking
    budget = int(budget)
    return {"max_tokens": budget, "exclude": exclude}, budget


class OpenRouterLM:
    """
    OpenRouter remote closed-source model wrapper.
    Works for: GPT-4o, GPT-4.1, Claude 3.7, Gemini-1.5-Pro, etc.
    """
    def __init__(self,
                 model_id=None,
                 api_key=None,
                 stop_strings=None,
                 do_sample=True,
                 top_k=None,
                 top_p=None,
                 temperature=1.0,
                 max_retries=3,
                 reasoning=None,
                 thinking_budget=0):


        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing API Key. Please export OPENROUTER_API_KEY in your .env or system env."
            )

        self.stop_strings = stop_strings or ["Stop."]
        self.max_retries = max_retries
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        # OpenRouter `reasoning` field (thinking control). None means it is not sent
        # and the model uses its default behavior; otherwise it is put into the payload as is.
        self.reasoning = reasoning
        # Thinking budget (= Gemini thinkingBudget). Thinking tokens count toward max_tokens,
        # so it is added to max_new_tokens in the request, ensuring the visible output budget is not consumed by thinking.
        self.thinking_budget = int(thinking_budget or 0)
        # finish_reason of the most recent successful response (e.g., "stop" / "length").
        # The upper layer (generation_core) records it in response_metadata to help locate truncation afterwards.
        self.last_finish_reason = None

    def generate(self, system, user, max_new_tokens=512):
        """
        Call the OpenRouter chat/completions endpoint and return the model's plain-text output.
        - Success: returns str (model output)
        - Failure: raises RuntimeError after max_retries retries
        """
        if not self.model_id:
            raise ValueError("model_id must be provided for OpenRouterLM")

        url = "https://openrouter.ai/api/v1/chat/completions"

        last_err = None

        for _ in range(self.max_retries):
            try:
                # --- Build the payload ---
                # Thinking tokens count toward max_tokens: add the thinking budget on top of the visible output budget,
                # so the visible output space is the same across thinking levels and truncation probabilities are comparable.
                payload = {
                    "model": self.model_id,
                    "messages": [],
                    "max_tokens": max_new_tokens + self.thinking_budget,
                    "temperature": self.temperature,
                }

                # Thinking/reasoning control (sent only when thinking is configured)
                if self.reasoning is not None:
                    payload["reasoning"] = self.reasoning

                # if self.top_p is not None:
                #     payload["top_p"] = self.top_p
                # if self.top_k is not None:
                #     payload["top_k"] = self.top_k
                # payload = {
                #     "model": self.model_id,
                #     "messages": [],
                #     "max_tokens": max_new_tokens,
                #     "temperature": self.temperature,
                #     "top_p": self.top_p,
                #     "top_k":self.top_k
                # }

                # Sampling parameters
                # Without sampling (do_sample=False), force temperature=0 to remove randomness
                if self.do_sample is False:
                    payload["temperature"] = 0.0
                else:
                    if self.temperature is not None:
                        payload["temperature"] = self.temperature
                    if self.top_p is not None:
                        payload["top_p"] = self.top_p
                    if self.top_k is not None:
                        payload["top_k"] = self.top_k

                # system / user messages
                if system:
                    payload["messages"].append({"role": "system", "content": system})
                payload["messages"].append({"role": "user", "content": user})

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                # --- Send the request ---
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                status = resp.status_code

                # Sometimes the error body is not JSON; handle that here
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw_text": resp.text}

                # --- HTTP-level errors ---
                if status != 200:
                    raise RuntimeError(f"OPENROUTER_HTTP_ERROR {status}: {data}")

                # --- OpenRouter internal errors ---
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"OPENROUTER_API_ERROR: {data['error']}")

                # --- Normal response ---
                if isinstance(data, dict) and "choices" in data:
                    choice  = data["choices"][0]
                    message = choice.get("message", {}) or {}
                    text    = message.get("content", None)
                    finish_reason = choice.get("finish_reason", "unknown")

                    # content is None: possibly content filtering / a tool_call response
                    if text is None:
                        raise RuntimeError(
                            f"OPENROUTER_NULL_CONTENT: content=None, "
                            f"finish_reason={finish_reason!r}, "
                            f"choice={choice}"
                        )

                    # Apply stop_strings (e.g., "Stop.")
                    if self.stop_strings:
                        for s in self.stop_strings:
                            if not s:
                                continue
                            idx = text.find(s)
                            if idx != -1:
                                text = text[:idx]
                                break

                    # Blank/degenerate output: the model occasionally falls into a repetition loop and emits only whitespace (often with
                    # finish_reason="length" truncation). Treat it as a failure and retry,
                    # to avoid counting pure whitespace as a successful sample and polluting downstream pass@k.
                    if not text.strip():
                        raise RuntimeError(
                            f"OPENROUTER_EMPTY_CONTENT: content is whitespace-only "
                            f"(len={len(text)}), finish_reason={finish_reason!r}"
                        )

                    # Record this finish_reason for the upper layer to write into response_metadata
                    self.last_finish_reason = finish_reason
                    return text

                # --- Malformed response structure ---
                raise RuntimeError(f"OPENROUTER_UNKNOWN_RESPONSE: {data}")

            except Exception as e:
                # Keep the last error and move on to the next retry
                last_err = e
                continue

        # All retries failed; raise the error for the upper layer to handle
        raise RuntimeError(f"OpenRouter LM request failed after {self.max_retries} retries: {last_err}")


