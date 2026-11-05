import os
from typing import List
from src.models.openrouter import OpenRouterLM

class OpenRouterGenerator:
    def __init__(
        self,
        model_id: str,
        temperature=0.6,
        top_p=None,
        top_k=None,
        stop_strings=("Stop.",)
    ):
        self.lm = OpenRouterLM(
            model_id=model_id,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=list(stop_strings),
        )

    def generate(self, inputs: List[dict], max_len=192, repetition_penalty=1.0) -> List[str]:
        """
        Adapter for the existing eval framework:
        inputs: [{ "prefix": "...", "docstr": "...", "signature": "..." }]
        """
        outputs = []
        for inp in inputs:
            prompt = inp["prefix"]
            text = self.lm.generate(
                system=None,
                user=prompt,
                max_new_tokens=max_len
            )
            outputs.append(text)
        return outputs