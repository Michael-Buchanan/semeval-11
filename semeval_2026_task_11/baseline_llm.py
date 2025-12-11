import json
import os
from dataclasses import dataclass
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMJudgment:
    validity: bool
    plausibility: bool


class GPT5BaselineClient:
    """
    End-to-end GPT-5 classifier baseline for the SemEval syllogistic reasoning task.
    No symbolic parser; model directly predicts validity (and plausibility).
    """

    def __init__(self, model: str = "gpt-5.1"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """
You are a careful logician evaluating English syllogisms.

Your job:
1. Decide whether the argument is **logically valid**:
   - VALID means: in classical first-order logic with standard interpretation
     of quantifiers, the conclusion must be true in every possible world
     where all the premises are true (i.e., no counterexample).
   - INVALID means: there exists at least one possible world where all premises
     are true and the conclusion is false (i.e., a counterexample is possible).

2. Decide whether the conclusion is **plausible in the real world**:
   - PLAUSIBLE (true) means the conclusion sounds factually realistic/typical
     given common real-world knowledge.
   - IMPLAUSIBLE (false) means the conclusion clashes with real-world knowledge.

IMPORTANT:
- When judging **validity**, completely ignore real-world plausibility.
  Treat predicates as abstract sets; use pure logical form.
- When judging **plausibility**, use normal world knowledge.

OUTPUT FORMAT (VERY IMPORTANT):
- You must output a single line containing exactly one valid JSON object:
  {
    "validity": true or false,
    "plausibility": true or false
  }

- Use lowercase true/false (valid JSON booleans).
- Do NOT add any extra text, comments, or explanations.
- Do NOT wrap the JSON in markdown.
""".strip()

    def judge_syllogism(self, syllogism_text: str) -> LLMJudgment:
        """
        Send the raw syllogism text to GPT-5 and parse its JSON output into
        an LLMJudgment(validity=bool, plausibility=bool).
        """

        user_msg = f"""Here is the syllogism you must evaluate:

\"\"\"{syllogism_text}\"\"\""""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_msg,
                },
            ],
        )

        raw = response.output_text.strip()

        # Robust JSON extraction in case the model adds stray text anyway
        try:
            json_str = raw[raw.index("{"): raw.rindex("}") + 1]
        except ValueError:
            raise ValueError(f"Could not find JSON object in model output: {raw!r}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from model output: {json_str!r}") from e

        if "validity" not in data:
            raise ValueError(f"JSON missing 'validity' field: {data}")
        if "plausibility" not in data:
            raise ValueError(f"JSON missing 'plausibility' field: {data}")

        validity = bool(data["validity"])
        plausibility = bool(data["plausibility"])

        return LLMJudgment(validity=validity, plausibility=plausibility)