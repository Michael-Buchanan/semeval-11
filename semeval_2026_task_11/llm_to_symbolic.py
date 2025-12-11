import json
import os
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv
#from symbolic_solver import is_valid_syllogism

load_dotenv()

# =========================
# LLM CLIENT
# =========================

class LLMClient:
  """
  Abstract-ish wrapper so you can later plug in different backends.
  For now, we implement a GPT-5-based client using OpenAI's Python SDK.
  """

  import json
import os
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================
# LLM CLIENT
# =========================

class LLMClient:
    """
    Wrapper around GPT-5.1 that turns English syllogisms into
    a canonical monadic logic format.
    """

    def __init__(self, model: str = "gpt-5.1"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        prompt = """
You are a logic parser. Your only job is to transform English syllogisms into a small,
formal, content-agnostic CANONICAL TEXT representation (not JSON).

------------------------------------------------------------
1. IDENTIFY PREMISES VS CONCLUSION
------------------------------------------------------------

The syllogism is given as 2–3 English sentences.

- Sentences that appear BEFORE words like:
    "Therefore", "Thus", "Hence", "Consequently",
    "It follows that", "It logically follows that",
    "As a result", "It is therefore the case that"
  are PREMISES.

- The sentence introduced by one of these cue phrases
  is the CONCLUSION.

- If there is NO explicit cue phrase, treat the LAST sentence
  as the CONCLUSION, and all previous sentences as PREMISES.

IMPORTANT:
- DO NOT list the conclusion as a PREMISE.
- Each logical statement should appear exactly once:
  as either a PREMISE_k or as the CONCLUSION, not both.

------------------------------------------------------------
2. CANONICAL FORMAT
------------------------------------------------------------

You MUST output exactly in this format:

TERMS: Term1, Term2, Term3

PREMISE 1: ALL TermA TermB
PREMISE 2: SOME TermC NOT TermD
...

CONCLUSION: ALL TermX TermY
or
CONCLUSION: SOME TermX NOT TermY

------------------------------------------------------------
3. RULES FOR TERM NAMES
------------------------------------------------------------

- Each term name MUST be a single identifier with no spaces, e.g.:
  Canine, Fish, Mammal, TeachingFellow, ZoneOfLife.
- Reuse the SAME identifier for the SAME concept across all premises
  and the conclusion.
- Do NOT introduce extra terms that are not in the syllogism.
- If the English surface form has spaces, convert to a single identifier:
  "odd-toed ungulate" -> OddToedUngulate
  "area of dense tropical forest" -> AreaOfDenseTropicalForest

------------------------------------------------------------
4. QUANTIFIERS AND NEGATION
------------------------------------------------------------

- Quantifier is either ALL or SOME (UPPERCASE).

- If the predicate is negated, insert the word NOT before it.
    Example: SOME Canine NOT Fish
- If the predicate is not negated, omit NOT.
    Example: ALL Canine Mammal

Logical mapping examples:
- "All A are B", "Every A is B", "Each A is B"
    => ALL A B
- "No A are B", "Nothing that is A is B"
    => ALL A NOT B
- "Some A are B"
    => SOME A B
- "Some A are not B"
    => SOME A NOT B
- "Not all A are B"
    => SOME A NOT B

------------------------------------------------------------
5. YOUR TASK
------------------------------------------------------------

- Read the English syllogism.
- Split into premises and conclusion as described above.
- Identify the minimal set of abstract terms.
- Write the TERMS line.
- Write each premise as a PREMISE k: ... line.
- Write the conclusion as the CONCLUSION: ... line.
- IGNORE real-world plausibility; treat terms as abstract symbols.
- Output ONLY these lines, no extra prose, no JSON, no explanations.

------------------------------------------------------------
6. EXAMPLES
------------------------------------------------------------

EXAMPLE 1
----------------------------
English:
"Not all canines are aquatic creatures known as fish.
It is certain that no fish belong to the class of mammals.
Therefore, every canine falls under the category of mammals."

Canonical output:
TERMS: Canine, Fish, Mammal
PREMISE 1: SOME Canine NOT Fish
PREMISE 2: ALL Fish NOT Mammal
CONCLUSION: ALL Canine Mammal

EXAMPLE 2
----------------------------
English:
"All birds lay eggs. All chickens lay eggs. All chickens are birds."

Here the syllogism has no explicit cue word, so:
- First two sentences are PREMISES.
- LAST sentence is the CONCLUSION.

Canonical output:
TERMS: Bird, Chicken, EggLayer
PREMISE 1: ALL Bird EggLayer
PREMISE 2: ALL Chicken EggLayer
CONCLUSION: ALL Chicken Bird

EXAMPLE 3
----------------------------
English:
"Every sanctuary is classified as a type of place.
All sanctuaries belong to the category of geographical areas.
It follows that some geographical areas are indeed places."

Canonical output:
TERMS: Sanctuary, Place, GeographicalArea
PREMISE 1: ALL Sanctuary Place
PREMISE 2: ALL Sanctuary GeographicalArea
CONCLUSION: SOME GeographicalArea Place

============================
END OF EXAMPLES
============================

When I give you a new syllogism, you MUST respond with ONLY the
canonical lines in this format (TERMS / PREMISE / CONCLUSION).
""".strip()

        return prompt

    def parse_syllogism(self, syllogism_text: str) -> Dict[str, Any]:
        """
        Step A: Call LLM to produce canonical textual form.
        Step B: Parse canonical text into the JSON structure expected
                by is_valid_syllogism().
        """
        user_msg = f'Here is the syllogism:\n"""{syllogism_text}"""'

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

        canonical_text = response.output_text.strip()
        structured = canonical_to_json(canonical_text)
        return structured

def canonical_to_json(canonical: str) -> Dict[str, Any]:
    """
    Parse the canonical textual form:

        TERMS: A, B, C
        PREMISE 1: ALL A B
        PREMISE 2: SOME A NOT C
        CONCLUSION: SOME B NOT C

    into the JSON structure:

    {
      "terms": [...],
      "premises": [
        {"quantifier": "ALL", "subject": "A", "predicate": "B", "negated_predicate": False},
        ...
      ],
      "conclusion": {...}
    }

    Assumes the model followed the format exactly.
    Raises ValueError if something important is missing.
    """
    # Strip blank lines
    lines = [ln.strip() for ln in canonical.splitlines() if ln.strip()]
    terms: List[str] = []
    premises: List[Dict[str, Any]] = []
    conclusion: Optional[Dict[str, Any]] = None

    # Helper to parse a "Q S [NOT] P" clause
    def parse_clause(tokens: List[str]) -> Dict[str, Any]:
        if len(tokens) < 3:
            raise ValueError(f"Cannot parse clause from tokens: {tokens}")
        q = tokens[0].upper()
        if q not in {"ALL", "SOME"}:
            raise ValueError(f"Unknown quantifier: {q}")

        subj = tokens[1]

        if len(tokens) == 3:
            # Q S P
            neg = False
            pred = tokens[2]
        else:
            # Q S NOT P
            if tokens[2].upper() != "NOT":
                raise ValueError(f"Unexpected token (expected NOT): {tokens[2]}")
            if len(tokens) != 4:
                raise ValueError(f"Cannot parse NOT-clause from tokens: {tokens}")
            neg = True
            pred = tokens[3]

        return {
            "quantifier": q,
            "subject": subj,
            "predicate": pred,
            "negated_predicate": neg,
        }

    # Main parse loop
    for line in lines:
        upper = line.upper()

        # TERMS line
        if upper.startswith("TERMS:"):
            after = line.split(":", 1)[1]
            raw_terms = [t.strip() for t in after.split(",") if t.strip()]
            if not raw_terms:
                raise ValueError("TERMS line found but no terms parsed.")
            terms = raw_terms

        # PREMISE lines
        elif upper.startswith("PREMISE"):
            # Pattern: "PREMISE k: Q S [NOT] P"
            m = re.match(r"PREMISE\s+\d+\s*:\s*(.+)$", line, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"Could not parse PREMISE line: {line}")
            clause_str = m.group(1).strip()
            tokens = clause_str.split()
            premises.append(parse_clause(tokens))

        # CONCLUSION line
        elif upper.startswith("CONCLUSION:"):
            after = line.split(":", 1)[1].strip()
            tokens = after.split()
            conclusion = parse_clause(tokens)

        else:
            # Ignore stray lines (ideally there are none)
            continue

    if not terms:
        raise ValueError("No TERMS line parsed from canonical output.")
    if not premises:
        raise ValueError("No PREMISE lines parsed from canonical output.")
    if conclusion is None:
        raise ValueError("No CONCLUSION line parsed from canonical output.")

    # Optionally ensure that all mentioned term names are in the terms list
    mentioned_terms = set()
    for p in premises:
        mentioned_terms.add(p["subject"])
        mentioned_terms.add(p["predicate"])
    mentioned_terms.add(conclusion["subject"])
    mentioned_terms.add(conclusion["predicate"])

    for t in sorted(mentioned_terms):
        if t not in terms:
            terms.append(t)

    return {
        "terms": terms,
        "premises": premises,
        "conclusion": conclusion,
    }