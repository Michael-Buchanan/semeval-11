import json
import os
from dataclasses import dataclass
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

class LLMClient:
    """
    Abstract-ish wrapper so you can later plug in different backends.
    For now, we implement a GPT-5-based client using OpenAI's Python SDK.
    """

    def __init__(self, model: str = "gpt-5.1"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        System prompt that defines the canonical representation and gives ONE example.
        The example is the canine/fish/mammal syllogism we annotated earlier.
        """
        example_syllogism = (
            "Not all canines are aquatic creatures known as fish. "
            "It is certain that no fish belong to the class of mammals. "
            "Therefore, every canine falls under the category of mammals."
        )

        # Annotation for the example syllogism
        example_json = {
            "terms": ["Canine", "Fish", "Mammal"],
            "premises": [
                {
                    "quantifier": "SOME",
                    "subject": "Canine",
                    "predicate": "Fish",
                    "negated_predicate": True
                },
                {
                    "quantifier": "ALL",
                    "subject": "Fish",
                    "predicate": "Mammal",
                    "negated_predicate": True
                }
            ],
            "conclusion": {
                "quantifier": "ALL",
                "subject": "Canine",
                "predicate": "Mammal",
                "negated_predicate": False
            }
        }

        # System prompt + one concrete example
        prompt = f"""
You are a logic parser. Your only job is to transform English syllogisms into a small, 
formal, content-agnostic JSON representation. You MUST ignore real-world plausibility and 
only encode the logical quantifiers and predicates visible in the text.

Your output MUST be valid JSON with this exact structure:

{{
  "terms": [ "Term1", "Term2", ... ],

  "premises": [
    {{
      "quantifier": "ALL" | "SOME",
      "subject": "TermName",
      "predicate": "TermName",
      "negated_predicate": true | false
    }},
    ...
  ],

  "conclusion": {{
    "quantifier": "ALL" | "SOME",
    "subject": "TermName",
    "predicate": "TermName",
    "negated_predicate": true | false
  }}
}}

Semantics:

- "ALL" means: for all x, if subject(x) then (negated_predicate ? NOT predicate(x) : predicate(x)).
- "SOME" means: there exists an x such that subject(x) AND (negated_predicate ? NOT predicate(x) : predicate(x)).

You must map common phrasings to this scheme:

- "All A are B", "Every A is B", "Each A is B", "Anything that is A is B"
    => quantifier = "ALL", subject = A, predicate = B, negated_predicate = false

- "No A are B", "Nothing that is A is B", "It is true that no A is B"
    => quantifier = "ALL", subject = A, predicate = B, negated_predicate = true
       (because "no A are B" = "all A are not B")

- "Some A are B", "A number of A are B", "A portion of A are B"
    => quantifier = "SOME", subject = A, predicate = B, negated_predicate = false

- "Some A are not B", "A portion of A are not B"
    => quantifier = "SOME", subject = A, predicate = B, negated_predicate = true

- "Not all A are B"
    => quantifier = "SOME", subject = A, predicate = B, negated_predicate = true
       (because "not all A are B" = "some A are not B")

You must:
- Use exactly the same term name string for the same concept across the premises and conclusion.
- Introduce a finite set of term names, listed in "terms".
- Not invent or remove logical content: only encode what is explicitly stated in the syllogism.

EXAMPLE
=======

Input syllogism (English):
\"\"\"{example_syllogism}\"\"\"

Correct JSON output:
{json.dumps(example_json, indent=2)}

END OF EXAMPLE
===============

When I give you a new syllogism, you MUST respond with ONLY the JSON, no explanations.
"""
        return prompt.strip()

    def parse_syllogism(self, syllogism_text: str) -> Dict:
        """
        Calls LLM to parse the syllogism into the canonical JSON representation.
        """
        user_msg = f'Here is the syllogism:\n"""{syllogism_text}"""'

        response = self.client.responses.create(
                model=self.model,
                input = [
                    {
                        "role": "system",
                        "content": self.system_prompt + "\n" + user_msg,
                    },
                ]
            )

        content = response.output_text
        # Parse JSON from the model's output
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON. Raw output:\n{content}") from e

        return parsed


# =========================
# SYMBOLIC REASONER
# =========================

@dataclass
class Statement:
    quantifier: str          # "ALL" or "SOME"
    subject: str             # term name
    predicate: str           # term name
    negated_predicate: bool  # True = NOT predicate(x)


def _build_statements(structured: Dict) -> (List[str], List[Statement], Statement):
    """
    Convert the JSON dict from the LLM into internal Statement objects.
    """
    terms: List[str] = structured["terms"]

    # Quick sanity
    allowed_q = {"ALL", "SOME"}

    premises: List[Statement] = []
    for p in structured["premises"]:
        q = p["quantifier"].upper()
        if q not in allowed_q:
            raise ValueError(f"Unsupported quantifier in premise: {q}")
        premises.append(
            Statement(
                quantifier=q,
                subject=p["subject"],
                predicate=p["predicate"],
                negated_predicate=bool(p["negated_predicate"]),
            )
        )

    concl_raw = structured["conclusion"]
    cq = concl_raw["quantifier"].upper()
    if cq not in allowed_q:
        raise ValueError(f"Unsupported quantifier in conclusion: {cq}")
    conclusion = Statement(
        quantifier=cq,
        subject=concl_raw["subject"],
        predicate=concl_raw["predicate"],
        negated_predicate=bool(concl_raw["negated_predicate"]),
    )

    return terms, premises, conclusion


def _satisfies_statement(interp: Dict[str, List[bool]], stmt: Statement) -> bool:
    """
    Check if a given interpretation satisfies one Statement.

    interp: dict term_name -> list[bool] over domain indices (0..d-1)
    """
    subject_vals = interp[stmt.subject]
    pred_vals = interp[stmt.predicate]

    if stmt.quantifier == "ALL":
        # ∀x (subject(x) -> (neg ? !predicate(x) : predicate(x)))
        for i in range(len(subject_vals)):
            if subject_vals[i]:
                pred_val = pred_vals[i]
                if stmt.negated_predicate:
                    pred_val = not pred_val
                if not pred_val:
                    return False
        return True

    elif stmt.quantifier == "SOME":
        # ∃x (subject(x) & (neg ? !predicate(x) : predicate(x)))
        for i in range(len(subject_vals)):
            if subject_vals[i]:
                pred_val = pred_vals[i]
                if stmt.negated_predicate:
                    pred_val = not pred_val
                if pred_val:
                    return True
        return False

    else:
        raise ValueError(f"Unknown quantifier: {stmt.quantifier}")


def _generate_all_interpretations(terms: List[str], domain_size: int):
    """
    Generate all possible truth assignments for each term over a finite domain
    of size `domain_size`.

    For each interpretation, returns a dict: term_name -> list[bool].
    """
    num_terms = len(terms)
    total_bits = num_terms * domain_size
    # Each bit pattern encodes membership for all term/domain pairs.
    for mask in range(1 << total_bits):
        interp: Dict[str, List[bool]] = {}
        for t_idx, term in enumerate(terms):
            vals = []
            for i in range(domain_size):
                bit_index = t_idx * domain_size + i
                bit = (mask >> bit_index) & 1
                vals.append(bool(bit))
            interp[term] = vals
        yield interp


def is_valid_syllogism(structured: Dict, max_domain_size: int = 4) -> bool:
    """
    Decide validity by checking for a countermodel.

    The argument is valid iff there is NO interpretation (up to max_domain_size)
    such that all premises are true and the conclusion is false.

    This is sound and complete for monadic FOL with a finite number of predicates,
    given a sufficiently large finite domain bound. Here we use up to 4, which is
    enough for small syllogistic cases.
    """
    terms, premises, conclusion = _build_statements(structured)

    for d in range(1, max_domain_size + 1):
        for interp in _generate_all_interpretations(terms, d):
            # Check premises
            if all(_satisfies_statement(interp, p) for p in premises):
                # Check if conclusion fails
                if not _satisfies_statement(interp, conclusion):
                    # Found a countermodel: premises true, conclusion false
                    return False

    # No countermodel found up to bound => treat as valid
    return True


# =========================
# MAIN
# =========================

def main():
    # Example from the task description (canine/fish/mammal)
    syllogism = (
        "Not all canines are aquatic creatures known as fish. "
        "It is certain that no fish belong to the class of mammals. "
        "Therefore, every canine falls under the category of mammals."
    )

    # Gold label from the dataset -- will need to replace this with actual data later
    gold_validity = False

    llm = LLMClient(model="gpt-5.1")  
    parsed = llm.parse_syllogism(syllogism)

    print("Parsed canonical form:")
    print(json.dumps(parsed, indent=2))

    predicted_validity = is_valid_syllogism(parsed)
    print(f"\nPredicted validity: {predicted_validity}")
    print(f"Gold validity:      {gold_validity}")


if __name__ == "__main__":
    main()
