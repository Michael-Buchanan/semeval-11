import json
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv
from analysis import LOG_PATH, analyze_log_file, run_pilot_evaluation
from symbolic_solver import is_valid_syllogism

load_dotenv()

# =========================
# LLM CLIENT
# =========================

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
        System prompt that defines the canonical representation and gives
        TEN concrete examples (1 canonical + 9 from pilot errors).
        """
        prompt = """
You are a logic parser. Your only job is to transform English syllogisms into a small, 
formal, content-agnostic JSON representation. You MUST ignore real-world plausibility and 
only encode the logical quantifiers and predicates visible in the text.

Your output MUST be valid JSON with this exact structure:

{
  "terms": [ "Term1", "Term2", ... ],

  "premises": [
    {
      "quantifier": "ALL" | "SOME",
      "subject": "TermName",
      "predicate": "TermName",
      "negated_predicate": true | false
    },
    ...
  ],

  "conclusion": {
    "quantifier": "ALL" | "SOME",
    "subject": "TermName",
    "predicate": "TermName",
    "negated_predicate": true | false
  }
}

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
- Respond with ONLY the JSON object, no explanations, no extra text.

====================================================
EXAMPLE 1
----------------------------------------------------
Input syllogism (English):
Not all canines are aquatic creatures known as fish. It is certain that no fish belong to the class of mammals. Therefore, every canine falls under the category of mammals.

Correct JSON output:
{
  "terms": ["Canine", "Fish", "Mammal"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Canine",
      "predicate": "Fish",
      "negated_predicate": true
    },
    {
      "quantifier": "ALL",
      "subject": "Fish",
      "predicate": "Mammal",
      "negated_predicate": true
    }
  ],
  "conclusion": {
    "quantifier": "ALL",
    "subject": "Canine",
    "predicate": "Mammal",
    "negated_predicate": false
  }
}

====================================================
EXAMPLE 2 (ID 1)
----------------------------------------------------
Input syllogism (English):
All birds lay eggs. All chickens lay eggs. All chickens are birds.

Correct JSON output:
{
  "terms": ["Bird", "Chicken", "EggLayer"],
  "premises": [
    {
      "quantifier": "ALL",
      "subject": "Bird",
      "predicate": "EggLayer",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Chicken",
      "predicate": "EggLayer",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Chicken",
      "predicate": "Bird",
      "negated_predicate": false
    }
  ],
  "conclusion": {
    "quantifier": "ALL",
    "subject": "Chicken",
    "predicate": "Bird",
    "negated_predicate": false
  }
}

====================================================
EXAMPLE 3 (ID 2)
----------------------------------------------------
Input syllogism (English):
There are some bridges that are considered artifacts. All items categorized as structures are artifacts. It is true that every bridge is a structure.

Correct JSON output:
{
  "terms": ["Bridge", "Artifact", "Structure"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Bridge",
      "predicate": "Artifact",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Structure",
      "predicate": "Artifact",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Bridge",
      "predicate": "Structure",
      "negated_predicate": false
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Bridge",
    "predicate": "Artifact",
    "negated_predicate": false
  }
}

====================================================
EXAMPLE 4 (ID 4)
----------------------------------------------------
Input syllogism (English):
There are certain activities that can be classified as coverings. Some acts of protecting function as coverings. Every act of protecting qualifies as an activity.

Correct JSON output:
{
  "terms": ["Activity", "Covering", "Act of protecting"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Activity",
      "predicate": "Covering",
      "negated_predicate": false
    },
    {
      "quantifier": "SOME",
      "subject": "Act of protecting",
      "predicate": "Covering",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Act of protecting",
      "predicate": "Activity",
      "negated_predicate": false
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Activity",
    "predicate": "Covering",
    "negated_predicate": false
  }
}

====================================================
EXAMPLE 5 (ID 5)
----------------------------------------------------
Input syllogism (English):
Some sadnesses are not trees. All sadnesses are emotions. No trees are emotions.

Correct JSON output:
{
  "terms": ["Sadness", "Tree", "Emotion"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Sadness",
      "predicate": "Tree",
      "negated_predicate": true
    },
    {
      "quantifier": "ALL",
      "subject": "Sadness",
      "predicate": "Emotion",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Tree",
      "predicate": "Emotion",
      "negated_predicate": true
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Sadness",
    "predicate": "Tree",
    "negated_predicate": true
  }
}

====================================================
EXAMPLE 6 (ID 9)
----------------------------------------------------
Input syllogism (English):
A portion of sculptures are considered plastic arts. Statues are among the various forms of plastic arts. Every single statue is a type of sculpture.

Correct JSON output:
{
  "terms": ["Sculpture", "Plastic art", "Statue"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Sculpture",
      "predicate": "Plastic art",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Statue",
      "predicate": "Plastic art",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Statue",
      "predicate": "Sculpture",
      "negated_predicate": false
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Sculpture",
    "predicate": "Plastic art",
    "negated_predicate": false
  }
}

====================================================
EXAMPLE 7 (ID 10)
----------------------------------------------------
Input syllogism (English):
A fraction of all organisms are classified as postdiluvian. Among those that are postdiluvian, a certain group does not include succulents. No members of the organism kingdom are succulents.

Correct JSON output:
{
  "terms": ["Organism", "Postdiluvian", "Succulent"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Organism",
      "predicate": "Postdiluvian",
      "negated_predicate": false
    },
    {
      "quantifier": "SOME",
      "subject": "Postdiluvian",
      "predicate": "Succulent",
      "negated_predicate": true
    },
    {
      "quantifier": "ALL",
      "subject": "Organism",
      "predicate": "Succulent",
      "negated_predicate": true
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Organism",
    "predicate": "Succulent",
    "negated_predicate": true
  }
}

====================================================
EXAMPLE 8 (ID 14)
----------------------------------------------------
Input syllogism (English):
There exist families that are also foster families. Some of the individuals who are foster families are not hammers. No family is a hammer.

Correct JSON output:
{
  "terms": ["Family", "FosterFamily", "Hammer"],
  "premises": [
    {
      "quantifier": "SOME",
      "subject": "Family",
      "predicate": "FosterFamily",
      "negated_predicate": false
    },
    {
      "quantifier": "SOME",
      "subject": "FosterFamily",
      "predicate": "Hammer",
      "negated_predicate": true
    },
    {
      "quantifier": "ALL",
      "subject": "Family",
      "predicate": "Hammer",
      "negated_predicate": true
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Family",
    "predicate": "Hammer",
    "negated_predicate": true
  }
}

====================================================
EXAMPLE 9 (ID 21)
----------------------------------------------------
Input syllogism (English):
Every single plankton is an organism. No cheese is a plankton. Consequently, there exist organisms that are not cheeses.

Correct JSON output:
{
  "terms": ["Plankton", "Organism", "Cheese"],
  "premises": [
    {
      "quantifier": "ALL",
      "subject": "Plankton",
      "predicate": "Organism",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Cheese",
      "predicate": "Plankton",
      "negated_predicate": true
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "Organism",
    "predicate": "Cheese",
    "negated_predicate": true
  }
}

====================================================
EXAMPLE 10 (ID 23)
----------------------------------------------------
Input syllogism (English):
Every sanctuary is classified as a type of place. All sanctuaries belong to the category of geographical areas. It follows that some geographical areas are indeed places.

Correct JSON output:
{
  "terms": ["Sanctuary", "Place", "GeographicalArea"],
  "premises": [
    {
      "quantifier": "ALL",
      "subject": "Sanctuary",
      "predicate": "Place",
      "negated_predicate": false
    },
    {
      "quantifier": "ALL",
      "subject": "Sanctuary",
      "predicate": "GeographicalArea",
      "negated_predicate": false
    }
  ],
  "conclusion": {
    "quantifier": "SOME",
    "subject": "GeographicalArea",
    "predicate": "Place",
    "negated_predicate": false
  }
}

====================================================
END OF EXAMPLES
====================================================

When I give you a new syllogism, you MUST respond with ONLY the JSON, no explanations.
"""
        return prompt.strip()

    def parse_syllogism(self, syllogism_text: str) -> Dict:
        """
        Calls LLM to parse the syllogism into the canonical JSON representation.
        """
        user_msg = f'Here is the syllogism:\n"""{syllogism_text}"""'

        # Using Responses API as in your code
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt + "\n" + user_msg,
                },
            ],
        )

        content = response.output_text
        # Parse JSON from the model's output
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON. Raw output:\n{content}") from e

        return parsed

# =========================
# MAIN MENU
# =========================

def main():
    print("=== Syllogistic Reasoning Evaluation ===")
    print("1) Run full pilot evaluation (LLM + solver + write log)")
    print("2) Compute metrics from existing log file")
    choice = input("Select option (1/2): ").strip()

    if choice == "1":
        run_pilot_evaluation()
    elif choice == "2":
        path = input(f"Enter log file path (default: {LOG_PATH}): ").strip()
        if not path:
            path = LOG_PATH
        analyze_log_file(path)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
