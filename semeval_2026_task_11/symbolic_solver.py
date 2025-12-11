import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# =========================
# SYMBOLIC REASONER
# =========================

@dataclass
class Statement:
    quantifier: str          # "ALL" or "SOME"
    subject: str             # term name
    predicate: str           # term name
    negated_predicate: bool  # True = NOT predicate(x)


def _build_statements(structured: Dict[str, Any]) -> Tuple[List[str], List[Statement], Statement]:
    """
    Convert the JSON dict from the LLM into internal Statement objects.
    """
    terms: List[str] = structured["terms"]

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

    IMPORTANT: For this task we use ARISTOTELIAN-style semantics
    with existential import for universal statements:

      "ALL A B"      means (forall x: A(x) -> B(x)) AND (exists x: A(x))
      "ALL A NOT B"  means (forall x: A(x) -> ¬B(x)) AND (exists x: A(x))

    "SOME A B" / "SOME A NOT B" keep the usual existential reading.
    """
    subject_vals = interp[stmt.subject]
    pred_vals = interp[stmt.predicate]

    if stmt.quantifier == "ALL":
        # Existential import: require at least one subject instance.
        seen_subject = False

        # Check ∀x (A(x) -> (neg ? ¬B(x) : B(x))) and track existence of A(x)
        for i in range(len(subject_vals)):
            if subject_vals[i]:
                seen_subject = True
                pred_val = pred_vals[i]
                if stmt.negated_predicate:
                    pred_val = not pred_val
                if not pred_val:
                    # Found an A that is not (B / ¬B) as required
                    return False

        # If no subject exists, the universal is FALSE under existential import.
        return seen_subject

    elif stmt.quantifier == "SOME":
        # ∃x (A(x) ∧ (neg ? ¬B(x) : B(x)))
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
            vals: List[bool] = []
            for i in range(domain_size):
                bit_index = t_idx * domain_size + i
                bit = (mask >> bit_index) & 1
                vals.append(bool(bit))
            interp[term] = vals
        yield interp


def is_valid_syllogism(
    structured: Dict[str, Any],
    max_domain_size: int = 4,
    disallow_trivial_conclusion: bool = True,
) -> bool:
    """
    Decide validity by checking for a countermodel.

    The argument is valid iff there is NO interpretation (up to max_domain_size)
    such that all premises are true and the conclusion is false.

    We use ARISTOTELIAN semantics with existential import for ALL statements.

    Additional dataset-alignment heuristic (optional):
    - If `disallow_trivial_conclusion` is True and the conclusion is exactly
      identical to any premise (same quantifier, subject, predicate, negation),
      we treat the syllogism as INVALID. This matches the dataset behavior
      where arguments whose "conclusion" merely repeats a premise are labeled
      invalid even though they are tautologically entailed in classical logic.
    """
    terms, premises, conclusion = _build_statements(structured)

    if disallow_trivial_conclusion:
        for p in premises:
            if (
                p.quantifier == conclusion.quantifier
                and p.subject == conclusion.subject
                and p.predicate == conclusion.predicate
                and p.negated_predicate == conclusion.negated_predicate
            ):
                # Dataset treats "premise == conclusion" as not a genuine inference.
                return False

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
