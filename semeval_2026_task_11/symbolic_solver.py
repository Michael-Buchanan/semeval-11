import json
import os
from dataclasses import dataclass
from typing import List, Dict

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