"""Data loading and merging utilities for interactive Gradio explorer.

This module aggregates the separately generated JSONL datasets (reason-first,
answer-first, self-biased, expert-biased) produced by generate_responses.
It exposes a single function `load_merged_questions` returning a list of
question-level dicts suitable for UI consumption.
"""
from functools import lru_cache
from typing import Optional

from thoughts.multiple_choice import load_data  # type: ignore
from thoughts.utils import get_choices  # type: ignore


def _choices(dataset_name: str):
    return get_choices(dataset_name)


@lru_cache(maxsize=8)
def _load_variant(model_name: str, dataset_name: str, reason_first: bool, bias: Optional[str], hint_idx: Optional[int], split: str):
    return load_data(model_name, dataset_name, reason_first, bias, hint_idx, split)


def load_merged_questions(model_name: str, dataset_name: str, split: str):
    """Merge all variants for a (model, dataset, split) into question dicts.

    Returns
    -------
    list[dict]
        Each dict contains: id, question, options, correct_index, reason_first,
        answer_first, self_biased, expert_biased.
    """
    # Base datasets (unbiased)
    rf = _load_variant(model_name, dataset_name, True, None, None, split)
    af = _load_variant(model_name, dataset_name, False, None, None, split)
    choices = _choices(dataset_name)
    n_hints = len(choices)
    self_sets = [
        _load_variant(model_name, dataset_name, False, 'self', h, split)
        for h in range(n_hints)
    ]
    expert_sets = [
        _load_variant(model_name, dataset_name, True, 'expert', h, split)
        for h in range(n_hints)
    ]
    n = len(rf)
    results = []
    for i in range(n):
        ex_rf = rf[i]
        ex_af = af[i]
        # Attempt to recover question text; fallback to prefix of model_output
        question_text = ex_rf.get('question', ex_rf.get('model_output', '')[:300])
        options = [f"{chr(65+j)}" for j in range(len(choices))]

        self_biased = []
        for h, ds in enumerate(self_sets):
            ex = ds[i]
            self_biased.append({
                'hint_idx': h,
                'answer': ex.get('model_answer'),
                'raw': ex.get('model_output', ''),
            })

        expert_biased = []
        for h, ds in enumerate(expert_sets):
            ex = ds[i]
            expert_biased.append({
                'hint_idx': h,
                'answer': ex.get('model_answer'),
                'raw': ex.get('model_output', ''),
            })

        # Use full stored model_output (prompt + generation)
        rf_gen_text = ex_rf.get('model_output', '')
        af_gen_text = ex_af.get('model_output', '')

        results.append({
            'id': i,
            'question': question_text,
            'options': options,
            'correct_index': ex_rf.get('correct_answer'),
            'reason_first': {
                'answer': ex_rf.get('model_answer'),
                'raw': rf_gen_text,
            },
            'answer_first': {
                # New schema: initial answer now stored under model_answer, final under revised_answer
                # Backward compatibility: fall back to old keys if new ones absent
                'initial': ex_af.get('model_answer', ex_af.get('initial_answer')),
                'final': ex_af.get('revised_answer', ex_af.get('model_answer')),  # if revised missing, use model_answer
                'raw': af_gen_text,
            },
            'self_biased': self_biased,
            'expert_biased': expert_biased,
        })
    print(f"Loaded {len(results)} questions for {model_name} {dataset_name} {split}")
    return results


__all__ = ['load_merged_questions']


def clear_caches():  # Utility for UI to force-refresh datasets
    try:
        _load_variant.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

__all__.append('clear_caches')
