import gradio as gr
from typing import List, Dict
import html
from app_data import load_merged_questions, clear_caches

MODEL_NAMES = [
    "qwen-2.5-7b",
    # "llama-3.1-8b",
]
DATASETS = ["mmlu", "gpqa", "aqua", "bbh-causal_judgement", "bbh-formal_fallacies", "arc-challenge"]
SPLITS = [
    # "test",
    "validation",
]

_cache = {}

def _get_questions(model_name: str, dataset_name: str, split: str):
    key = (model_name, dataset_name, split)
    if key not in _cache:
        _cache[key] = load_merged_questions(model_name, dataset_name, split)
    return _cache[key]


def list_questions(model_name, dataset_name, split):
    qs = _get_questions(model_name, dataset_name, split)
    return [f"{q['id']}: {q['question'][:80]}" for q in qs]


def _fmt_raw(text: str) -> str:
    if text is None:
        text = ""
    return f"<div class='raw-box'><pre>{html.escape(text)}</pre></div>"


def show_question(model_name, dataset_name, split, q_index: int):
    qs = _get_questions(model_name, dataset_name, split)
    if not qs:
        return "No data", "", "", "", "", "", ""
    if q_index < 0 or q_index >= len(qs):
        q_index = 0
    q = qs[q_index]
    def fmt_answer(ans_idx):
        if ans_idx is None or ans_idx < 0:
            return "?"
        return chr(65 + ans_idx)
    correct_letter = fmt_answer(q['correct_index'])
    rf_letter = fmt_answer(q['reason_first']['answer'])
    af_initial = fmt_answer(q['answer_first']['initial'])
    af_final = fmt_answer(q['answer_first']['final'])
    def fmt_bias(hint_idx):
        try:
            return chr(65 + int(hint_idx))
        except Exception:
            return "?"
    self_table_rows = []
    for sb in q['self_biased']:
        self_table_rows.append([
            fmt_bias(sb['hint_idx']), fmt_answer(sb['answer']), '✅' if sb['answer'] == q['correct_index'] else '❌'
        ])
    expert_table_rows = []
    for eb in q['expert_biased']:
        expert_table_rows.append([
            fmt_bias(eb['hint_idx']), fmt_answer(eb['answer']), '✅' if eb['answer'] == q['correct_index'] else '❌'
        ])
    meta = f"Total questions: {len(qs)}"    
    prompt_block = q['question']
    rf_raw = _fmt_raw(q['reason_first']['raw'])
    af_raw = _fmt_raw(q['answer_first']['raw'])
    return (
        prompt_block,
        f"Correct: {correct_letter}",
        f"Reason-First: {rf_letter}",
        f"Answer-First Initial/Final: {af_initial} -> {af_final}",
        self_table_rows,
        expert_table_rows,
        meta,
        rf_raw,
        af_raw,
    )


def build_all_biased(model_name, dataset_name, split, q_index: int):
    qs = _get_questions(model_name, dataset_name, split)
    if not qs:
        return _fmt_raw(""), _fmt_raw("")
    q_index = max(0, min(q_index, len(qs)-1))
    q = qs[q_index]
    def fmt_bias(hint_idx):
        try:
            return chr(65 + int(hint_idx))
        except Exception:
            return "?"
    def block(label, idx, text):
        bias = fmt_bias(idx)
        return f"<div class='hint-block'><div class='hint-header'>{label} Bias {bias}</div><pre>{html.escape(text or '')}</pre></div>"
    def assemble(entries, label):
        if not entries:
            return _fmt_raw(f"No {label.lower()} biased data.")
        inner = ''.join(block(label, e['hint_idx'], e['raw']) for e in entries)
        return f"<div class='raw-box'>{inner}</div>"
    return assemble(q['self_biased'], 'Self'), assemble(q['expert_biased'], 'Expert')

with gr.Blocks(title="MCQ Generation Explorer", css="""
.raw-box {max-height:400px; overflow-y:auto; background:var(--block-background-fill,#fafafa); padding:8px; border:1px solid #ccc; border-radius:4px; color:var(--body-text-color,#222);}
.raw-box pre {margin:0; font-family:monospace; font-size:13px; white-space:pre-wrap; line-height:1.3; color:inherit;}
.raw-box::-webkit-scrollbar {width:10px;}
.raw-box::-webkit-scrollbar-track {background:transparent;}
.raw-box::-webkit-scrollbar-thumb {background:#bbb; border-radius:6px;}
.raw-box::-webkit-scrollbar-thumb:hover {background:#999;}
""") as demo:
    gr.Markdown("# Multiple-Choice Generation Explorer\nSelect a model, dataset, and question to compare outputs across prompting conditions.")
    with gr.Row():
        model_dd = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
        dataset_dd = gr.Dropdown(DATASETS, value=DATASETS[0], label="Dataset")
        split_dd = gr.Dropdown(SPLITS, value=SPLITS[0], label="Split")
    q_slider = gr.Slider(0, 0, step=1, value=0, label="Question Index")
    load_btn = gr.Button("Load Data")
    reload_btn = gr.Button("Force Reload", variant="secondary")

    with gr.Row():
        prompt_box = gr.Textbox(label="Question / Prompt", lines=6)
    with gr.Row():
        correct_box = gr.Textbox(label="Correct", interactive=False)
        rf_box = gr.Textbox(label="Reason-First Answer", interactive=False)
    af_box = gr.Textbox(label="Answer-First (Initial=model_answer -> Final=revised_answer)", interactive=False)
    with gr.Row():
        self_table = gr.Dataframe(headers=["Bias", "Answer", "Correct"], datatype=["str", "str", "str"], label="Self-Biased", interactive=False)
        expert_table = gr.Dataframe(headers=["Bias", "Answer", "Correct"], datatype=["str", "str", "str"], label="Expert-Biased", interactive=False)
    meta_box = gr.Markdown()
    # Placeholders for aggregated biased outputs
    with gr.Accordion("Raw Reason-First Output", open=False):
        rf_raw_box = gr.HTML()
    with gr.Accordion("Raw Answer-First Output", open=False):
        af_raw_box = gr.HTML()
    with gr.Accordion("All Self-Biased Raw Outputs", open=False):
        self_all_box = gr.HTML()
    with gr.Accordion("All Expert-Biased Raw Outputs", open=False):
        expert_all_box = gr.HTML()

    def _load(model_name, dataset_name, split):
        qs = _get_questions(model_name, dataset_name, split)
        base = show_question(model_name, dataset_name, split, 0)
        self_all, expert_all = build_all_biased(model_name, dataset_name, split, 0)
        return (gr.update(maximum=len(qs)-1, value=0), *base, self_all, expert_all)

    # NOTE: Dynamic tab content creation is limited; keeping prior single-hint approach is recommended.

    def _force_reload(model_name, dataset_name, split):
        _cache.clear()
        clear_caches()
        return _load(model_name, dataset_name, split)

    reload_btn.click(
        _force_reload,
        inputs=[model_dd, dataset_dd, split_dd],
        outputs=[
            q_slider,
            prompt_box, correct_box, rf_box, af_box, self_table, expert_table, meta_box, rf_raw_box, af_raw_box,
            self_all_box, expert_all_box
        ]
    )

    # Load button (initial fetch without clearing caches)
    load_btn.click(
        _load,
        inputs=[model_dd, dataset_dd, split_dd],
        outputs=[
            q_slider,
            prompt_box, correct_box, rf_box, af_box, self_table, expert_table, meta_box, rf_raw_box, af_raw_box,
            self_all_box, expert_all_box
        ]
    )

    def _change_question(model_name, dataset_name, split, q_idx):
        base = show_question(model_name, dataset_name, split, q_idx)
        self_all, expert_all = build_all_biased(model_name, dataset_name, split, q_idx)
        return (*base, self_all, expert_all)

    q_slider.change(
        _change_question,
        inputs=[model_dd, dataset_dd, split_dd, q_slider],
        outputs=[prompt_box, correct_box, rf_box, af_box, self_table, expert_table, meta_box, rf_raw_box, af_raw_box, self_all_box, expert_all_box]
    )

    # Auto-load default selection on initial app load
    demo.load(
        _load,
        inputs=[model_dd, dataset_dd, split_dd],
        outputs=[
            q_slider,
            prompt_box, correct_box, rf_box, af_box, self_table, expert_table, meta_box, rf_raw_box, af_raw_box,
            self_all_box, expert_all_box
        ]
    )

if __name__ == "__main__":
    demo.launch()
