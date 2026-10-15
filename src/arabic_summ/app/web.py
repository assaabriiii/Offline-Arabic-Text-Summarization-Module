"""Local Gradio UI (binds to 127.0.0.1 by default; analytics disabled; no external calls)."""

from __future__ import annotations

from typing import Mapping

from .service import SummarizerService, force_offline

CSS = ".rtl textarea, .rtl .prose {direction: rtl; text-align: right; font-size: 1.05rem;}"


def _accepted(fn, **kw) -> dict:
    """Keyword arguments the installed Gradio version accepts (Gradio 6 moved css/show_api)."""
    import inspect

    params = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in params}


def launch(demo, host: str, port: int) -> None:
    demo.launch(**_accepted(demo.launch, server_name=host, server_port=port, share=False, show_api=False,
                            css=CSS, inbrowser=False))


def build(service: SummarizerService, cfg: Mapping):
    force_offline()
    import gradio as gr

    from ..pipeline_info import STRATEGY_LABELS

    p = cfg["pipeline"]

    def run(text, ratio, strategy):
        try:
            r = service.summarize(text, ratio=float(ratio), strategy=strategy)
        except ValueError as e:
            return "", f"⚠️ {e}"
        warn = f" | ⚠️ numbers not found in the input: {' '.join(r['unsupported_numbers'])}" if r["unsupported_numbers"] else ""
        stats = (f"{r['input_words']} → {r['output_words']} words (allowed {r['bounds'][0]}-{r['bounds'][1]}, "
                 f"{'ok' if r['length_ok'] else 'outside bounds'}) | {r['seconds']['total']} s | "
                 f"{r['n_model_calls']} model calls | {r['variant']}{warn}")
        return r["summary"], stats

    with gr.Blocks(**_accepted(gr.Blocks.__init__, title="Arabic summarizer (offline)", css=CSS)) as demo:
        gr.Markdown("### ملخص النصوص العربية — يعمل محليا دون اتصال بالإنترنت")
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="النص (300–1000 كلمة)", lines=16, elem_classes="rtl")
                ratio = gr.Slider(p["min_ratio"], p["max_ratio"], value=p["default_ratio"], step=0.01,
                                  label="نسبة الملخص (10–30 %)")
                strat = gr.Dropdown(list(STRATEGY_LABELS), value=service.info.strategy, label="strategy")
                btn = gr.Button("لخص", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="الملخص", lines=12, elem_classes="rtl")
                stats = gr.Markdown()
        btn.click(run, [inp, ratio, strat], [out, stats])
    return demo
