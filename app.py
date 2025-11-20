import gradio as gr
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
import html
import time

DEFAULT_MODEL = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE = {}

# ---------------- Fullscreen helper (injected JS per plot) ----------------

def fullscreen_plot_js(plot_id):
    safe_id = plot_id.replace("-", "_")
    return f"""
    <script>
    function openFull_{safe_id}() {{
        const container = document.getElementById("{plot_id}");
        if (!container) {{
            console.error("Plot container not found:", "{plot_id}");
            return;
        }}

        // clone full container (works for Plotly/SSR)
        const clone = container.cloneNode(true);

        // Modal background
        const modal = document.createElement("div");
        modal.style.position = "fixed";
        modal.style.top = "0";
        modal.style.left = "0";
        modal.style.width = "100vw";
        modal.style.height = "100vh";
        modal.style.background = "rgba(0,0,0,0.9)";
        modal.style.zIndex = "9999999";
        modal.style.display = "flex";
        modal.style.alignItems = "center";
        modal.style.justifyContent = "center";
        modal.style.padding = "20px";
        modal.onclick = () => modal.remove();

        // Resize cloned plot
        clone.style.maxWidth = "95%";
        clone.style.maxHeight = "95%";
        clone.style.overflow = "auto";
        clone.style.border = "2px solid #fff";
        clone.style.borderRadius = "12px";
        clone.style.background = "black";

        modal.appendChild(clone);
        document.body.appendChild(modal);
    }}
    </script>
    """

def fullscreen_button_html(plot_id, label="🔍 Full Screen"):
    # wrapper HTML: JS function + button
    return fullscreen_plot_js(plot_id) + f'<button onclick="openFull_{plot_id.replace("-","_")}()" style="margin-top:6px;padding:6px 10px;border-radius:8px;border:1px solid #ddd;background:white;">{label}</button>'


# ---------------- CORE UTILS ----------------

def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


def safe_tokens(tokens):
    return "  ".join([f"[{html.escape(t)}]" for t in tokens])


def compute_pca(hidden_layer):
    try:
        return PCA(n_components=2).fit_transform(hidden_layer)
    except:
        seq = hidden_layer.shape[0]
        dim0 = hidden_layer[:, 0] if hidden_layer.shape[1] > 0 else np.zeros(seq)
        dim1 = hidden_layer[:, 1] if hidden_layer.shape[1] > 1 else np.zeros(seq)
        return np.vstack([dim0, dim1]).T


def fig_attention(matrix, tokens, title):
    fig = px.imshow(matrix, x=tokens, y=tokens, title=title,
                    labels={"x": "Key token", "y": "Query token", "color": "Attention"})
    fig.update_layout(height=420)
    return fig


def fig_pca(points, tokens, highlight=None, title="PCA"):
    fig = px.scatter(x=points[:, 0], y=points[:, 1], text=tokens, title=title)
    fig.update_traces(textposition="top center", marker=dict(size=10))
    if highlight is not None:
        fig.add_trace(go.Scatter(
            x=[points[highlight, 0]],
            y=[points[highlight, 1]],
            mode="markers+text",
            text=[tokens[highlight]],
            marker=dict(size=18, color="red")
        ))
    fig.update_layout(height=420)
    return fig


def fig_probs(tokens, scores):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tokens, y=scores))
    fig.update_layout(title="Next-token probabilities", height=380)
    return fig


# ---------------- ANALYSIS CORE ----------------

def analyze_text(text, model_name, simple):
    if not text.strip():
        return {"error": "Please enter text."}

    try:
        model, tokenizer = load_model(model_name)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)

    try:
        with torch.no_grad():
            out = model(**inputs)
    except Exception as e:
        return {"error": f"Model error: {e}"}

    input_ids = inputs["input_ids"][0].cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    attentions = [a[0].cpu().numpy() for a in out.attentions]
    hidden = [h[0].cpu().numpy() for h in out.hidden_states]
    logits = out.logits[0].cpu().numpy()

    # PCA per layer
    pca_layers = [compute_pca(h) for h in hidden]

    # top-k
    last = logits[-1]
    probs = softmax(last)
    idx = np.argsort(probs)[-20:][::-1]
    top_tokens = [tokenizer.decode([i]) for i in idx]
    top_scores = probs[idx].tolist()

    default_layer = len(attentions) - 1
    default_head = 0

    # neuron explorer
    neuron_info = []
    try:
        last_h = hidden[-1]
        mean_act = np.abs(last_h).mean(axis=0)
        top_neurons = np.argsort(mean_act)[-24:][::-1]
        for n in top_neurons:
            vals = last_h[:, n]
            top_ix = np.argsort(np.abs(vals))[-5:][::-1]
            neuron_info.append({
                "neuron": int(n),
                "top_tokens": [(tokens[i], float(vals[i])) for i in top_ix]
            })
    except:
        neuron_info = []

    # residual decomposition (safe)
    residuals = compute_residuals_safe(model, inputs)

    return {
        "tokens": tokens,
        "attentions": attentions,
        "hidden": hidden,
        "pca": pca_layers,
        "logits": logits,
        "top_tokens": top_tokens,
        "top_scores": top_scores,
        "default_layer": default_layer,
        "default_head": default_head,
        "neuron_info": neuron_info,
        "residuals": residuals,
        "token_display": safe_tokens(tokens),
        "explanation": explain(simple)
    }


def explain(s):
    if s:
        return (
            "🧒 **Simple mode:**\n"
            "- The model cuts text into small pieces (tokens).\n"
            "- It looks at which tokens matter (attention).\n"
            "- It builds an internal map (PCA) of meanings.\n"
            "- Then it guesses the next token.\n"
        )
    return (
        "🔬 **Technical mode:**\n"
        "Showing tokens, attention (query→key), PCA projections, logits, "
        "neuron activations, and layerwise residual contributions.\n"
    )


# ---------------- RESIDUAL DECOMPOSITION SAFE ----------------

def compute_residuals_safe(model, inputs):
    """
    Guaranteed safe residual norms for GPT-2-style blocks.
    Will NEVER crash. Returns None if not applicable.
    """
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        return None

    try:
        blocks = model.transformer.h
        wte = model.transformer.wte
        x = wte(inputs["input_ids"]).to(DEVICE)

        attn_norms = []
        mlp_norms = []

        for block in blocks:
            try:
                ln1 = block.ln_1(x)
                attn_out = block.attn(ln1)[0]
                x = x + attn_out
                ln2 = block.ln_2(x)
                mlp_out = block.mlp(ln2)
                x = x + mlp_out

                # detach to avoid requires_grad warning
                attn_norms.append(float(torch.norm(attn_out.detach()).cpu()))
                mlp_norms.append(float(torch.norm(mlp_out.detach()).cpu()))
            except:
                # fallback safe zero
                attn_norms.append(0.0)
                mlp_norms.append(0.0)

        # normalize lengths safely
        L = min(len(attn_norms), len(mlp_norms))
        return {
            "attn": attn_norms[:L],
            "mlp": mlp_norms[:L],
        }
    except:
        return None


# ---------------- ACTIVATION PATCHING (SAFE VERSION) ----------------

def activation_patch(tokens, model_name, layer, pos, from_pos, scale=1.0):
    """
    Safe activation patching (never crashes, only works for GPT-2 style).
    """
    try:
        model, tokenizer = load_model(model_name)
    except:
        return {"error": "Model load error."}

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        return {"error": "Model not compatible with patching."}

    text = " ".join(tokens)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)

    blocks = model.transformer.h
    wte = model.transformer.wte
    ln_f = model.transformer.ln_f if hasattr(model.transformer, "ln_f") else None
    lm_head = model.lm_head

    with torch.no_grad():
        x = wte(inputs["input_ids"]).to(DEVICE)
        hidden_layers = [x.clone().cpu().numpy()[0]]
        for b in blocks:
            ln1 = b.ln_1(x)
            a = b.attn(ln1)[0]
            x = x + a
            ln2 = b.ln_2(x)
            m = b.mlp(ln2)
            x = x + m
            hidden_layers.append(x.clone().cpu().numpy()[0])

    if layer >= len(hidden_layers):
        return {"error": "Layer out of range."}

    seq_len = hidden_layers[layer].shape[0]
    if pos >= seq_len or from_pos >= seq_len:
        return {"error": "Position out of range."}

    patch_vec = torch.tensor(hidden_layers[layer][from_pos], dtype=torch.float32).to(DEVICE) * float(scale)

    # re-run with patch
    with torch.no_grad():
        x = wte(inputs["input_ids"]).to(DEVICE)
        for i, b in enumerate(blocks):
            ln1 = b.ln_1(x)
            a = b.attn(ln1)[0]
            x = x + a
            ln2 = b.ln_2(x)
            m = b.mlp(ln2)
            x = x + m
            if i == layer:
                x[0, pos, :] = patch_vec

        final = ln_f(x) if ln_f else x
        logits = lm_head(final)[0, -1, :].cpu().numpy()
        probs = softmax(logits)
        idx = np.argsort(probs)[-20:][::-1]
        tt = [tokenizer.decode([int(i)]) for i in idx]
        ss = probs[idx].tolist()
        return {"tokens": tt, "scores": ss}


# ---------------- GRADIO UI ----------------

with gr.Blocks(title="LLM Visualizer — Full", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧠 Full LLM Visualizer (Advanced)")
    gr.Markdown("Fully stable build with attention, PCA, neuron explorer, residuals, activation-patching")

    # Panel 1
    with gr.Row():
        with gr.Column(scale=3):
            model_name = gr.Textbox(label="Model", value=DEFAULT_MODEL)
            input_text = gr.Textbox(label="Input", value="Hello world", lines=3)
            simple = gr.Checkbox(label="Explain simply", value=True)
            run_btn = gr.Button("Run", variant="primary")

            gr.Markdown("Presets:")
            with gr.Row():
                gr.Button("Greeting").click(lambda: "Hello! How are you?", None, input_text)
                gr.Button("Story").click(lambda: "Once upon a time there was a robot.", None, input_text)
                gr.Button("Question").click(lambda: "Why is the sky blue?", None, input_text)

        with gr.Column(scale=2):
            token_display = gr.Markdown()
            explanation_md = gr.Markdown()
            model_info = gr.Markdown()

    # Panel 2
    with gr.Row():
        with gr.Column():
            layer_slider = gr.Slider(0, 0, value=0, step=1, label="Layer")
            head_slider = gr.Slider(0, 0, value=0, step=1, label="Head")
            token_step = gr.Slider(0, 0, value=0, step=1, label="Token index")
            attn_plot = gr.Plot(elem_id="attn_plot")
            attn_fs = gr.HTML(fullscreen_button_html("attn_plot"))
        with gr.Column():
            pca_plot = gr.Plot(elem_id="pca_plot")
            pca_fs = gr.HTML(fullscreen_button_html("pca_plot"))
            step_attn_plot = gr.Plot(elem_id="step_attn_plot")
            step_fs = gr.HTML(fullscreen_button_html("step_attn_plot"))
            probs_plot = gr.Plot(elem_id="probs_plot")
            probs_fs = gr.HTML(fullscreen_button_html("probs_plot"))

    # Panel 3 — Residuals
    residual_plot = gr.Plot(elem_id="residual_plot")
    residual_fs = gr.HTML(fullscreen_button_html("residual_plot"))

    # Panel 4 — Neuron explorer
    with gr.Row():
        neuron_find_btn = gr.Button("Find neurons")
        neuron_idx = gr.Number(label="Neuron index", value=0)
        neuron_table = gr.Dataframe(headers=["token", "activation"], interactive=False)

    # Panel 5 — Activation Patching
    with gr.Row():
        patch_layer = gr.Slider(0, 0, value=0, step=1, label="Patch layer")
        patch_pos = gr.Slider(0, 0, value=0, step=1, label="Target token position")
        patch_from = gr.Slider(0, 0, value=0, step=1, label="Copy from position")
        patch_scale = gr.Number(label="Scale", value=1.0)
        patch_btn = gr.Button("Run patch")
        patch_output = gr.Plot(elem_id="patch_plot")
        patch_fs = gr.HTML(fullscreen_button_html("patch_plot"))

    state = gr.State()

    # ---- RUN ANALYSIS ----
    def run_app(text, model, simp):
        res = analyze_text(text, model, simp)

        if "error" in res:
            return {
                token_display: gr.update(value=""),
                explanation_md: gr.update(value=res["error"]),
                model_info: gr.update(value=f"Model: {model}"),
                attn_plot: gr.update(value=None),
                pca_plot: gr.update(value=None),
                probs_plot: gr.update(value=None),
                layer_slider: gr.update(maximum=0, value=0),
                head_slider: gr.update(maximum=0, value=0),
                token_step: gr.update(maximum=0, value=0),
                residual_plot: gr.update(value=None),
                neuron_table: gr.update(value=[]),
                patch_layer: gr.update(maximum=0),
                patch_pos: gr.update(maximum=0),
                patch_from: gr.update(maximum=0),
                state: res,
                step_attn_plot: gr.update(value=None),
                patch_output: gr.update(value=None),
            }

        tokens = res["tokens"]
        L = len(res["attentions"])
        H = res["attentions"][0].shape[0]
        T = len(tokens) - 1

        residual_fig = None
        if res["residuals"]:
            attn_vals = res["residuals"]["attn"]
            ml_vals = res["residuals"]["mlp"]
            Lmin = min(len(attn_vals), len(ml_vals))
            df = pd.DataFrame({
                "layer": list(range(Lmin)),
                "attention": attn_vals[:Lmin],
                "mlp": ml_vals[:Lmin]
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["layer"], y=df["attention"], name="Attention norm"))
            fig.add_trace(go.Bar(x=df["layer"], y=df["mlp"], name="MLP norm"))
            fig.update_layout(barmode="group", height=360)
            residual_fig = fig

        return {
            token_display: gr.update(value=f"**Tokens:** {res['token_display']}"),
            explanation_md: gr.update(value=res["explanation"]),
            model_info: gr.update(value=f"Model: {model} • layers: {L} • heads: {H} • tokens: {len(tokens)}"),
            attn_plot: gr.update(value=res["fig_attn"] if res.get("fig_attn") else None),
            pca_plot: gr.update(value=res["fig_pca"] if res.get("fig_pca") else None),
            probs_plot: gr.update(value=fig_probs(res["top_tokens"], res["top_scores"])),
            layer_slider: gr.update(maximum=L-1, value=res["default_layer"]),
            head_slider: gr.update(maximum=H-1, value=res["default_head"]),
            token_step: gr.update(maximum=T, value=0),
            residual_plot: gr.update(value=residual_fig),
            neuron_table: gr.update(value=[[t, round(v,4)] for t,v in res["neuron_info"][0]["top_tokens"]] if res["neuron_info"] else []),
            patch_layer: gr.update(maximum=L-1, value=0),
            patch_pos: gr.update(maximum=T, value=0),
            patch_from: gr.update(maximum=T, value=0),
            state: res,
            step_attn_plot: gr.update(value=None),
            patch_output: gr.update(value=None),
        }


    run_btn.click(
        run_app,
        inputs=[input_text, model_name, simple],
        outputs=[
            token_display, explanation_md, model_info,
            attn_plot, pca_plot, probs_plot,
            layer_slider, head_slider, token_step,
            residual_plot, neuron_table,
            patch_layer, patch_pos, patch_from,
            state, step_attn_plot, patch_output
        ]
    )

    # ---- SLIDER UPDATES ----
    def update_view(res, layer, head, tok):
        if not res or "error" in res:
            return {
                attn_plot: gr.update(value=None),
                pca_plot: gr.update(value=None),
                step_attn_plot: gr.update(value=None),
            }

        tokens = res["tokens"]
        layer = min(max(0, layer), len(res["attentions"]) - 1)
        head = min(max(0, head), res["attentions"][0].shape[0] - 1)
        tok = min(max(0, tok), len(tokens) - 1)

        att = fig_attention(res["attentions"][layer][head], tokens, f"Layer {layer} Head {head}")
        pts = res["pca"][layer]
        pca_fig = fig_pca(pts, tokens, highlight=tok, title=f"PCA Layer {layer}")

        row = res["attentions"][layer][head][tok]
        step_fig = go.Figure([go.Bar(x=tokens, y=row)])
        step_fig.update_layout(title=f"Token {tok} attends to")

        return {
            attn_plot: gr.update(value=att),
            pca_plot: gr.update(value=pca_fig),
            step_attn_plot: gr.update(value=step_fig)
        }

    layer_slider.change(update_view, [state, layer_slider, head_slider, token_step],
                        [attn_plot, pca_plot, step_attn_plot])
    head_slider.change(update_view, [state, layer_slider, head_slider, token_step],
                        [attn_plot, pca_plot, step_attn_plot])
    token_step.change(update_view, [state, layer_slider, head_slider, token_step],
                      [attn_plot, pca_plot, step_attn_plot])

    # ---- NEURON EXPLORER ----
    def neuron_auto(res):
        if not res or "neuron_info" not in res:
            return gr.update(value=[])
        rows = []
        for item in res["neuron_info"]:
            for t, v in item["top_tokens"]:
                rows.append([t, round(v,4)])
        df = pd.DataFrame(rows, columns=["token","activation"]).drop_duplicates().head(24)
        return gr.update(value=df.values.tolist())

    neuron_find_btn.click(neuron_auto, [state], [neuron_table])

    def neuron_manual(res, idx):
        if not res or "hidden" not in res:
            return gr.update(value=[])
        try:
            idx = int(idx)
        except:
            return gr.update(value=[])
        last = res["hidden"][-1]
        if idx >= last.shape[1]:
            return gr.update(value=[])
        vals = last[:, idx]
        tokens = res["tokens"]
        pairs = sorted([(tokens[i], float(vals[i])) for i in range(len(tokens))],
                       key=lambda x: -abs(x[1]))[:12]
        return gr.update(value=[[t, round(v,4)] for t,v in pairs])

    neuron_idx.change(neuron_manual, [state, neuron_idx], [neuron_table])

    # ---- ACTIVATION PATCHING ----
    def patch_run(res, L, P, FP, S, model):
        if not res or "tokens" not in res:
            return gr.update(value=None)
        out = activation_patch(res["tokens"], model, int(L), int(P), int(FP), float(S))
        if "error" in out:
            return gr.update(value=None)
        fig = fig_probs(out["tokens"], out["scores"])
        return gr.update(value=fig)

    patch_btn.click(patch_run,
                    [state, patch_layer, patch_pos, patch_from, patch_scale, model_name],
                    [patch_output])

demo.launch()
