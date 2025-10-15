from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ─────────────────────────── Paths (robust to any CWD) ───────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "processed_dir": REPO_ROOT / "data" / "processed",
    "artifacts_dir": REPO_ROOT / "models" / "artifacts",
    "reports_dir":   REPO_ROOT / "models" / "reports",   # per-algo
    "root_reports":  REPO_ROOT / "reports",              # winner mirrored here
    "target": "target",
}

def load_config_paths() -> Dict[str, Path | str]:
    cfg = DEFAULTS.copy()
    cfg_file = REPO_ROOT / "configs" / "default.yaml"
    if cfg_file.exists():
        try:
            import yaml
            y = yaml.safe_load(cfg_file.read_text())
            p = y.get("paths", {})
            d = y.get("dataset", {})
            cfg["processed_dir"] = REPO_ROOT / p.get("processed_dir", DEFAULTS["processed_dir"])
            cfg["artifacts_dir"] = REPO_ROOT / p.get("artifacts_dir", DEFAULTS["artifacts_dir"])
            cfg["reports_dir"]   = REPO_ROOT / p.get("reports_dir",   DEFAULTS["reports_dir"])
            cfg["target"]        = d.get("target", DEFAULTS["target"])
        except Exception:
            pass
    return cfg

CFG = load_config_paths()
PROCESSED_DIR = Path(CFG["processed_dir"])
ARTIFACTS_DIR = Path(CFG["artifacts_dir"])
REPORTS_DIR   = Path(CFG["reports_dir"])
ROOT_REPORTS  = Path(CFG.get("root_reports", DEFAULTS["root_reports"]))
TARGET_COL    = CFG["target"]

# Optional profile link (override with env var DASH_GITHUB_URL)
GITHUB_URL = os.getenv("DASH_GITHUB_URL", "https://github.com/p-akash-iisc")

# ─────────────────────────── Theme (light/dark) ───────────────────────────
def theme_css(mode: str = "light") -> str:
    palette = {
        "light": {
            "bg": "#ffffff", "panel": "#F6F8FB", "card": "#ffffff",
            "text": "#111827", "muted": "#6B7280", "border": "#E5E7EB",
            "primary": "#2563EB", "accent": "#22C55E", "tab": "#E0E7FF"
        },
        "dark": {
            "bg": "#0B1220", "panel": "#0F172A", "card": "#0F172A",
            "text": "#E5E7EB", "muted": "#9CA3AF", "border": "#1F2937",
            "primary": "#7C3AED", "accent": "#10B981", "tab": "#1F2937"
        },
    }[("dark" if mode.lower().startswith("d") else "light")]

    return f"""
    <style>
      :root {{
        --bg:{palette['bg']}; --panel:{palette['panel']}; --card:{palette['card']};
        --text:{palette['text']}; --muted:{palette['muted']}; --border:{palette['border']};
        --primary:{palette['primary']}; --accent:{palette['accent']}; --tab:{palette['tab']};
      }}

      /* base */
      .stApp, .main {{ background: var(--bg) !important; color: var(--text); }}
      .block {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }}
      .kpi   {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }}
      .big-metric {{ font-size: 1.6rem; font-weight: 700; margin-bottom: .2rem; }}
      .metric-sub {{ color: var(--muted); font-size: .9rem; margin-bottom: .6rem; }}
      img {{ border-radius: 10px; }}
      h1 span.title-gradient {{
        background: linear-gradient(90deg, var(--primary), var(--accent));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }}

      /* tabs: keep ALL labels visible; highlight only selected */
      div[role="tablist"] [role="tab"] {{ opacity: 1 !important; }}
      div[role="tablist"] [role="tab"] * {{ color: var(--text) !important; opacity: 1 !important; }}
      div[role="tablist"] [role="tab"][aria-selected="true"] {{
        background: var(--tab); border-radius: 10px; border: 1px solid var(--border);
      }}
      div[role="tablist"] [role="tab"][aria-selected="true"] * {{
        color: var(--primary) !important; font-weight: 700;
      }}

      /* radios & form labels: make text fully visible in both themes */
      div[role="radiogroup"] * {{ color: var(--text) !important; opacity: 1 !important; }}
      label {{ color: var(--text) !important; opacity: 1 !important; }}

      /* buttons & sliders */
      .stDownloadButton button, .stButton button {{
        background: var(--primary); color:#fff; border-radius: 10px; border: none;
      }}
      div.stSlider > div > div > div[role='slider'] {{ background: var(--primary); }}
    </style>
    """



# ─────────────────────────── IO helpers ───────────────────────────

def list_algorithms() -> List[str]:
    return sorted([p.stem for p in ARTIFACTS_DIR.glob("*.joblib") if p.stem != "best_model"])

def read_meta() -> Dict[str, Any]:
    p = ARTIFACTS_DIR / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}

def load_model(algo: str):
    return load(ARTIFACTS_DIR / f"{algo}.joblib")

def proba_like(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):  return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X).astype(float);  return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(float)

def load_metrics(algo: str) -> Dict[str, Any]:
    p = REPORTS_DIR / algo / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

def read_png(path: Path):
    return str(path) if path.exists() else None

def feature_columns() -> List[str]:
    for fn in ["train.csv", "val.csv", "test.csv"]:
        p = PROCESSED_DIR / fn
        if p.exists():
            df = pd.read_csv(p);  return [c for c in df.columns if c != TARGET_COL]
    return []

def load_dataset(split: str) -> Tuple[pd.DataFrame, np.ndarray | None]:
    file = "val.csv" if "val" in split.lower() else "test.csv"
    p = PROCESSED_DIR / file
    if not p.exists(): return pd.DataFrame(), None
    df = pd.read_csv(p)
    y = df[TARGET_COL].astype(int).values if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
    return X, y

def align_uploaded_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy();  df = df[[c for c in df.columns if c in cols]]
    for c in cols:
        if c not in df.columns: df[c] = 0
    return df[cols].fillna(0)

def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""

def load_summary_df() -> pd.DataFrame:
    """Read summary produced by evaluate.py from models/reports/ (JSON only)."""
    p_json = REPORTS_DIR / "summary.json"   # models/reports/summary.json
    if not p_json.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(p_json)  # JSON array written with orient="records"
    except Exception:
        return pd.DataFrame()



# ─────────────────────────── UI ───────────────────────────

st.set_page_config(page_title="Chronic Risk XAI Dashboard", layout="wide", page_icon="🩺")

# theme toggle
if "theme" not in st.session_state: st.session_state["theme"] = "light"
st.sidebar.markdown("### Appearance")
_dark = st.sidebar.toggle("🌗 Dark mode", value=(st.session_state["theme"] == "dark"))
st.session_state["theme"] = "dark" if _dark else "light"
st.markdown(theme_css(st.session_state["theme"]), unsafe_allow_html=True)

st.markdown('<h1><span class="title-gradient">Chronic Risk — Model Monitor & Explainability</span></h1>', unsafe_allow_html=True)

meta  = read_meta()
algos = list_algorithms()
if not algos:
    st.error("No models found under models/artifacts. Train & evaluate first.")
    st.stop()

default_algo = meta.get("best_model", algos[0]) if meta else algos[0]
if default_algo not in algos: default_algo = algos[0]

# Sidebar controls
st.sidebar.header("⚙️ Controls")
selected_algo = st.sidebar.selectbox("Model", algos, index=algos.index(default_algo))
data_split    = st.sidebar.selectbox("Sample split", ["Test (test.csv)", "Validation (val.csv)"], index=0)
sample_mode   = st.sidebar.selectbox("Sample mode", ["Top high risk", "Top low risk", "Random"], index=0)
top_k         = st.sidebar.slider("Rows to show", 1, 100, 5)


st.sidebar.markdown("---")
st.sidebar.markdown(
    f"Made with ❤️"
)

st.sidebar.markdown(
    f"© [ P_Akash_Singh]({GITHUB_URL})",
    unsafe_allow_html=False
)





# load selection
feat_cols = feature_columns()
model     = load_model(selected_algo)
metrics   = load_metrics(selected_algo)

# ── Metrics + Winner ──
left, right = st.columns([2,1])

with left:
    st.subheader("Performance at a glance")
    st.caption("Shows the selected model’s **test-set** performance at the saved operating threshold. "
               "**PR/ROC AUC** reflect ranking quality; **Brier** reflects calibration (lower=better). "
               "**F1** balances precision & recall at that threshold.")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        v = metrics.get("pr_auc") or 0.0
        st.markdown(f'<div class="big-metric">{v:.3f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">PR AUC {fmt_ci(metrics.get("pr_auc_ci"))}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        v = metrics.get("roc_auc") or 0.0
        st.markdown(f'<div class="big-metric">{v:.3f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">ROC AUC {fmt_ci(metrics.get("roc_auc_ci"))}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-metric">{(metrics.get("brier") or 0):.3f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Brier score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-metric">{(metrics.get("f1") or 0):.3f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">F1 @ threshold</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.subheader("Winner")
    st.caption("The best model selected during evaluation using the **primary metric**. "
               "Switch models from the sidebar to explore alternatives.")
    if meta:
        st.write(f"**Best model:** `{meta.get('best_model','?')}`")
        st.write(f"**Primary metric:** `{meta.get('primary_metric','?')}`")
        st.write(f"**Best score:** `{meta.get('best_score','?')}`")
        st.write(f"**Threshold:** `{meta.get('threshold','?')}`")
    else:
        st.info("Run evaluation to set the winner.")

# ── Curves & Diagnostics ──
st.markdown("### 📈 Curves & Diagnostics")
st.caption("Explore how the selected model behaves on the **held-out test set**. "
           "Use the tabs for **PR/ROC**, **Confusion**, **Calibration**, **Gains & Lift**, and **Explainability (SHAP)**.")

curves, confusion_tab, calibration_tab, gains_tab, explain_tab = st.tabs(
    ["PR/ROC", "Confusion", "Calibration", "Gains & Lift", "Explainability (SHAP)"]
)


# with curves:
#     st.markdown(
#         "**What these show**  \n"
#         "- **PR curve**: precision vs recall across thresholds; area = **Average Precision (PR AUC)** (best for imbalance).  \n"
#         "- **ROC curve**: true-positive vs false-positive rate; area = **ROC AUC**. Top-left ≈ better."
#     )
#     c1, c2, _ = st.columns(3)  # keep a 1×3 grid; last column is just spacing
#     pr_path  = REPORTS_DIR / selected_algo / "pr_curve.png"
#     roc_path = REPORTS_DIR / selected_algo / "roc_curve.png"
#     c1.image(read_png(pr_path),  use_container_width=True)
#     c1.caption("Precision vs Recall across thresholds; area is PR AUC.")
#     c2.image(read_png(roc_path), use_container_width=True)
#     c2.caption("TPR vs FPR across thresholds; area is ROC AUC.")

# with confusion_tab:
#     st.markdown(
#         "Counts at the saved operating **threshold**.  \n"
#         "**TP/TN** are correct; **FP/FN** are errors—tune threshold during evaluation to trade recall vs false alarms."
#     )
#     img = read_png(REPORTS_DIR / selected_algo / "confusion_matrix.png")
#     st.image(img, use_container_width=True)
#     st.caption("Confusion matrix computed on the held-out test set at the chosen threshold.")

# with calibration_tab:
#     st.markdown(
#         "Compares predicted probabilities to observed outcomes.  \n"
#         "A perfectly calibrated model lies on the diagonal; deviations indicate over/under-confidence."
#     )
#     img = read_png(REPORTS_DIR / selected_algo / "calibration.png")
#     st.image(img, use_container_width=True)
#     st.caption("Closer to the diagonal ⇒ better calibration. Related scalar: Brier score (lower is better).")


# with gains_tab:
#     st.markdown(
#         "**Gains**: fraction of positives captured as you target top-scored cases.  \n"
#         "**Lift**: improvement vs random selection (1.0 = random)."
#     )
#     g1, g2, _ = st.columns(3)  # 1×3 row; third column is spacing
#     g1.image(read_png(REPORTS_DIR / selected_algo / "eval_gains.png"),
#              use_container_width=True)
#     g1.caption("Cumulative Gains — how quickly positives are captured when ranking by score.")
#     g2.image(read_png(REPORTS_DIR / selected_algo / "eval_lift.png"),
#              use_container_width=True)
#     g2.caption("Lift vs random. Lift > 1 means better targeting than random.")


# with explain_tab:
#     st.markdown(
#         "**SHAP** attributes feature contributions.  \n"
#         "Global: **bar** and **beeswarm** show which features drive predictions and in what direction.  \n"
#         "Local: **waterfall** explains one high-risk case."
#     )
#     c1, c2, c3 = st.columns(3)  # 1×3 layout
#     c1.image(read_png(REPORTS_DIR / selected_algo / "shap_importance_bar.png"),
#              use_container_width=True)
#     c1.caption("Global importance — mean |SHAP| per feature.")
#     c2.image(read_png(REPORTS_DIR / selected_algo / "shap_summary_beeswarm.png"),
#              use_container_width=True)
#     c2.caption("Beeswarm — distribution and direction of feature effects.")
#     c3.image(read_png(REPORTS_DIR / selected_algo / "shap_local_waterfall_top1.png"),
#              use_container_width=True)
#     c3.caption("Local waterfall — how features push the top-risk prediction.")

# ---- put this once near your other helpers ----
IMG_WIDTH = 1440  # adjust to taste (same everywhere)

def show_img(col, path, caption=""):
    img = read_png(path)
    col.image(img, width=IMG_WIDTH)
    if caption:
        col.caption(caption)
# -----------------------------------------------


with curves:
    st.markdown(
        "**What these show**  \n"
        "- **PR curve**: precision vs recall across thresholds; area = **Average Precision (PR AUC)** (best for imbalance).  \n"
        "- **ROC curve**: true-positive vs false-positive rate; area = **ROC AUC**. Top-left ≈ better."
    )
    c1, c2, c3 = st.columns(3)
    pr_path  = REPORTS_DIR / selected_algo / "pr_curve.png"
    roc_path = REPORTS_DIR / selected_algo / "roc_curve.png"
    show_img(c1, pr_path,  "Precision vs Recall across thresholds; area is PR AUC.")
    show_img(c2, roc_path, "TPR vs FPR across thresholds; area is ROC AUC.")
    c3.empty()  # spacer


with confusion_tab:
    st.markdown(
        "Counts at the saved operating **threshold**.  \n"
        "**TP/TN** are correct; **FP/FN** are errors—tune threshold during evaluation to trade recall vs false alarms."
    )
    c1, c2, c3 = st.columns(3)
    show_img(
        c2,
        REPORTS_DIR / selected_algo / "confusion_matrix.png",
        "Confusion matrix computed on the held-out test set at the chosen threshold."
    )
    c1.empty(); c3.empty()


with calibration_tab:
    st.markdown(
        "Compares predicted probabilities to observed outcomes.  \n"
        "A perfectly calibrated model lies on the diagonal; deviations indicate over/under-confidence."
    )
    c1, c2, c3 = st.columns(3)
    show_img(
        c2,
        REPORTS_DIR / selected_algo / "calibration.png",
        "Closer to the diagonal ⇒ better calibration. Related scalar: Brier score (lower is better)."
    )
    c1.empty(); c3.empty()


with gains_tab:
    st.markdown(
        "**Gains**: fraction of positives captured as you target top-scored cases.  \n"
        "**Lift**: improvement vs random selection (1.0 = random)."
    )
    g1, g2, g3 = st.columns(3)
    show_img(g1, REPORTS_DIR / selected_algo / "eval_gains.png",
             "Cumulative Gains — how quickly positives are captured when ranking by score.")
    show_img(g2, REPORTS_DIR / selected_algo / "eval_lift.png",
             "Lift vs random. Lift > 1 means better targeting than random.")
    g3.empty()


with explain_tab:
    st.markdown(
        "**SHAP** attributes feature contributions.  \n"
        "Global: **bar** and **beeswarm** show which features drive predictions and in what direction.  \n"
        "Local: **waterfall** explains one high-risk case."
    )
    e1, e2, e3 = st.columns(3)
    show_img(e1, REPORTS_DIR / selected_algo / "shap_importance_bar.png",
             "Global importance — mean |SHAP| per feature.")
    show_img(e2, REPORTS_DIR / selected_algo / "shap_summary_beeswarm.png",
             "Beeswarm — distribution and direction of feature effects.")
    show_img(e3, REPORTS_DIR / selected_algo / "shap_local_waterfall_top1.png",
             "Local waterfall — how features push the top-risk prediction.")





# ── Model comparison (summary) ──
st.markdown("### 📊 Model comparison (all algorithms)")
st.caption("Aggregates metrics across algorithms: **PR/ROC AUC (95% CI)**, **Brier**, **F1**, "
           "**precision/recall**, **threshold**, and **n_test**. Data from `reports/summary.json`.")
sum_df = load_summary_df()
if sum_df.empty:
    st.info("No summary found in `reports/summary.json`. Run the evaluation step first.")
else:
    if {"pr_auc_lo","pr_auc_hi","roc_auc_lo","roc_auc_hi"}.issubset(sum_df.columns):
        sum_df["PR AUC (95% CI)"]  = sum_df.apply(lambda r: f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f}, {r['pr_auc_hi']:.3f}]", axis=1)
        sum_df["ROC AUC (95% CI)"] = sum_df.apply(lambda r: f"{r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}]", axis=1)
    show_cols = [c for c in ["algo","PR AUC (95% CI)","ROC AUC (95% CI)","brier","f1","precision","recall","threshold","n_test"] if c in sum_df.columns]
    if not show_cols: show_cols = list(sum_df.columns)
    st.dataframe(sum_df[show_cols], use_container_width=True)
    st.download_button("Download summary CSV", data=sum_df.to_csv(index=False).encode("utf-8"),
                       file_name="summary.csv")

# ── Sample Predictions ──
st.markdown("### 🔎 Sample Predictions")
st.caption("Peek at individual rows from **validation/test** with the model’s **risk score** (`pred_proba`). "
           "Choose **Top high risk**, **Top low risk**, or **Random**; download the preview as CSV.")
X_s, y_s = load_dataset(data_split)
if X_s is None or X_s.empty:
    st.info("No processed data found under data/processed/.")
else:
    probs = proba_like(model, X_s)
    view = X_s.copy();  view["pred_proba"] = probs
    if y_s is not None: view[TARGET_COL] = y_s

    mode = st.radio("Pick sample:", ["Top high risk", "Top low risk", "Random"], horizontal=True, index=0)
    k = st.slider("Rows to display", 5, min(200, len(view)), min(top_k, len(view)))
    if mode == "Top high risk": view = view.sort_values("pred_proba", ascending=False).head(k)
    elif mode == "Top low risk": view = view.sort_values("pred_proba", ascending=True).head(k)
    else: view = view.sample(n=min(k, len(view)), random_state=42)
    st.dataframe(view.reset_index(drop=True), use_container_width=True)
    st.download_button("Download these rows", data=view.to_csv(index=False).encode("utf-8"),
                       file_name=f"{selected_algo}_sample.csv")

# ── Upload & Predict ──
st.markdown("### ⬆️ Upload CSV and Predict Risk")
st.caption("Upload a CSV with the **same feature columns** as training. We align columns and fill missing values with 0; "
           "your file is not stored. Download predictions as CSV.")
upl = st.file_uploader("Upload CSV with feature columns", type=["csv"])
if upl is not None:
    try:
        df = pd.read_csv(upl)
        cols = feature_columns()
        df = align_uploaded_df(df, cols)
        probs = proba_like(model, df)
        out = df.copy()
        out["pred_proba"] = probs
        thr = float(metrics.get("threshold", 0.5))
        out["pred_label"] = (out["pred_proba"] >= thr).astype(int)
        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(100), use_container_width=True)
        st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"),
                           file_name=f"{selected_algo}_predictions.csv")
    except Exception as e:
        st.error(f"Scoring failed: {e}")



