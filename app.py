# -*- coding: utf-8 -*-
"""
ISR Risk Predictor — open-source Streamlit demo
(Companion to the paper: multimodal + full-combination ISR prediction model)

Loads model/pipeline.joblib (exported by export_pipeline.py) and provides:
  - 7-feature input -> ISR probability + ISR/non-ISR verdict (Youden threshold)
  - per-instance SHAP explanation (force plot, native style, matches the paper)
  - data dictionary + research-use disclaimer

Deploy: push this folder to GitHub and deploy via Streamlit Community Cloud
(main file: app.py). Free, no HF PRO required.
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE_PATH = os.path.join(HERE, "model", "pipeline.joblib")

# ---------------------------------------------------------------------------
# Custom UI styling (light theme, clean clinical look)
# ---------------------------------------------------------------------------
CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #1f6feb 0%, #0b3d91 100%);
    color: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 6px 18px rgba(11, 61, 145, .18);
    margin-bottom: .4rem;
}
.hero-title { font-size: 1.5rem; font-weight: 800; letter-spacing: .5px; }
.hero-sub  { font-size: .9rem; opacity: .92; margin-top: .45rem; }
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #0b3d91;
    margin: 1.3rem 0 .7rem; padding-left: .55rem;
    border-left: 4px solid #1f6feb;
}
.verdict-high {
    display: inline-block; background: #fdecea; color: #c0392b;
    border: 1px solid #f5c6c0; border-radius: 999px;
    padding: .4rem 1.1rem; font-weight: 700; font-size: 1.05rem;
}
.verdict-low {
    display: inline-block; background: #eafaf1; color: #1e8449;
    border: 1px solid #bfe6cd; border-radius: 999px;
    padding: .4rem 1.1rem; font-weight: 700; font-size: 1.05rem;
}
.pbar-wrap {
    position: relative; height: 28px; background: #eceff3;
    border-radius: 999px; overflow: hidden;
}
.pbar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #f1c40f 0%, #e67e22 55%, #e74c3c 100%);
    transition: width .45s ease;
}
.pbar-thresh { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #2c3e50; }
.hint { font-size: .82rem; color: #6b7280; }
</style>
"""


def load_pipeline(path=PIPE_PATH):
    return joblib.load(path)


def predict(pipeline, values):
    """values: dict feature_name -> float (raw, in original units)."""
    feats = pipeline["features"]
    X = np.array([[float(values[f]) for f in feats]], dtype=float)
    Xs = pipeline["scaler"].transform(X)
    prob = float(pipeline["model"].predict_proba(Xs)[0, 1])
    isr = prob >= pipeline["threshold"]
    return dict(X=X, Xs=Xs, prob=prob, isr=bool(isr),
                threshold=pipeline["threshold"], expected_value=pipeline["expected_value"])


def shap_force_figure(pipeline, res):
    """Render a per-instance SHAP force plot (matplotlib) consistent with the paper's Fig.12."""
    model = pipeline["model"]
    feats = pipeline["features"]
    disp = pipeline["display"]
    ev = pipeline["expected_value"]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(res["Xs"])  # list[2] for binary
    if isinstance(sv, list):
        sv_pos = np.asarray(sv[1])
    else:
        sv_pos = np.asarray(sv)[:, :, 1]
    # feature values (rounded for cleaner labels)
    fvals = np.round(res["X"].ravel(), 3)
    fig = shap.plots.force(
        ev, sv_pos[0], features=fvals,
        feature_names=[disp[f] for f in feats],
        matplotlib=True, show=False,
    )
    return fig


def shap_table(pipeline, res):
    model = pipeline["model"]
    feats = pipeline["features"]
    disp = pipeline["display"]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(res["Xs"])
    if isinstance(sv, list):
        sv_pos = np.asarray(sv[1])
    else:
        sv_pos = np.asarray(sv)[:, :, 1]
    rows = [dict(Feature=disp[f], SHAP=float(sv_pos[0, j]),
                 Value=float(res["X"][0, j])) for j, f in enumerate(feats)]
    return pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["SHAP"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def run_app():
    import streamlit as st
    st.set_page_config(page_title="ISR Risk Predictor", page_icon="🩺", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    pipeline = load_pipeline()
    feats = pipeline["features"]
    disp = pipeline["display"]
    units = pipeline["units"]
    descr = pipeline["descr"]
    stats = pipeline["stats"]
    medians = {f: float(stats[f]["median"]) for f in feats}

    # ---- English UI labels (override the model's Chinese metadata in-memory;
    #      the joblib file is NOT modified, model params stay identical) ----
    EN_DESCR = {
        "Mean_nwi_Total": "Mean normalized wall index (Mean NWI)",
        "Max_nwi_Total": "Max normalized wall index (Max NWI)",
        "Residual_stenosis_rate": "Residual stenosis rate",
        "Mean_lumen_area(mm²)_Total": "Mean lumen area",
        "wavelet-HLL_glcm_Correlation": "Radiomic texture: wavelet-HLL GLCM correlation",
        "original_shape_Flatness": "Radiomic shape: flatness",
        "TC": "Total cholesterol",
    }
    EN_UNITS = {
        "Mean_nwi_Total": "dimensionless",
        "Max_nwi_Total": "dimensionless",
        "Residual_stenosis_rate": "proportion (0–1)",
        "Mean_lumen_area(mm²)_Total": "mm²",
        "wavelet-HLL_glcm_Correlation": "dimensionless",
        "original_shape_Flatness": "dimensionless",
        "TC": "mmol/L",
    }
    descr = {f: EN_DESCR.get(f, pipeline["descr"][f]) for f in feats}
    units = {f: EN_UNITS.get(f, pipeline["units"][f]) for f in feats}

    # ---- session state ----
    if "result" not in st.session_state:
        st.session_state.result = predict(pipeline, medians)
        st.session_state.shap_fig = shap_force_figure(pipeline, st.session_state.result)
        st.session_state.shap_df = shap_table(pipeline, st.session_state.result)
    if "submitted" not in st.session_state:
        st.session_state.submitted = dict(medians)

    # ---- apply a pending RESET *before* widgets are instantiated ----
    # In Streamlit >= 1.37 a widget's session_state key cannot be modified after
    # the widget is created (neither in a callback nor in the body). So the reset
    # button only flips this flag; the actual reset happens here, at the very top
    # of the next script run, before the number_inputs exist.
    if st.session_state.get("_reset_pending"):
        for f in feats:
            st.session_state[f"in_{f}"] = medians[f]
        res = predict(pipeline, medians)
        st.session_state.result = res
        st.session_state.shap_fig = shap_force_figure(pipeline, res)
        st.session_state.shap_df = shap_table(pipeline, res)
        st.session_state.submitted = dict(medians)
        st.session_state._reset_pending = False

    # ---- hero header ----
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🩺 ISR Risk Predictor</div>'
        '<div class="hero-sub">Explainable prediction model for in-stent restenosis after '
        'intracranial stenting &middot; open-source online demo</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter the following **7 features** (identical to the final model in the paper) to obtain "
        "the **predicted probability** of in-stent restenosis (ISR) and a **per-instance SHAP "
        "explanation**. The model was trained on the internal derivation cohort (n=237) with a "
        "cross-validated AUC of **0.823**. After editing the inputs, click **🔄 Update prediction** "
        "to refresh."
    )

    # ---- ① inputs (inside a FORM so typed values are committed on submit) ----
    st.markdown('<div class="section-title">① Input features</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Type a value directly or use the +/- stepper; then click '
        ' <b>🔄 Update prediction</b> at the bottom of the form (all changes are committed at once). '
        'Each input is bounded by the <b>physically / clinically possible range</b> of the feature. '
        'Values outside the training-set range trigger an <b>extrapolation warning</b> in the results '
        'section &mdash; interpret such predictions with caution.</div>',
        unsafe_allow_html=True,
    )
    # initialize widget keys once (only first render)
    for f in feats:
        if f"in_{f}" not in st.session_state:
            st.session_state[f"in_{f}"] = medians[f]

    # Hard bounds = the physically / clinically *possible* range of each feature
    # (not the training-set min/max). Values can still be typed outside the
    # training range -> flagged as extrapolation in the results section.
    BOUNDS = {
        "Mean_nwi_Total": (0.0, 1.0),          # normalized wall index, dimensionless 0–1
        "Max_nwi_Total": (0.0, 1.0),           # normalized wall index, dimensionless 0–1
        "Residual_stenosis_rate": (0.0, 1.0),  # proportion (0–1)
        "Mean_lumen_area(mm²)_Total": (0.0, 30.0),  # mm², generous clinical headroom
        "wavelet-HLL_glcm_Correlation": (0.0, 1.0),  # correlation coefficient 0–1
        "original_shape_Flatness": (0.0, 1.0),       # flatness 0–1
        "TC": (0.0, 30.0),                     # total cholesterol, mmol/L (clinical extreme)
    }

    with st.form("input_form", clear_on_submit=False):
        cols = st.columns(2)
        for i, f in enumerate(feats):
            s = stats[f]
            with cols[i % 2]:
                step = 0.1 if f == "TC" else 0.001
                lo, hi = BOUNDS[f]
                st.number_input(
                    label=f"**{disp[f]}**  ({descr[f]})",
                    min_value=float(lo), max_value=float(hi),
                    step=step, format="%.4f",
                    key=f"in_{f}",
                    help=f"Unit: {units[f]}; allowed range {lo:.3f}–{hi:.3f}; "
                         f"training-set range {s['min']:.3f}–{s['max']:.3f}",
                )
        update_clicked = st.form_submit_button(
            "🔄 Update prediction", type="primary", width='stretch')

    # ---- reset button (OUTSIDE the form; plain buttons aren't allowed in forms) ----
    reset_clicked = st.button("↩ Reset to medians", width='stretch')

    if update_clicked:
        # Inside a form, ALL widget edits (typed or +/-) are committed to
        # session_state exactly when the submit button fires -> read them here.
        vals = {f: float(st.session_state[f"in_{f}"]) for f in feats}
        res = predict(pipeline, vals)
        st.session_state.result = res
        st.session_state.shap_fig = shap_force_figure(pipeline, res)
        st.session_state.shap_df = shap_table(pipeline, res)
        st.session_state.submitted = dict(vals)
        st.toast("✅ Prediction updated", icon="✅")

    if reset_clicked:
        # flip the flag; the actual reset runs at the top of the next script run
        st.session_state._reset_pending = True
        st.rerun()

    res = st.session_state.result

    # ---- extrapolation warning: any submitted value outside training range ----
    oor = [disp[f] for f in feats
           if st.session_state.submitted[f] < stats[f]["min"]
           or st.session_state.submitted[f] > stats[f]["max"]]
    if oor:
        st.warning(
            "⚠️ The current value of the following feature(s) is **outside the training-set "
            "range**; the prediction is an extrapolation and should be interpreted with caution: "
            + ", ".join(oor)
        )

    # ---- ② results ----
    st.markdown('<div class="section-title">② Prediction result</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        pct = res["prob"] * 100
        st.metric("Predicted ISR probability", f"{pct:.1f}%",
                  delta=f"Youden cutoff {res['threshold']:.3f}")
        if res["isr"]:
            st.markdown('<span class="verdict-high">⚠️ High ISR risk</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="verdict-low">✅ Low ISR risk</span>', unsafe_allow_html=True)
    with c2:
        # CSS probability bar with Youden-threshold marker (replaces the old matplotlib prob_bar)
        thr = res["threshold"] * 100
        bar_html = f"""
        <div style="margin-top:16px;">
          <div class="pbar-wrap">
            <div class="pbar-fill" style="width:{pct:.1f}%"></div>
            <div class="pbar-thresh" style="left:{thr:.1f}%"></div>
          </div>
          <div style="display:flex; justify-content:space-between;
                      font-size:.75rem; color:#6b7280; margin-top:5px;">
            <span>0%</span>
            <span style="color:#2c3e50; font-weight:600;">Youden cutoff {res['threshold']:.3f}</span>
            <span>100%</span>
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

    # ---- ③ SHAP ----
    st.markdown('<div class="section-title">③ Per-instance SHAP explanation</div>', unsafe_allow_html=True)
    st.markdown(
        f"The figure below shows how each feature pushes the predicted probability from the baseline "
        f"**E[f(x)] = {res['expected_value']:.3f}** to the model output **f(x) = {res['prob']:.3f}**. "
        f"Red segments indicate a high feature value, blue segments a low value; each segment is "
        f"colored by the feature's value."
    )
    st.pyplot(st.session_state.shap_fig, width='stretch')

    with st.expander("Feature contribution details"):
        st.dataframe(st.session_state.shap_df, width='stretch')

    # ---- data dictionary ----
    with st.expander("📋 Data dictionary (7 features)"):
        dd = pd.DataFrame([
            dict(Feature=disp[f], Description=descr[f], Unit=units[f],
                 Median=f"{stats[f]['median']:.3f}",
                 Range=f"{stats[f]['min']:.3f}–{stats[f]['max']:.3f}")
            for f in feats
        ])
        st.dataframe(dd, width='stretch')

    # ---- disclaimer ----
    st.divider()
    st.caption(
        "⚠️ **Research-use disclaimer**: This tool is a research demonstration trained on a "
        "single-center derivation cohort. Its predictions **must not replace clinical judgment "
        "or serve as a diagnostic basis**. When applied across centers, be aware of differences "
        "in feature distributions (e.g. systematic shifts in residual stenosis rate between "
        "centers) and interpret results cautiously together with local data."
    )


if __name__ == "__main__":
    run_app()
