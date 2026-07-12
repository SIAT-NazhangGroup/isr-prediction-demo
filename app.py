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
    st.set_page_config(page_title="ISR 风险预测器", page_icon="🩺", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    pipeline = load_pipeline()
    feats = pipeline["features"]
    disp = pipeline["display"]
    units = pipeline["units"]
    descr = pipeline["descr"]
    stats = pipeline["stats"]

    # ---- session state: keep last *submitted* prediction ----
    if "result" not in st.session_state:
        init = {f: float(stats[f]["median"]) for f in feats}
        st.session_state.result = predict(pipeline, init)
        st.session_state.shap_fig = shap_force_figure(pipeline, st.session_state.result)
        st.session_state.shap_df = shap_table(pipeline, st.session_state.result)
        st.session_state.submitted = dict(init)

    # ---- hero header ----
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🩺 颅内支架植入术后再狭窄（ISR）风险预测器</div>'
        '<div class="hero-sub">基于多模态特征与全组合优化框架的可解释预测模型 · 开源在线演示版</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "输入以下 **7 个特征**（与论文最终模型一致），获得该患者发生 ISR 的**预测概率**及"
        "**逐例 SHAP 解释**。模型在内部推导队列（n=237）上训练，交叉验证 AUC = **0.823**。"
        "修改输入后，请点击 **🔄 更新预测结果** 刷新。"
    )

    # ---- ① inputs ----
    st.markdown('<div class="section-title">① 输入特征</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, f in enumerate(feats):
        s = stats[f]
        with cols[i % 2]:
            step = 0.1 if f == "TC" else 0.001
            st.number_input(
                label=f"**{disp[f]}**  ({descr[f]})",
                min_value=float(s["min"]), max_value=float(s["max"]),
                value=float(s["median"]), step=step, format="%.4f",
                key=f"in_{f}",
                help=f"单位：{units[f]}；训练集范围 {s['min']:.3f}–{s['max']:.3f}",
            )

    # ---- action callbacks (set widget session_state BEFORE re-render) ----
    def do_update():
        values = {f: float(st.session_state[f"in_{f}"]) for f in feats}
        res = predict(pipeline, values)
        st.session_state.result = res
        st.session_state.shap_fig = shap_force_figure(pipeline, res)
        st.session_state.shap_df = shap_table(pipeline, res)
        st.session_state.submitted = dict(values)
        st.toast("✅ 预测结果已更新", icon="✅")

    def do_reset():
        vals = {f: float(stats[f]["median"]) for f in feats}
        for f in feats:
            st.session_state[f"in_{f}"] = vals[f]
        res = predict(pipeline, vals)
        st.session_state.result = res
        st.session_state.shap_fig = shap_force_figure(pipeline, res)
        st.session_state.shap_df = shap_table(pipeline, res)
        st.session_state.submitted = dict(vals)

    # ---- action buttons ----
    b1, b2 = st.columns([1, 1])
    with b1:
        st.button("🔄 更新预测结果", type="primary", use_container_width=True,
                  on_click=do_update)
    with b2:
        st.button("↩ 重置为中位数", use_container_width=True, on_click=do_reset)

    res = st.session_state.result

    # stale-input hint
    current = {f: float(st.session_state[f"in_{f}"]) for f in feats}
    if current != st.session_state.submitted:
        st.info("⚠️ 输入已修改，点击「🔄 更新预测结果」以刷新预测。")

    # ---- ② results ----
    st.markdown('<div class="section-title">② 预测结果</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        pct = res["prob"] * 100
        st.metric("ISR 预测概率", f"{pct:.1f}%",
                  delta=f"Youden 阈值 {res['threshold']:.3f}")
        if res["isr"]:
            st.markdown('<span class="verdict-high">⚠️ 高 ISR 风险</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="verdict-low">✅ 低 ISR 风险</span>', unsafe_allow_html=True)
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
            <span style="color:#2c3e50; font-weight:600;">Youden 阈值 {res['threshold']:.3f}</span>
            <span>100%</span>
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

    # ---- ③ SHAP ----
    st.markdown('<div class="section-title">③ 逐例 SHAP 解释</div>', unsafe_allow_html=True)
    st.markdown(
        f"下图展示每个特征如何将预测概率从基准值 **E[f(x)] = {res['expected_value']:.3f}** "
        f"推移到模型输出 **f(x) = {res['prob']:.3f}**。红色段=特征值偏高，蓝色段=偏低；"
        f"各段按特征值大小着色。"
    )
    st.pyplot(st.session_state.shap_fig, use_container_width=True)

    with st.expander("查看特征贡献明细"):
        st.dataframe(st.session_state.shap_df, use_container_width=True)

    # ---- data dictionary ----
    with st.expander("📋 数据字典（7 个特征说明）"):
        dd = pd.DataFrame([
            dict(特征=disp[f], 含义=descr[f], 单位=units[f],
                 训练集中位数=f"{stats[f]['median']:.3f}",
                 训练集范围=f"{stats[f]['min']:.3f}–{stats[f]['max']:.3f}")
            for f in feats
        ])
        st.dataframe(dd, use_container_width=True)

    # ---- disclaimer ----
    st.divider()
    st.caption(
        "⚠️ **研究用途声明**：本工具为科研演示，基于单中心推导队列训练，预测结果"
        "**不能替代临床判断或作为诊断依据**。跨中心使用时需注意特征分布差异"
        "（如残余狭窄率在不同中心的系统性偏移），建议结合本地数据谨慎解读。"
    )


if __name__ == "__main__":
    run_app()
