# -*- coding: utf-8 -*-
"""
ISR Risk Predictor — open-source Streamlit demo
(Companion to the paper: multimodal + full-combination ISR prediction model)

Loads model/pipeline.joblib (exported by export_pipeline.py) and provides:
  - 7-feature input -> ISR probability + ISR/non-ISR verdict (Youden threshold)
  - per-instance SHAP explanation (force plot, native style, matches the paper)
  - data dictionary + research-use disclaimer

Deploy: push this folder to a Hugging Face Space (Streamlit SDK).
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
    pipeline = load_pipeline()
    feats = pipeline["features"]
    disp = pipeline["display"]
    units = pipeline["units"]
    descr = pipeline["descr"]
    stats = pipeline["stats"]
    meta = pipeline["meta"]

    st.title("🩺 颅内支架植入术后再狭窄（ISR）风险预测器")
    st.caption("基于多模态特征与全组合优化框架的可解释预测模型 · 开源在线演示版")

    st.markdown(
        "输入以下 **7 个特征**（与论文最终模型一致），获得该患者发生 ISR 的**预测概率**及"
        "**逐例 SHAP 解释**。模型在内部推导队列（n=237）上训练，交叉验证 AUC = **0.823**。"
    )

    # ---- inputs ----
    st.subheader("① 输入特征")
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(feats):
        s = stats[f]
        with cols[i % 2]:
            step = 0.1 if f == "TC" else 0.001
            val = st.number_input(
                label=f"**{disp[f]}**  ({descr[f]})",
                min_value=float(s["min"]), max_value=float(s["max"]),
                value=float(s["median"]), step=step, format="%.4f",
                help=f"单位：{units[f]}；训练集范围 {s['min']:.3f}–{s['max']:.3f}",
            )
            values[f] = val

    # ---- predict ----
    res = predict(pipeline, values)
    st.subheader("② 预测结果")

    c1, c2 = st.columns([1, 2])
    with c1:
        pct = res["prob"] * 100
        st.metric("ISR 预测概率", f"{pct:.1f}%",
                  delta=f"Youden 阈值 {res['threshold']:.3f}")
        verdict = "⚠️ 高 ISR 风险" if res["isr"] else "✅ 低 ISR 风险"
        st.markdown(f"### {verdict}")
    with c2:
        # simple horizontal bar
        fig, ax = plt.subplots(figsize=(5, 0.6))
        ax.barh(0, res["prob"], color="#D55E00", height=0.5)
        ax.barh(0, 1 - res["prob"], left=res["prob"], color="#E5E5E5", height=0.5)
        ax.axvline(res["threshold"], color="#0072B2", lw=1.5, ls="--")
        ax.text(res["threshold"], 0.45, f"阈值 {res['threshold']:.3f}",
                color="#0072B2", fontsize=8, ha="center", va="bottom")
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("ISR probability")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    # ---- SHAP ----
    st.subheader("③ 逐例 SHAP 解释")
    st.markdown(
        f"下图展示每个特征如何将预测概率从基准值 **E[f(x)] = {res['expected_value']:.3f}** "
        f"推移到模型输出 **f(x) = {res['prob']:.3f}**。红色段=特征值偏高，蓝色段=偏低；"
        f"各段按特征值大小着色。"
    )
    fig = shap_force_figure(pipeline, res)
    st.pyplot(fig, use_container_width=True)

    with st.expander("查看特征贡献明细"):
        st.dataframe(shap_table(pipeline, res), use_container_width=True)

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
