# -*- coding: utf-8 -*-
"""
Export the trained ISR prediction pipeline for the open-source online demo.

This script replicates the EXACT training logic used to produce the model
reported in the paper (gen_et_all.py): same features, same ExtraTreesClassifier
hyper-parameters, same random_state=42, trained on the FULL internal derivation
cohort. The exported artifact is therefore identical to the deployment model
behind tab_internal (AUC = 0.823) and the SHAP figures.

Output: model/pipeline.joblib  (self-contained, no Excel/data needed at runtime)
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve
import shap

# ---- paths (same as gen_et_all.py) -----------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INT_DIR = r"C:/Users/lpp/Desktop/ISR_paper/final_experience/调优模型实验/final_experience/data/"
MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_7 = ["Mean_nwi_Total", "Residual_stenosis_rate", "Mean_lumen_area(mm²)_Total",
              "wavelet-HLL_glcm_Correlation", "original_shape_Flatness", "Max_nwi_Total", "TC"]
DISPLAY = {"Mean_nwi_Total": "Mean_nwi_Total", "Residual_stenosis_rate": "Residual_stenosis_rate",
           "Mean_lumen_area(mm²)_Total": "Mean_lumen_area_Total", "wavelet-HLL_glcm_Correlation": "wavelet-HLL_glcm_Correlation",
           "original_shape_Flatness": "original_shape_Flatness", "Max_nwi_Total": "Max_nwi_Total", "TC": "TC"}
LABEL = "Label"

# data dictionary (units + plain-language description) for the UI / model card
UNITS = {
    "Mean_nwi_Total": "无量纲",
    "Residual_stenosis_rate": "比例 (0–1)",
    "Mean_lumen_area(mm²)_Total": "mm²",
    "wavelet-HLL_glcm_Correlation": "无量纲",
    "original_shape_Flatness": "无量纲",
    "Max_nwi_Total": "无量纲",
    "TC": "mmol/L",
}
DESCR = {
    "Mean_nwi_Total": "全段平均管壁指数 (Mean NWI)",
    "Residual_stenosis_rate": "残余狭窄率",
    "Mean_lumen_area(mm²)_Total": "全段平均管腔面积",
    "wavelet-HLL_glcm_Correlation": "影像组学纹理特征 (wavelet-HLL GLCM 相关性)",
    "original_shape_Flatness": "影像组学形状特征 (扁平度)",
    "Max_nwi_Total": "全段最大管壁指数 (Max NWI)",
    "TC": "总胆固醇 (Total Cholesterol)",
}


def et_model():
    return ExtraTreesClassifier(n_estimators=100, max_depth=4, max_features='sqrt',
                                min_samples_split=10, min_samples_leaf=1, criterion='gini',
                                class_weight=None, random_state=42, n_jobs=-1)


def load_int():
    dfs = [pd.read_excel(INT_DIR + f) for f in ["斑块指标特征.xlsx", "临床特征.xlsx", "影像组学特征_cleaned.xlsx"]]
    m = dfs[0]
    for n in dfs[1:]:
        n = n.drop(columns=[LABEL], errors="ignore")
        m = m.merge(n, on="patient_id", how="inner")
    return m


def youden(y_true, y_prob):
    fpr, tpr, th = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(th[int(np.argmax(j))])


def main():
    df = load_int()
    Xi = df[FEATURES_7].apply(pd.to_numeric, errors="coerce").fillna(0).values
    yi = pd.to_numeric(df[LABEL]).astype(int).values
    print(f"Loaded internal cohort: n={len(Xi)}, ISR={int(yi.sum())} ({yi.mean():.3f})")

    # ---- train on FULL internal cohort (deployment model) ----
    sc = StandardScaler().fit(Xi)
    m = et_model()
    m.fit(sc.transform(Xi), yi)

    # ---- Youden threshold from 10x5-fold OOF (matches paper int_th ~0.173) ----
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    oot, oop = [], []
    for tr, te in cv.split(Xi, yi):
        s2 = StandardScaler().fit(Xi[tr])
        mm = et_model()
        mm.fit(s2.transform(Xi[tr]), yi[tr])
        oop.extend(mm.predict_proba(s2.transform(Xi[te]))[:, 1].tolist())
        oot.extend(yi[te].tolist())
    oot = np.array(oot); oop = np.array(oop)
    threshold = youden(oot, oop)
    print(f"Youden threshold (OOF) = {threshold:.4f}")

    # ---- SHAP expected value (base probability) ----
    expl = shap.TreeExplainer(m)
    ev = float(np.asarray(expl.expected_value)[1])  # positive class, probability space
    print(f"SHAP expected value (base prob) = {ev:.4f}")

    # ---- per-feature stats for UI sliders ----
    stats = {}
    for j, f in enumerate(FEATURES_7):
        col = Xi[:, j]
        stats[f] = dict(min=float(col.min()), max=float(col.max()),
                        median=float(np.median(col)), mean=float(col.mean()))

    bundle = dict(
        scaler=sc,
        model=m,
        threshold=threshold,
        expected_value=ev,
        features=FEATURES_7,
        display=DISPLAY,
        units=UNITS,
        descr=DESCR,
        stats=stats,
        meta=dict(
            model_type="ExtraTreesClassifier",
            n_estimators=100, max_depth=4, random_state=42,
            trained_on="Full internal derivation cohort (n=237, ISR=38)",
            internal_oof_auc=0.823,
            youden_threshold=threshold,
            base_probability=ev,
            feature_count=len(FEATURES_7),
        ),
    )
    out_path = os.path.join(MODEL_DIR, "pipeline.joblib")
    joblib.dump(bundle, out_path)
    print(f"Saved pipeline -> {out_path}")

    # human-readable sidecar (data dictionary)
    with open(os.path.join(MODEL_DIR, "model_card.json"), "w", encoding="utf-8") as fh:
        json.dump({k: bundle[k] for k in ["features", "display", "units", "descr", "stats", "meta"]},
                  fh, ensure_ascii=False, indent=2)
    print("Saved model_card.json (data dictionary)")


if __name__ == "__main__":
    main()
