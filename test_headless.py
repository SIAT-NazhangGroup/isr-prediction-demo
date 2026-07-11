# -*- coding: utf-8 -*-
"""Headless sanity test for the ISR online app (no Streamlit server)."""
import os, sys, json
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app as appmod

INT_DIR = r"C:/Users/lpp/Desktop/ISR_paper/final_experience/调优模型实验/final_experience/data/"
FEATURES_7 = ["Mean_nwi_Total","Residual_stenosis_rate","Mean_lumen_area(mm²)_Total",
              "wavelet-HLL_glcm_Correlation","original_shape_Flatness","Max_nwi_Total","TC"]

def load_int():
    dfs=[pd.read_excel(INT_DIR+f) for f in ["斑块指标特征.xlsx","临床特征.xlsx","影像组学特征_cleaned.xlsx"]]
    m=dfs[0]
    for n in dfs[1:]:
        n=n.drop(columns=["Label"],errors="ignore"); m=m.merge(n,on="patient_id",how="inner")
    return m

pl = joblib.load(os.path.join(HERE,"model","pipeline.joblib"))
print("loaded pipeline: threshold=%.4f base=%.4f" % (pl["threshold"], pl["expected_value"]))

df = load_int()
Xi = df[FEATURES_7].apply(pd.to_numeric,errors="coerce").fillna(0).values

# predict on every internal patient with exported pipeline
probs = []
for i in range(Xi.shape[0]):
    vals = {f: float(Xi[i,j]) for j,f in enumerate(FEATURES_7)}
    r = appmod.predict(pl, vals)
    probs.append(r["prob"])
probs = np.array(probs)
hi = int(np.argmax(probs)); lo = int(np.argmin(probs))
print("MAX prob patient: %.3f  (paper Fig.11 ISR patient = 0.557)" % probs[hi])
print("MIN prob patient: %.3f  (paper Fig.12 non-ISR patient = 0.037)" % probs[lo])

# SHAP on the max-prob patient
vals_hi = {f: float(Xi[hi,j]) for j,f in enumerate(FEATURES_7)}
res = appmod.predict(pl, vals_hi)
fig = appmod.shap_force_figure(pl, res)
import matplotlib.pyplot as plt
plt.close(fig)
tbl = appmod.shap_table(pl, res)
print("SHAP force figure rendered OK; top contributor:", tbl.iloc[0]["Feature"], "SHAP=%.3f" % tbl.iloc[0]["SHAP"])

# sanity: predict() prob == model.predict_proba directly
Xs = pl["scaler"].transform(Xi[hi:hi+1])
direct = float(pl["model"].predict_proba(Xs)[0,1])
assert abs(direct - probs[hi]) < 1e-9, "predict() mismatch"
print("predict() == direct model prob: OK")

print("ALL HEADLESS CHECKS PASSED")
