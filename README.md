# ISR 风险预测器 · 开源在线演示版

> 颅内支架植入术后再狭窄（ISR）可解释性预测模型的在线交互版本，作为论文《基于多模态特征与全组合优化框架的颅内支架植入术后再狭窄可解释性预测模型的建立与验证》的配套开源组件。

本仓库提供一个 **Streamlit** 网页应用：输入 7 个临床/影像特征，即可得到 ISR 的**预测概率**、**高/低风险判定**，以及**逐例 SHAP 解释**（与论文图 11–12 一致的力图样式）。模型与论文最终部署模型**完全一致**（同一训练数据、同一超参数、固定随机种子）。

---

## 部署方式

### 方式一（推荐，免费）：Streamlit Community Cloud
Streamlit Community Cloud 是 Streamlit 官方免费托管平台，与本仓库一键集成，不受 Hugging Face 近期收费政策影响：

1. 打开 https://streamlit.io/cloud ，使用 **GitHub 登录**并授权 `SIAT-NazhangGroup` 组织。
2. 点击 **New app**（或 *Deploy an app*）。
3. 配置：
   - Repository：`SIAT-NazhangGroup/isr-prediction-demo`
   - Branch：`main`
   - Main file path：`app.py`
4. 点击 **Deploy**，等待 2–5 分钟（首次需安装 shap / scikit-learn 等依赖）。
5. 部署完成获得公开链接：`https://<app-name>.streamlit.app`。

> 仓库已包含 `runtime.txt`（锁定 Python 3.11）与 `requirements.txt`，确保依赖一致。之后每次 `git push` 到 `main` 分支，Community Cloud 会自动重新部署。

### 方式二（付费，留在 Hugging Face Spaces）：HF PRO + Docker
> 说明：HF 于 2025-04-30 弃用 Streamlit 内置 SDK；截至 2026-07-11，免费档仅支持 **Static**，运行 Python demo（Gradio / Docker）需订阅 **HF PRO（$9/月）**。

若仍需以 HF 链接作为引用：
1. 订阅 HF PRO。
2. 新建 Space：SDK 选 **Docker** → 使用 **Streamlit 模板** → Public。
3. 推送本仓库文件（已含 `Dockerfile`）；并将本文件顶部元数据改为 `sdk: docker` 并指定 `app_port: 8501`（Streamlit 默认监听 8501，而非 HF Docker 默认的 7860）。
4. 即可在 `https://huggingface.co/spaces/SIAT-NazhangGroup/isr-prediction-demo` 构建运行。

---

## 模型信息

| 项目 | 数值 |
|---|---|
| 算法 | 极端随机树 ExtraTreesClassifier（n_estimators=100, max_depth=4, random_state=42） |
| 特征数 | 7（多模态：斑块形态学 4 + 影像组学 2 + 临床 1） |
| 训练队列 | 内部推导队列 n=237（ISR 38，患病率 16.0%） |
| 内部交叉验证 AUC | **0.823**（95% CI 0.670–0.954） |
| 分类阈值（Youden） | **0.173**（基于 OOF 预测） |
| 基准概率 E[f(x)] | **0.160** |
| 预处理 | 各特征 StandardScaler 标准化 |

---

## 数据字典（7 个特征）

| 特征 | 含义 | 单位 | 训练集中位数 | 训练集范围 |
|---|---|---|---|---|
| `Mean_nwi_Total` | 全段平均管壁指数 (Mean NWI) | 无量纲 | 0.497 | 0.168–0.934 |
| `Residual_stenosis_rate` | 残余狭窄率 | 比例 (0–1) | 0.343 | 0.000–0.857 |
| `Mean_lumen_area_Total` | 全段平均管腔面积 | mm² | 2.054 | 0.383–5.559 |
| `wavelet-HLL_glcm_Correlation` | 影像组学纹理特征 (wavelet-HLL GLCM 相关性) | 无量纲 | 0.660 | 0.108–0.998 |
| `original_shape_Flatness` | 影像组学形状特征 (扁平度) | 无量纲 | 0.188 | 0.046–0.484 |
| `Max_nwi_Total` | 全段最大管壁指数 (Max NWI) | 无量纲 | 0.690 | 0.296–1.000 |
| `TC` | 总胆固醇 (Total Cholesterol) | mmol/L | 4.220 | 2.410–6.890 |

> 注：以上范围/中位数来自内部推导队列，仅用于约束界面输入；模型本身不要求输入必须落在该范围内。

---

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

首次运行前需生成模型文件（已包含在 `model/` 中；如需重新导出，运行 `python export_pipeline.py`，需能访问原始训练数据 Excel）。

---

## 逐例 SHAP 解释

每次预测都会用 `shap.TreeExplainer` 计算该患者的特征贡献，并渲染力图为：从基准值 `E[f(x)] = 0.160` 出发，各特征段按其取值（蓝→红）将预测概率推移到模型输出 `f(x)`。红色段=特征值偏高，蓝色段=偏低。该可视化与论文图 11/12 风格一致，体现模型的可解释性。

---

## ⚠️ 重要说明（跨中心使用）

- 模型在**单中心推导队列**上训练。外部验证显示，不同中心间**残余狭窄率**分布存在系统性偏移（外部队列均值约 0.158，明显低于内部 0.343），会使外部队列预测概率被整体压低、敏感性下降；在将残余狭窄率均值对齐后，外部 AUC 由 0.656 升至 0.692、敏感性由 0.286 升至 0.571。
- 因此跨中心直接使用时，建议注意特征分布差异，必要时参照论文的"残余狭窄率均值对齐"敏感性分析思路进行本地校准。
- **研究用途声明**：本工具仅供科研演示，**不能替代临床判断或作为诊断依据**。

---

## 引用

> （论文正式发表后在此处填入引用信息，例如：）
> Author et al. *Development and validation of an interpretable multimodal prediction model for in-stent restenosis after intracranial stenting.* Journal, Year.

## 文件结构

```
ISR_online/
├── app.py                 # Streamlit 主程序
├── export_pipeline.py     # 模型导出脚本（生成 model/pipeline.joblib）
├── requirements.txt       # Python 依赖
├── runtime.txt            # 锁定 Python 3.11（Streamlit Community Cloud）
├── Dockerfile             # HF PRO + Docker 部署备选
├── test_headless.py       # 无头逻辑测试（开发者用）
├── model/
│   ├── pipeline.joblib     # 序列化的 StandardScaler + ET + 阈值 + 基准值 + 特征元数据
│   └── model_card.json     # 数据字典（机器可读）
└── README.md              # 本文件
```
