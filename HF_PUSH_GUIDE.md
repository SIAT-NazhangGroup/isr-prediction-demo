# 将本仓库部署为 Hugging Face Space —— 操作指南

> **重要前提**：当前用于自动化的 WorkBuddy 执行环境**无法访问 Hugging Face**
> （`huggingface.co` 被网络层以 SNI 方式封锁、DNS 也被污染）。因此下面两步
> **需要你在自己能联网的机器上执行**（你本机 Windows / 可连 HF 的服务器均可），
> 自动化环境只能帮你把代码和模型准备好，并已在 GitHub 上发布完整仓库。

代码与模型已在 GitHub 就绪：
**https://github.com/SIAT-NazhangGroup/isr-prediction-demo**

本目录（`ISR_online`）也已是最新状态，含 HF Space 所需的 `README.md` frontmatter
（`sdk: streamlit` 等），push 后 HF 会自动构建在线 demo。

---

## 方式一（推荐，最简单）：HF 网页从 GitHub 一键导入

1. 浏览器打开 https://huggingface.co 并登录（用的组织账号 SIAT-NazhangGroup）。
2. 右上角 **New** → **Space**。
3. 选择 **Streamlit** 作为 SDK，命名 `isr-prediction-demo`，设为 **Public**。
4. 创建后在 Space 的 **Settings** 页面找到 "**Link a GitHub repository**"，
   选择 `SIAT-NazhangGroup/isr-prediction-demo`（已发布好的仓库），按提示绑定并同步。
   - 部分版本在创建页即可直接选 "Import from GitHub repository"，效果相同。
5. 绑定后 HF 会按仓库根目录 `README.md` 的 frontmatter 自动 `streamlit run app.py` 构建，
   几分钟后给出公开 URL，可直接写进论文作交互式补充材料。

---

## 方式二：本地 git 推送（命令行）

在你本机（能连 HF）的终端执行：

```bash
# 1) 安装并登录 HF CLI（仅需第一次）
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # 粘贴你的 HF token

# 2) 在 HF 网页或用 CLI 创建 Space（组织 SIAT-NazhangGroup，SDK=streamlit，Public）
huggingface-cli repo create isr-prediction-demo --type space --sdk streamlit --organization SIAT-NazhangGroup --public

# 3) 进入本仓库目录并推送
cd C:\Users\lpp\Desktop\ISR_paper\ISR_online
git remote add hf https://SIAT-NazhangGroup:<YOUR_HF_TOKEN>@huggingface.co/spaces/SIAT-NazhangGroup/isr-prediction-demo
git push -u hf main
git remote remove hf          # 推送后移除含 token 的 remote，避免凭证残留
```

推送成功后 HF 自动构建，URL 形如
`https://huggingface.co/spaces/SIAT-NazhangGroup/isr-prediction-demo`。

---

## 备注

- `model/pipeline.joblib` 已随 GitHub 仓库发布，HF Space 直接使用，无需重新导出。
- 如需更新模型/代码：在本地改完 `git push` 到 GitHub，再到 HF Space 重新同步（方式一）
  或直接 `git push` 到 HF remote（方式二）即可。
- **安全提醒**：本项目的 HF / GitHub token 曾在对话中明文出现，发布完成后请到对应平台
  后台轮换（revoke）这两个 token。
