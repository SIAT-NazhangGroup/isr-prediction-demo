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

## 方式一（推荐）：用 GitHub Actions 自动同步到 HF Space

> 本仓库已内置工作流 `.github/workflows/sync-to-hub.yml`（官方
> `huggingface/hub-sync` Action）。它在每次 push 到 `main` 时，由 **GitHub 的
> runner（美国，不受本机/本环境网络限制）** 自动把内容镜像为一个 Hugging Face
> Space——**首次运行会自动创建 Space，无需在 HF 网页手动建立**。

你只需做两件事：

1. 打开 https://github.com/SIAT-NazhangGroup/isr-prediction-demo ，
   进入 **Settings → Secrets and variables → Actions → New repository secret**。
2. Name 填 `HF_TOKEN`，Value 填你的 Hugging Face 访问令牌（即 `hf_...` 那个），保存。
3. 进入仓库 **Actions** 标签，找到 "Sync to Hugging Face Hub" 工作流，点
   **Run workflow**（之后每次 push 会自动触发）。

几分钟后 HF Space 自动创建于
`https://huggingface.co/spaces/SIAT-NazhangGroup/isr-prediction-demo`，
并按根目录 `README.md` 的 frontmatter 用 Streamlit 运行 `app.py` 构建在线 demo。

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
