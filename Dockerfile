# Dockerfile for deploying the ISR Streamlit demo on Hugging Face Spaces
# (Streamlit built-in SDK is deprecated; use Docker SDK + Streamlit template)
# HF Docker Spaces route to app_port (set in README.md YAML). Streamlit listens on 8501.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit's default port is 8501 (not 7860). Keep README app_port in sync.
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
