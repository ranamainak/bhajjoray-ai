FROM python:3.10-slim

WORKDIR /app

ENV HF_HOME=/app/model_cache
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the lightweight 1.1 GB 4-bit GGUF model during cloud build
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF', filename='DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf', local_dir='/app/models')"

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
