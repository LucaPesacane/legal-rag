FROM python:3.11-slim

# [+] non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER user
# [~] owned by the user, so the app can write logs and vector DBs
WORKDIR /home/user/app

# [+] CPU-only torch
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# [+] embedding model downloaded at build time
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', ignore_patterns=['onnx/*', '*.onnx*'])"

COPY --chown=user . .

ENV PYTHONUNBUFFERED=1
# [+] ragas imports GitPython, but git is not installed
ENV GIT_PYTHON_REFRESH=quiet
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
