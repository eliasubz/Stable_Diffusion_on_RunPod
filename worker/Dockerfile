FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV HF_HOME=/app/hf-cache PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_weights.py .
RUN python download_weights.py

COPY common.py handler.py train.py ./
RUN python -c "import train; train.ensure_script()"

# Generation endpoint by default; for the training endpoint override the start command with
# `python -u train.py` in the RunPod endpoint settings (same image, no second build).
CMD ["python", "-u", "handler.py"]
