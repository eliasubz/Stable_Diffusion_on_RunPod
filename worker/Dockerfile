FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV HF_HOME=/app/hf-cache PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_weights.py .
RUN python download_weights.py

COPY handler.py .
CMD ["python", "-u", "handler.py"]
