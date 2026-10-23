# Built for RunPod L4s (Ada, sm89). Also runs on any Ampere+ GPU; training picks bf16 automatically.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV HF_HOME=/app/hf-cache PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_weights.py .
RUN python download_weights.py

COPY common.py multi_lora.py train_fast.py describe.py handler.py train.py ./

# Generation endpoint by default. For the training endpoint, set the container start command to
# `python -u train.py` in the RunPod endpoint settings (same image, no second build).
CMD ["python", "-u", "handler.py"]
