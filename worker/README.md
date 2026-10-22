# RunPod SDXL + IP-Adapter worker

Serverless worker that the proxy server's BullMQ queue calls to make pet images.

## Build & deploy
```bash
docker build --platform linux/amd64 -t <dockerhub-user>/petsalive-sdxl:0.1 .
docker push <dockerhub-user>/petsalive-sdxl:0.1
```
On RunPod, go to Serverless → New Endpoint → Custom image, pick a 24GB GPU (A5000/4090/L4), and set min workers to 0 and max to 2.
Put the endpoint ID and API key in the proxy server's `.env` (`RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`).

## Local test (needs a CUDA GPU)
`python handler.py --test_input "$(cat test_input.json)"` (runpod SDK local mode)

## Input / output
See the docstring in `handler.py`. `timings_ms` comes back with every response. That's the raw data for the latency benchmarks.

## Training endpoint (DreamBooth-LoRA)
Same image, second endpoint: set the start command to `python -u train.py`, and give it a longer execution timeout (~30 min).
Set `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` and `S3_PUBLIC_URL` (any S3-compatible bucket, e.g. Cloudflare R2) so it returns a `lora_url`.
Pass that URL as `lora` to the generation endpoint.

## Week 2 experiment
`week2_dreambooth.ipynb` trains a LoRA on one pet, then compares text-only / IP-Adapter / LoRA on DINO, CLIP-I and CLIP-T scores and latency. Run it on Colab.
