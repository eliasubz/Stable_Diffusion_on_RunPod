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
