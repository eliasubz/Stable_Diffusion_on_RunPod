"""DreamBooth-LoRA (SDXL) training worker: 3-10 pet photos in, LoRA weights out.

Wraps diffusers' official train_dreambooth_lora_sdxl.py in a subprocess so the GPU is fully
released when training ends.

Input:
  images (list of base64 or http(s) URLs, required), class_word ("cat" / "dog"),
  token (rare identifier, default "sks"), steps, learning_rate, rank, resolution,
  prior_preservation (bool, slower), seed, checkpoint_every (also keep LoRAs from earlier steps)
Output:
  lora_url (when S3_* env is set) or lora_path (local / Colab), instance_prompt,
  timings_ms {prep, train, upload, total}, step_times_s {step: seconds into training},
  checkpoints {step: local path} (with checkpoint_every)
"""
import collections
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import runpod
from PIL import ImageOps

from common import load_image

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "train_dreambooth_lora_sdxl.py")
# Pinned to the installed diffusers version; the script checks it on startup.
SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/v0.32.2/examples/dreambooth/train_dreambooth_lora_sdxl.py"
OUT_ROOT = os.environ.get("LORA_DIR", os.path.join(HERE, "loras"))


def ensure_script():
    if not os.path.exists(SCRIPT):
        urllib.request.urlretrieve(SCRIPT_URL, SCRIPT)


# tqdm line from the diffusers script, e.g. "Steps:  12%|█▏ | 60/500 [01:02<07:40,  1.05s/it, loss=0.12, lr=0.0001]"
STEP_RE = re.compile(r"Steps:.*?(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)(?:.*?loss=([\d.e+-]+))?")
# Milestones from the script worth showing; everything else (config warnings etc.) only goes to the log tail
KEY_LINES = ("Running training", "Num examples", "Total optimization steps", "Traceback", "Error", "OutOfMemory")


def run_training(cmd, progress=None):
    """Run the training script, printing compact progress.

    Returns (returncode, last 200 log lines, {step: seconds since launch} for each printed step).
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TRANSFORMERS_VERBOSITY": "error", "DIFFUSERS_VERBOSITY": "error"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    tail = collections.deque(maxlen=200)
    last_step, buf = -1, ""
    t0, step_times = time.perf_counter(), {}

    # tqdm redraws with \r, so split on both \r and \n
    for chunk in iter(lambda: proc.stdout.read(256), ""):
        buf += chunk
        *lines, buf = re.split(r"[\r\n]", buf)
        for line in filter(None, (l.strip() for l in lines)):
            tail.append(line)
            m = STEP_RE.search(line)
            if m:
                step, total = int(m[1]), int(m[2])
                if step != last_step and (step % max(1, total // 20) == 0 or step == total):
                    last_step = step
                    step_times[step] = round(time.perf_counter() - t0, 1)
                    speed = f"{m[5]} {m[6]}"
                    loss = f"  loss {float(m[7]):.4f}" if m[7] else ""
                    msg = f"{step:>4}/{total} ({step / total:4.0%})  {speed}{loss}  elapsed {m[3]}  eta {m[4]}"
                    print(f"[train] {msg}", flush=True)
                    if progress:
                        progress(msg)
            elif any(k in line for k in KEY_LINES):
                print(f"[train] {line}", flush=True)

    return proc.wait(), list(tail), step_times


def to_fp16(path: str):
    """Halve the LoRA file size (diffusers saves fp32); no visible quality difference."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(path, framework="pt") as f:
        tensors = {k: f.get_tensor(k).half() for k in f.keys()}
        metadata = f.metadata()
    save_file(tensors, path, metadata=metadata)


def collect_checkpoints(out_dir: str) -> dict:
    """Keep only the LoRA weights from each checkpoint-N dir (drops optimizer/scheduler state)."""
    found = {}
    for ckpt in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        step = int(ckpt.rsplit("-", 1)[1])
        src = os.path.join(ckpt, "pytorch_lora_weights.safetensors")
        if os.path.exists(src):
            dst = os.path.join(out_dir, f"lora-step{step}.safetensors")
            shutil.move(src, dst)
            to_fp16(dst)
            found[step] = dst
        shutil.rmtree(ckpt, ignore_errors=True)
    return dict(sorted(found.items()))


def upload(path: str, key: str):
    """Upload to any S3-compatible bucket (R2, S3, B2). Returns None if not configured."""
    if not os.environ.get("S3_BUCKET"):
        return None
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )
    s3.upload_file(path, os.environ["S3_BUCKET"], key)
    return f"{os.environ['S3_PUBLIC_URL'].rstrip('/')}/{key}"


def train(inp: dict, progress=None) -> dict:
    images = inp.get("images") or []
    if len(images) < 3:
        return {"error": "need at least 3 images"}

    t_start = time.perf_counter()
    timings = {}
    class_word = inp.get("class_word", "pet")
    token = inp.get("token", "sks")
    instance_prompt = f"a photo of {token} {class_word}"
    resolution = int(inp.get("resolution", 1024))
    run_id = inp.get("run_id") or time.strftime("%Y%m%d-%H%M%S")

    work = tempfile.mkdtemp(prefix="dreambooth-")
    data_dir = os.path.join(work, "instance")
    os.makedirs(data_dir)
    out_dir = os.path.join(OUT_ROOT, run_id)

    t = time.perf_counter()
    ensure_script()
    for i, src in enumerate(images):
        # Center-crop to square so the script's own crop doesn't cut off the pet
        img = ImageOps.fit(load_image(src), (resolution, resolution))
        img.save(os.path.join(data_dir, f"{i:02d}.png"))
    timings["prep"] = (time.perf_counter() - t) * 1000

    cmd = [
        sys.executable, SCRIPT,
        f"--pretrained_model_name_or_path={MODEL_ID}",
        # fp16 weights are the ones baked into the image (no 14GB fp32 download at runtime). No separate
        # VAE: the fp16-fix repo has no fp16 variant, and the script keeps the stock VAE in fp32 instead.
        "--variant=fp16",
        f"--instance_data_dir={data_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--output_dir={out_dir}",
        f"--resolution={resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        # No 8-bit Adam: LoRA optimizer state is ~100s of MB, and bitsandbytes breaks on new CUDA/Python builds
        "--mixed_precision=fp16",
        f"--learning_rate={float(inp.get('learning_rate', 1e-4))}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={int(inp.get('steps', 500))}",
        f"--rank={int(inp.get('rank', 8))}",
        f"--checkpointing_steps={int(inp.get('checkpoint_every', 100000))}",  # default: none
        f"--seed={int(inp.get('seed', 0))}",
    ]
    if inp.get("prior_preservation"):
        cmd += [
            "--with_prior_preservation",
            "--prior_loss_weight=1.0",
            f"--class_data_dir={os.path.join(work, 'class')}",
            f"--class_prompt=a photo of a {class_word}",
            f"--num_class_images={int(inp.get('num_class_images', 50))}",
        ]

    print(f"[train] {len(images)} photos, {instance_prompt!r}, {resolution}px, "
          f"{inp.get('steps', 500)} steps, rank {inp.get('rank', 8)}", flush=True)
    t = time.perf_counter()
    returncode, log, step_times = run_training(cmd, progress)
    timings["train"] = (time.perf_counter() - t) * 1000
    shutil.rmtree(work, ignore_errors=True)
    if returncode != 0:
        print(f"[train] failed after {timings['train'] / 60000:.1f} min (exit {returncode})", flush=True)
        return {"error": "training failed", "log_tail": "\n".join(log[-40:])}

    lora_path = os.path.join(out_dir, "pytorch_lora_weights.safetensors")
    to_fp16(lora_path)
    checkpoints = collect_checkpoints(out_dir)
    t = time.perf_counter()
    lora_url = upload(lora_path, f"loras/{run_id}.safetensors")
    timings["upload"] = (time.perf_counter() - t) * 1000
    timings["total"] = (time.perf_counter() - t_start) * 1000

    print(f"[train] done: {timings['train'] / 60000:.1f} min training, {timings['total'] / 60000:.1f} min total, "
          f"LoRA {os.path.getsize(lora_path) / 1e6:.0f} MB", flush=True)
    out = {"instance_prompt": instance_prompt, "timings_ms": timings, "step_times_s": step_times,
           "size_mb": os.path.getsize(lora_path) / 1e6}
    if checkpoints:
        out["checkpoints"] = checkpoints
    if lora_url:
        out["lora_url"] = lora_url
    else:
        out["lora_path"] = lora_path
    return out


def handler(job):
    # Progress shows up in the RunPod /status response while the job runs
    return train(job["input"], progress=lambda msg: runpod.serverless.progress_update(job, msg))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
