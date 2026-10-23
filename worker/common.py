import base64
import io

import requests
from PIL import Image, ImageOps


def load_image(src: str) -> Image.Image:
    """Load from an http(s) URL or base64 (data: URIs tolerated)."""
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        data = r.content
    else:
        data = base64.b64decode(src.split(",", 1)[-1])
    # Phone photos are often stored sideways with an EXIF rotation flag
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def upload(path: str, key: str):
    """Upload to any S3-compatible bucket (R2, S3, B2). Returns the public URL, or None if not configured."""
    import os

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


def to_fp16(path: str):
    """Rewrite a LoRA safetensors file in fp16 (half the size; no visible quality difference)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(path, framework="pt") as f:
        tensors = {k: f.get_tensor(k).half() for k in f.keys()}
        metadata = f.metadata()
    save_file(tensors, path, metadata=metadata)
