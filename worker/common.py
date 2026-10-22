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
