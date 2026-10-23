"""Pet description with Qwen2-VL-2B: a few photos in, a short appearance phrase out.

Run 4 showed adding the description to every prompt removes identity failures with fast samplers.
The model lives on the CPU between jobs and moves to the GPU only while describing, so it doesn't
compete with training for VRAM.
"""
import re
import time

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
# Run 4's prompt, which produced the description used for the production config
PROMPT = (
    "These photos all show the same {cls}. Describe only the {cls}'s appearance for an image generation "
    "prompt: likely breed, coat color and pattern, fur length, eye color and build. Ignore the background. "
    "Answer with one short comma-separated phrase of at most 15 words, without mentioning the photos."
)


class PetDescriber:
    def __init__(self, model_id=VLM_ID):
        # sdpa: the default "eager" vision attention materializes a full [patches x patches] matrix per head
        # (a single ~3.7 GB allocation for 4 photos), which doesn't fit next to the training UNet on a T4
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                     attn_implementation="sdpa")
        # Breed, coat and eye color don't need high resolution; fewer patches = less memory and time
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)

    @torch.no_grad()
    def describe(self, images, class_word="pet"):
        """images: PIL images (the first 4 are used). Returns (description, raw model output, seconds)."""
        t = time.perf_counter()
        self.model.to("cuda")
        try:
            imgs = images[:4]
            cls = class_word if class_word in ("cat", "dog") else "pet"
            messages = [{"role": "user", "content": [*({"type": "image"} for _ in imgs),
                                                     {"type": "text", "text": PROMPT.format(cls=cls)}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
            out = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
            raw = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        finally:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        # Keep it prompt-shaped: one line, no "the cat is", no trailing period, at most 20 words
        desc = re.sub(rf"^(the |this )?{cls} (is|has|appears to be)\s*", "", raw.strip().splitlines()[0], flags=re.I)
        desc = " ".join(desc.rstrip(".").split()[:20])
        return desc, raw, time.perf_counter() - t
