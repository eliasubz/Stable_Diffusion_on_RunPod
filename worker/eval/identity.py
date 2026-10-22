"""Identity and prompt-adherence scores for generated pet images.

- dino: DINOv2 image-image cosine vs. the reference photos (best single signal for "same animal")
- clip_i: CLIP image-image cosine vs. the references (coarser identity)
- clip_t: CLIP image-text cosine vs. the prompt (did it follow the prompt, e.g. "pirate")

Identity scores are averaged over all reference photos, as in the DreamBooth paper.
"""
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

DINO_ID = "facebook/dinov2-base"
CLIP_ID = "openai/clip-vit-large-patch14"


class IdentityScorer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
        self.dino = AutoModel.from_pretrained(DINO_ID).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
        self.clip = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
        self._refs = None

    @torch.no_grad()
    def _dino(self, images):
        x = self.dino_proc(images=images, return_tensors="pt").to(self.device)
        return F.normalize(self.dino(**x).last_hidden_state[:, 0], dim=-1)  # CLS token

    @torch.no_grad()
    def _clip_img(self, images):
        x = self.clip_proc(images=images, return_tensors="pt").to(self.device)
        return F.normalize(self.clip.get_image_features(**x), dim=-1)

    @torch.no_grad()
    def _clip_txt(self, texts):
        x = self.clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return F.normalize(self.clip.get_text_features(**x), dim=-1)

    def set_references(self, images):
        self._refs = (self._dino(images), self._clip_img(images))

    def score(self, image, prompt):
        ref_dino, ref_clip = self._refs
        d, c = self._dino([image]), self._clip_img([image])
        return {
            "dino": (d @ ref_dino.T).mean().item(),
            "clip_i": (c @ ref_clip.T).mean().item(),
            "clip_t": (c @ self._clip_txt([prompt]).T).item(),
        }
