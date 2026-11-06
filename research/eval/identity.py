"""Identity and prompt-adherence scores for generated pet images.

- dino: DINOv2 image-image cosine vs. the reference photos (best single signal for "same animal")
- clip_i: CLIP image-image cosine vs. the references (coarser identity)
- clip_t: CLIP image-text cosine vs. the prompt (did it follow the prompt, e.g. "pirate")

Identity scores are averaged over all reference photos, as in the DreamBooth paper.
With crop_label set (e.g. "a photo of a cat"), identity is also scored on a crop around the detected
pet, so a method can't score well by copying the reference photo's background (seen in run 1).
"""
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor, pipeline

DINO_ID = "facebook/dinov2-base"
CLIP_ID = "openai/clip-vit-large-patch14"
DETECTOR_ID = "google/owlvit-base-patch32"


class IdentityScorer:
    def __init__(self, device=None, crop_label=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crop_label = crop_label
        self.detector = pipeline("zero-shot-object-detection", model=DETECTOR_ID, device=self.device) if crop_label else None
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

    def crop(self, image, margin=0.1):
        """Crop to the highest-scoring detection. Returns (image, found)."""
        hits = self.detector(image, candidate_labels=[self.crop_label])
        if not hits or hits[0]["score"] < 0.1:
            return image, False
        b = hits[0]["box"]
        mx, my = margin * (b["xmax"] - b["xmin"]), margin * (b["ymax"] - b["ymin"])
        box = (max(0, b["xmin"] - mx), max(0, b["ymin"] - my),
               min(image.width, b["xmax"] + mx), min(image.height, b["ymax"] + my))
        return image.crop(box), True

    def set_references(self, images):
        self._refs = (self._dino(images), self._clip_img(images))
        if self.detector:
            crops = [self.crop(im)[0] for im in images]
            self._ref_crops = (self._dino(crops), self._clip_img(crops))

    def score(self, image, prompt):
        ref_dino, ref_clip = self._refs
        d, c = self._dino([image]), self._clip_img([image])
        out = {
            "dino": (d @ ref_dino.T).mean().item(),
            "clip_i": (c @ ref_clip.T).mean().item(),
            "clip_t": (c @ self._clip_txt([prompt]).T).item(),
        }
        if self.detector:
            crop, found = self.crop(image)
            crop_dino, crop_clip = self._ref_crops
            out["dino_crop"] = (self._dino([crop]) @ crop_dino.T).mean().item()
            out["clip_i_crop"] = (self._clip_img([crop]) @ crop_clip.T).mean().item()
            out["pet_found"] = found
        return out
