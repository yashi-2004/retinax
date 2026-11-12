"""Optional BiomedCLIP vision-language foundation model experiments.

Provenance & License:
- Model: Microsoft BiomedCLIP (`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`).
- License: MIT (Microsoft Research / PMC-15M pretraining).
- Execution Policy: Strictly optional. Never crashes RetinaX if `open_clip_torch`
  or model weights (~780 MB) are not installed/cached locally.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import torch
from PIL import Image

DEFAULT_DR_CLINICAL_PROMPTS: dict[int, str] = {
    0: "a retinal fundus photograph showing a healthy retina with no diabetic retinopathy",
    1: "a retinal fundus photograph showing mild non-proliferative diabetic retinopathy with sparse microaneurysms",
    2: "a retinal fundus photograph showing moderate diabetic retinopathy with blot hemorrhages and hard exudates",
    3: "a retinal fundus photograph showing severe non-proliferative diabetic retinopathy with extensive retinal hemorrhages and cotton wool spots",
    4: "a retinal fundus photograph showing proliferative diabetic retinopathy with neovascularization and widespread lesions",
}


def check_biomedclip_availability() -> dict[str, Any]:
    """Check whether optional `open_clip` dependency is installed without forcing download."""
    has_open_clip = importlib.util.find_spec("open_clip") is not None
    has_transformers = importlib.util.find_spec("transformers") is not None
    available = has_open_clip and has_transformers
    reason = (
        "Optional BiomedCLIP dependencies (`open_clip_torch`, `transformers`) are installed."
        if available
        else (
            "Optional dependencies `open_clip_torch` / `transformers` not installed in core CPU environment. "
            "Install via `pip install open_clip_torch transformers` to enable BiomedCLIP zero-shot experiments."
        )
    )
    return {
        "available": available,
        "open_clip_installed": has_open_clip,
        "transformers_installed": has_transformers,
        "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "license": "MIT",
        "reason": reason,
    }


class BiomedCLIPExperiment:
    """Zero-shot and few-shot retinal embedding evaluator using optional BiomedCLIP."""

    def __init__(
        self,
        model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        prompts: dict[int, str] | None = None,
    ) -> None:
        status = check_biomedclip_availability()
        if not status["available"]:
            raise RuntimeError(status["reason"])

        import open_clip  # type: ignore

        self.model_id = model_id
        self.prompts = prompts or DEFAULT_DR_CLINICAL_PROMPTS
        try:
            self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
            self.tokenizer = open_clip.get_tokenizer(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"BiomedCLIP model '{model_id}' could not be loaded locally. "
                "Verify cached weights or internet connectivity. Core RetinaX functionality remains unaffected."
            ) from exc

        self.model.eval()

    def encode_images(self, pil_images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in pil_images], dim=0)
        with torch.inference_mode():
            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def encode_prompts(self) -> np.ndarray:
        texts = [self.prompts[c] for c in sorted(self.prompts.keys())]
        tokens = self.tokenizer(texts)
        with torch.inference_mode():
            text_feats = self.model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats.cpu().numpy()

    def zero_shot_predict(self, pil_images: list[Image.Image], temperature: float = 0.07) -> dict[str, Any]:
        """Compute zero-shot image-text similarity logits and softmax probabilities."""
        img_feats = self.encode_images(pil_images)
        txt_feats = self.encode_prompts()
        logits = (img_feats @ txt_feats.T) / max(temperature, 1e-4)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": np.argmax(probs, axis=1),
            "image_embeddings": img_feats,
            "prompts": self.prompts,
        }

    def few_shot_prototype_predict(
        self,
        support_images: list[Image.Image],
        support_labels: np.ndarray,
        query_images: list[Image.Image],
        num_classes: int = 5,
    ) -> dict[str, Any]:
        """Classify query fundus images using L2-normalized class prototypes from support set."""
        sup_embs = self.encode_images(support_images)
        qry_embs = self.encode_images(query_images)
        sup_y = np.asarray(support_labels, dtype=np.int64)

        prototypes = np.zeros((num_classes, sup_embs.shape[1]), dtype=np.float32)
        for c in range(num_classes):
            mask = sup_y == c
            if mask.any():
                proto = sup_embs[mask].mean(axis=0)
                prototypes[c] = proto / max(float(np.linalg.norm(proto)), 1e-8)

        sims = qry_embs @ prototypes.T
        logits = sims * 15.0
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return {"logits": logits, "probabilities": probs, "predictions": np.argmax(probs, axis=1)}
