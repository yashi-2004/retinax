"""Self-supervised learning (SimCLR) and Vision Foundation Models (DINOv2, optional BiomedCLIP)."""

from src.foundation.biomedical_clip import BiomedCLIPExperiment, check_biomedclip_availability
from src.foundation.dinov2 import DINOv2FeatureExtractor, train_embedding_probe
from src.foundation.embeddings import compute_2d_projection, extract_embeddings_from_model
from src.foundation.ssl_simclr import SimCLREncoderModel, SimCLRPairDataset, train_simclr_ssl

__all__ = [
    "BiomedCLIPExperiment",
    "DINOv2FeatureExtractor",
    "SimCLREncoderModel",
    "SimCLRPairDataset",
    "check_biomedclip_availability",
    "compute_2d_projection",
    "extract_embeddings_from_model",
    "train_embedding_probe",
    "train_simclr_ssl",
]
