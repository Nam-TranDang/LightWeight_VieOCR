import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ..registry import Get_Yolo11_Model
from ..utils import Vocab
from .Seq2Seq import Seq2Seq, CNN


class Seq2Seq__Independ_VietOCR(torch.nn.Module):
    """
    Thin wrapper that combines CNN backbone + Seq2Seq transformer into a single module.
    This matches the structure used in the existing inference scripts.
    """

    def __init__(self, vocab_size: int, backbone_name: str, cnn_cfg: Dict[str, Any], seq_cfg: Dict[str, Any]):
        super().__init__()
        self.cnn = CNN(
            backbone=backbone_name,
            ss=cnn_cfg["ss"],
            ks=cnn_cfg["ks"],
            hidden=cnn_cfg["hidden"],
        )
        self.transformer = Seq2Seq(
            vocab_size=vocab_size,
            encoder_hidden=seq_cfg["encoder_hidden"],
            decoder_hidden=seq_cfg["decoder_hidden"],
            img_channel=seq_cfg["img_channel"],
            decoder_embedded=seq_cfg["decoder_embedded"],
            dropout=seq_cfg["dropout"],
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.cnn(img)
        features = self.transformer(features)
        return features


def translate(
    img_tensor: torch.Tensor,
    model: Seq2Seq__Independ_VietOCR,
    vocab: Vocab,
    device: torch.device,
    max_seq_length: int = 128,
) -> str:
    """
    Inference helper for Seq2Seq recognition.
    Mirrors the logic from the current inference scripts.
    """
    model.eval()
    with torch.no_grad():
        features = model.cnn(img_tensor)

        if features.dim() == 4:
            # [B, C, H, W] -> [T, B, C]  (time-major sequence)
            features = features.squeeze(2).permute(2, 0, 1)

        encoder_outputs, hidden = model.transformer.encoder(features)

        decoder_input = torch.tensor([vocab.go], device=device)
        decoded_indices = []

        for _ in range(max_seq_length):
            output, hidden, _ = model.transformer.decoder(
                decoder_input,
                hidden,
                encoder_outputs,
            )

            topv, topi = output.data.topk(1)
            token_index = topi.item()

            if token_index == vocab.eos:
                break

            decoded_indices.append(token_index)
            decoder_input = topi.squeeze(1).detach()

    return vocab.decode(decoded_indices)


def process_image_crop(img: Image.Image, height: int, device: torch.device) -> torch.Tensor:
    """
    Prepare a PIL Image crop for the Seq2Seq model.

    Steps (kept identical to your tested pipeline):
    1. Convert RGB
    2. Resize to fixed height, maintain aspect ratio (LANCZOS)
    3. ToTensor (0-1 scale)
    4. NO Normalize (mean/std)
    """
    img = img.convert("RGB")

    # Maintain aspect ratio for the given target height
    w_new = int(img.width * height / img.height)
    img = img.resize((w_new, height), Image.Resampling.LANCZOS)

    img_tensor = ToTensor()(img)  # [0, 255] -> [0.0, 1.0]
    return img_tensor.unsqueeze(0).to(device)


@dataclass
class OCRModels:
    """
    Container for the core OCR components.

    This keeps the MinIO / Mongo / file I/O logic separate from model construction.
    """

    yolo_model: torch.nn.Module
    rec_model: Seq2Seq__Independ_VietOCR
    vocab: Vocab
    cfg: Dict[str, Any]
    device: torch.device


def build_ocr_models_from_cfg(cfg: Dict[str, Any]) -> OCRModels:
    """
    Build YOLO11 detection model + Seq2Seq recognition model from an already-loaded config dict.
    The logic here is intentionally identical to what you currently do in the inference scripts.
    """
    device = torch.device(cfg["device"])

    # --- YOLO ---
    yolo_model = Get_Yolo11_Model(cfg["yolo"]["variant"], cfg["yolo"]["nc"])
    yolo_checkpoint = torch.load(cfg["yolo"]["weights"], map_location="cpu")
    yolo_state = yolo_checkpoint["model"] if "model" in yolo_checkpoint else yolo_checkpoint
    yolo_model.load_state_dict(yolo_state, strict=True)
    yolo_model.to(device)
    yolo_model.eval()

    # --- Seq2Seq ---
    vocab = Vocab(cfg["seq2seq"]["vocab"])

    rec_model = Seq2Seq__Independ_VietOCR(
        vocab_size=len(vocab),
        backbone_name=cfg["seq2seq"]["backbone"],
        cnn_cfg=cfg["seq2seq"]["cnn"],
        seq_cfg=cfg["seq2seq"]["seq_modeling"],
    )
    rec_model.to(device)

    rec_checkpoint = torch.load(cfg["seq2seq"]["weights"], map_location=device)
    if "state_dict" in rec_checkpoint:
        rec_checkpoint = rec_checkpoint["state_dict"]
    rec_model.load_state_dict(rec_checkpoint, strict=True)
    rec_model.eval()

    return OCRModels(
        yolo_model=yolo_model,
        rec_model=rec_model,
        vocab=vocab,
        cfg=cfg,
        device=device,
    )


def build_ocr_models_from_config_path(config_path: str) -> OCRModels:
    """
    Convenience wrapper when you only have a path to the YAML config.
    """
    import yaml

    config_path = str(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return build_ocr_models_from_cfg(cfg)

