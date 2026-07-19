
#Feature extraction: builds x_u for every UserNode.
#x_u = concat(metadata_features, roberta_text_embedding)


from __future__ import annotations

import numpy as np
from src.pipeline.schema import UserNode

METADATA_FIELDS = ["followers_count", "following_count", "listed_count"]


def extract_metadata_features(node: UserNode) -> np.ndarray:
    vals = []
    for field_name in METADATA_FIELDS:
        raw = node.metadata.get(field_name, 0)
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = 0.0
        vals.append(np.log1p(max(raw, 0.0)))
    vals.append(1.0 if node.metadata.get("verified") else 0.0)
    return np.array(vals, dtype=np.float32)


class RobertaTextEncoder:
    

    def __init__(self, model_name: str = "roberta-base", device: str = "cpu", max_length: int = 128):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from transformers import RobertaModel, RobertaTokenizer
        import torch

        self._tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self._model = RobertaModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        #Returns (N, hidden_size) array of pooled embeddings for a batch of texts.
        self._lazy_load()
        torch = self._torch
        embeddings = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch = [t if t.strip() else "[empty]" for t in batch]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                out = self._model(**enc)
                # CLS-token pooling (first token of last_hidden_state)
                pooled = out.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(pooled)
        return np.concatenate(embeddings, axis=0)


def build_node_features(
    nodes: dict[str, UserNode],
    text_encoder: "RobertaTextEncoder | None" = None,
) -> None:
    
    ids = list(nodes.keys())
    metadata_feats = np.stack([extract_metadata_features(nodes[i]) for i in ids])

    if text_encoder is not None:
        texts = [nodes[i].text_blob for i in ids]
        text_feats = text_encoder.encode(texts)
        combined = np.concatenate([metadata_feats, text_feats], axis=1)
    else:
        combined = metadata_feats

    for idx, uid in enumerate(ids):
        nodes[uid].x = combined[idx].tolist()
