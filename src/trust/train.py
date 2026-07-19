
from __future__ import annotations

import torch
import torch.nn as nn

from src.pipeline.schema import UnifiedGraph
from src.trust.build_trust_graph import build_trust_inputs
from src.trust.model import EvAGNNTrustModel


def get_labeled_indices(graph: UnifiedGraph, node_ids: list[str], split_name: str) -> list[int]:
    idx = []
    for i, uid in enumerate(node_ids):
        node = graph.nodes[uid]
        if node.label is None:
            continue
        if graph.split.get(uid) == split_name:
            idx.append(i)
    return idx


def train_trust_model(
    graph: UnifiedGraph,
    epochs: int = 100,
    lr: float = 0.01,
    hidden_dim: int = 32,
    use_trust: bool = True,
    verbose: bool = True,
    seed: int = 0,
) -> tuple[EvAGNNTrustModel, dict]:
    torch.manual_seed(seed)

    inputs = build_trust_inputs(graph)
    node_ids = inputs["node_ids"]

    train_idx = get_labeled_indices(graph, node_ids, "train")
    val_idx = get_labeled_indices(graph, node_ids, "val")
    test_idx = get_labeled_indices(graph, node_ids, "test")

    if not train_idx:
        raise ValueError("No labeled training nodes found — check graph.split and node.label.")

    labels = torch.tensor(
        [graph.nodes[uid].label if graph.nodes[uid].label is not None else 0 for uid in node_ids],
        dtype=torch.float32,
    )

    model = EvAGNNTrustModel(in_dim=inputs["h0"].shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long) if val_idx else None
    test_idx_t = torch.tensor(test_idx, dtype=torch.long) if test_idx else None

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, tau = model(
            inputs["h0"], inputs["edge_index"], inputs["edge_features"],
            inputs["is_celebrity_target"], use_trust=use_trust,
        )
        loss = loss_fn(logits[train_idx_t], labels[train_idx_t])
        loss.backward()
        optimizer.step()
        history["train_loss"].append(loss.item())

        if val_idx_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(
                    inputs["h0"], inputs["edge_index"], inputs["edge_features"],
                    inputs["is_celebrity_target"], use_trust=use_trust,
                )
                val_preds = (torch.sigmoid(val_logits[val_idx_t]) > 0.5).float()
                val_acc = (val_preds == labels[val_idx_t]).float().mean().item()
                history["val_acc"].append(val_acc)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            val_str = f", val_acc={history['val_acc'][-1]:.3f}" if val_idx_t is not None else ""
            print(f"epoch {epoch:3d} | loss={loss.item():.4f}{val_str}")

    test_acc = None
    if test_idx_t is not None:
        model.eval()
        with torch.no_grad():
            test_logits, final_tau = model(
                inputs["h0"], inputs["edge_index"], inputs["edge_features"],
                inputs["is_celebrity_target"], use_trust=use_trust,
            )
            test_preds = (torch.sigmoid(test_logits[test_idx_t]) > 0.5).float()
            test_acc = (test_preds == labels[test_idx_t]).float().mean().item()

    if verbose and test_acc is not None:
        print(f"\nFinal test accuracy: {test_acc:.3f} (n={len(test_idx)} labeled test nodes)")

    history["test_acc"] = test_acc
    history["inputs"] = inputs
    return model, history
