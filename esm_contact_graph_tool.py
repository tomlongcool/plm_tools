#!/usr/bin/env python3
"""Generate ESM residue embeddings and contact-like graphs.

This module can be used as a command line tool or imported from Python code.
It supports FASTA, CSV/TSV, and Excel inputs, and saves per-sequence graph
artifacts as .pt, .npz, or .pkl files.

Recommended output for PyTorch/DGL workflows is .pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


def require_numpy():
    if np is None:
        raise ImportError("Missing dependency 'numpy'. Install with: pip install numpy")


def require_torch():
    if torch is None:
        raise ImportError(
            "Missing dependency 'torch'. Install PyTorch from https://pytorch.org/get-started/locally/"
        )


@dataclass(frozen=True)
class SequenceRecord:
    """A single protein sequence record."""

    record_id: str
    sequence: str


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for converting contact maps into graph edges."""

    edge_policy: str = "threshold"
    threshold: float = 0.2
    top_k_factor: float = 1.0
    min_sep: int = 0
    bidirectional: bool = True
    include_self_loops: bool = False


def clean_sequence(sequence: object, invalid_policy: str = "replace") -> str | None:
    """Clean a protein sequence.

    Args:
        sequence: Raw sequence value.
        invalid_policy: One of "replace", "error", or "skip".

    Returns:
        Cleaned uppercase sequence, or None if skipped.
    """

    seq = re.sub(r"\s+", "", str(sequence).upper())
    if not seq or seq == "NAN":
        return None

    invalid = sorted(set(seq) - AA_ALPHABET)
    if not invalid:
        return seq

    if invalid_policy == "replace":
        return "".join(aa if aa in AA_ALPHABET else "X" for aa in seq)
    if invalid_policy == "skip":
        return None
    raise ValueError(f"Invalid amino acid character(s) {invalid} in sequence: {seq[:50]}...")


def safe_record_id(value: object, fallback: str) -> str:
    """Convert a user-provided id into a safe filename stem."""

    raw = str(value).strip() if value is not None else ""
    if not raw or raw.lower() == "nan":
        raw = fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return safe.strip("._") or fallback


def parse_fasta(path: str | Path, invalid_policy: str = "replace") -> list[SequenceRecord]:
    """Parse a FASTA file without requiring Biopython."""

    records: list[SequenceRecord] = []
    current_id: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal current_id, chunks
        if current_id is None:
            return
        seq = clean_sequence("".join(chunks), invalid_policy=invalid_policy)
        if seq is not None:
            records.append(SequenceRecord(safe_record_id(current_id, f"seq_{len(records) + 1:06d}"), seq))
        current_id = None
        chunks = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].strip().split()[0] or f"seq_{len(records) + 1:06d}"
            else:
                chunks.append(line)
        flush()

    return records


def _read_table(path: str | Path, input_format: str, sheet_name: str | int) -> "object":
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'pandas'. Install with: pip install pandas openpyxl"
        ) from exc

    if input_format == "csv":
        return pd.read_csv(path)
    if input_format == "tsv":
        return pd.read_csv(path, sep="\t")
    if input_format == "excel":
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported table format: {input_format}")


def detect_input_format(path: str | Path, input_format: str = "auto") -> str:
    """Detect input format from filename suffix."""

    if input_format != "auto":
        return input_format

    suffix = Path(path).suffix.lower()
    if suffix in {".fa", ".faa", ".fasta", ".fna"}:
        return "fasta"
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".tab"}:
        return "tsv"
    if suffix in {".xls", ".xlsx"}:
        return "excel"
    raise ValueError(
        "Could not infer input format. Use --input-format fasta|csv|tsv|excel."
    )


def load_sequence_records(
    input_path: str | Path,
    input_format: str = "auto",
    seq_col: str = "sequence",
    id_col: str | None = None,
    sheet_name: str | int = 0,
    invalid_policy: str = "replace",
    trunc_len: int | None = None,
    limit: int | None = None,
) -> list[SequenceRecord]:
    """Load protein sequences from FASTA, CSV/TSV, or Excel."""

    fmt = detect_input_format(input_path, input_format)

    if fmt == "fasta":
        records = parse_fasta(input_path, invalid_policy=invalid_policy)
    else:
        table = _read_table(input_path, fmt, sheet_name)
        if seq_col not in table.columns:
            raise ValueError(f"Sequence column {seq_col!r} not found. Columns: {list(table.columns)}")
        if id_col is not None and id_col not in table.columns:
            raise ValueError(f"ID column {id_col!r} not found. Columns: {list(table.columns)}")

        records = []
        for row_idx, row in table.iterrows():
            fallback = f"seq_{len(records) + 1:06d}"
            rec_id = safe_record_id(row[id_col], fallback) if id_col else fallback
            seq = clean_sequence(row[seq_col], invalid_policy=invalid_policy)
            if seq is None:
                continue
            records.append(SequenceRecord(rec_id, seq))

    if trunc_len is not None and trunc_len > 0:
        records = [SequenceRecord(r.record_id, r.sequence[:trunc_len]) for r in records]
    if limit is not None and limit > 0:
        records = records[:limit]
    return records


def load_esm_model(model_name: str, device: str = "auto"):
    """Load a fair-esm model by pretrained loader name."""

    require_torch()
    try:
        import esm
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'fair-esm'. Install with: pip install fair-esm"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(esm.pretrained, model_name):
        available = sorted(name for name in dir(esm.pretrained) if name.startswith("esm"))
        raise ValueError(
            f"Unknown ESM model loader {model_name!r}. Examples: "
            f"esm2_t12_35M_UR50D, esm2_t33_650M_UR50D. "
            f"Available loaders include: {available[:20]}"
        )

    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval().to(device)
    return model, alphabet, torch.device(device)


def iter_token_batches(
    records: Sequence[SequenceRecord],
    toks_per_batch: int,
) -> Iterator[list[SequenceRecord]]:
    """Yield batches constrained approximately by total tokens."""

    batch: list[SequenceRecord] = []
    token_count = 0
    for record in records:
        record_tokens = len(record.sequence) + 2
        if batch and token_count + record_tokens > toks_per_batch:
            yield batch
            batch = []
            token_count = 0
        batch.append(record)
        token_count += record_tokens
    if batch:
        yield batch


def normalize_adj(matrix: np.ndarray) -> np.ndarray:
    """GCN-style symmetric adjacency normalization."""

    require_numpy()
    rowsum = np.asarray(matrix.sum(axis=1), dtype=np.float64)
    with np.errstate(divide="ignore"):
        r_inv = np.power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0
    return np.diag(r_inv) @ matrix @ np.diag(r_inv)


def build_contact_edges(
    contact_map: np.ndarray,
    config: GraphBuildConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a contact-like matrix into edge_index and edge weights.

    Returns:
        edge_index: int64 array with shape [2, num_edges].
        edge_weight: normalized edge weights, float32 array with shape [num_edges].
        edge_raw_score: raw contact scores, float32 array with shape [num_edges].
    """

    require_numpy()
    if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
        raise ValueError(f"contact_map must be square, got {contact_map.shape}")

    length = contact_map.shape[0]
    if length == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    start_diag = max(1, config.min_sep)
    candidate_mask = np.triu(np.ones((length, length), dtype=bool), k=start_diag)
    pairs = np.argwhere(candidate_mask)

    if config.edge_policy == "threshold":
        keep = contact_map[pairs[:, 0], pairs[:, 1]] >= config.threshold
        pairs = pairs[keep]
    elif config.edge_policy == "topk":
        scores = contact_map[pairs[:, 0], pairs[:, 1]]
        k = max(1, int(math.ceil(config.top_k_factor * length)))
        selected = np.argsort(scores)[::-1][: min(k, len(scores))]
        pairs = pairs[selected]
    elif config.edge_policy == "all":
        pass
    else:
        raise ValueError(f"Unsupported edge_policy: {config.edge_policy}")

    if config.include_self_loops:
        self_pairs = np.column_stack([np.arange(length), np.arange(length)])
        pairs = np.vstack([self_pairs, pairs]) if len(pairs) else self_pairs

    if config.bidirectional and len(pairs):
        reverse = pairs[pairs[:, 0] != pairs[:, 1]][:, ::-1]
        pairs = np.vstack([pairs, reverse]) if len(reverse) else pairs

    if len(pairs) == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    normalized = normalize_adj(contact_map)
    src = pairs[:, 0].astype(np.int64)
    dst = pairs[:, 1].astype(np.int64)
    edge_index = np.stack([src, dst], axis=0)
    edge_weight = normalized[src, dst].astype(np.float32)
    edge_raw_score = contact_map[src, dst].astype(np.float32)
    return edge_index, edge_weight, edge_raw_score


def _to_output_dtype(tensor: torch.Tensor, float_dtype: str) -> torch.Tensor:
    require_torch()
    if float_dtype == "float16":
        return tensor.to(dtype=torch.float16)
    if float_dtype == "bfloat16":
        return tensor.to(dtype=torch.bfloat16)
    return tensor.to(dtype=torch.float32)


def make_graph_artifact(
    record: SequenceRecord,
    embedding: torch.Tensor,
    contact_map: np.ndarray,
    model_name: str,
    repr_layer: int,
    graph_config: GraphBuildConfig,
    float_dtype: str = "float32",
    save_contact_map: bool = True,
) -> dict[str, object]:
    """Create a serializable graph artifact for one protein."""

    require_numpy()
    require_torch()
    edge_index, edge_weight, edge_raw_score = build_contact_edges(contact_map, graph_config)
    artifact: dict[str, object] = {
        "id": record.record_id,
        "sequence": record.sequence,
        "length": len(record.sequence),
        "model_name": model_name,
        "repr_layer": repr_layer,
        "node_x": _to_output_dtype(embedding.cpu(), float_dtype),
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_weight": torch.from_numpy(edge_weight),
        "edge_raw_score": torch.from_numpy(edge_raw_score),
        "graph_config": graph_config.__dict__.copy(),
    }
    if save_contact_map:
        contact_tensor = torch.from_numpy(contact_map)
        artifact["contact_map"] = _to_output_dtype(contact_tensor, float_dtype)
    return artifact


def save_artifact(artifact: dict[str, object], output_path: Path, save_format: str) -> None:
    """Save one graph artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "pt":
        require_torch()
        torch.save(artifact, output_path)
        return
    if save_format == "pkl":
        with output_path.open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
    if save_format == "npz":
        require_numpy()
        arrays = {}
        metadata = {}
        for key, value in artifact.items():
            if torch is not None and isinstance(value, torch.Tensor):
                tensor = value.cpu()
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                arrays[key] = tensor.numpy()
            elif isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        np.savez_compressed(output_path, **arrays)
        return
    raise ValueError(f"Unsupported save format: {save_format}")


def extract_contact_graphs(
    records: Sequence[SequenceRecord],
    output_dir: str | Path,
    model_name: str = "esm2_t12_35M_UR50D",
    device: str = "auto",
    repr_layer: int = -1,
    toks_per_batch: int = 4096,
    graph_config: GraphBuildConfig | None = None,
    save_format: str = "pt",
    float_dtype: str = "float32",
    save_contact_map: bool = True,
    overwrite: bool = False,
    quiet: bool = False,
) -> list[dict[str, object]]:
    """Extract ESM embeddings/contact maps and save graph artifacts.

    Returns:
        A manifest list describing saved files.
    """

    require_numpy()
    require_torch()
    if graph_config is None:
        graph_config = GraphBuildConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, alphabet, device_obj = load_esm_model(model_name, device=device)
    batch_converter = alphabet.get_batch_converter()
    if repr_layer < 0:
        repr_layer = model.num_layers

    batches = list(iter_token_batches(records, toks_per_batch=toks_per_batch))
    iterator: Iterable[list[SequenceRecord]]
    iterator = tqdm(batches, desc="Extracting ESM graphs") if tqdm and not quiet else batches

    manifest: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in iterator:
            data = [(record.record_id, record.sequence) for record in batch]
            labels, seqs, tokens = batch_converter(data)
            tokens = tokens.to(device_obj)
            out = model(tokens, repr_layers=[repr_layer], return_contacts=True)
            reps = out["representations"][repr_layer].detach().cpu()
            contacts = out["contacts"].detach().cpu().numpy()

            for i, record in enumerate(batch):
                length = len(record.sequence)
                embedding = reps[i, 1 : length + 1].contiguous()
                contact_map = contacts[i, :length, :length].astype(np.float32, copy=False)
                artifact = make_graph_artifact(
                    record=record,
                    embedding=embedding,
                    contact_map=contact_map,
                    model_name=model_name,
                    repr_layer=repr_layer,
                    graph_config=graph_config,
                    float_dtype=float_dtype,
                    save_contact_map=save_contact_map,
                )

                output_path = output_dir / f"{record.record_id}.{save_format}"
                if output_path.exists() and not overwrite:
                    raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
                save_artifact(artifact, output_path, save_format=save_format)

                manifest.append(
                    {
                        "id": record.record_id,
                        "sequence_length": length,
                        "num_edges": int(artifact["edge_index"].shape[1]),
                        "node_feature_dim": int(artifact["node_x"].shape[1]),
                        "path": str(output_path),
                    }
                )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "sequence_length", "num_edges", "node_feature_dim", "path"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    return manifest


def _parse_sheet_name(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ESM residue embeddings and contact-like graphs from FASTA/CSV/Excel."
    )
    parser.add_argument("input", type=str, help="Input FASTA, CSV, TSV, XLS, or XLSX file.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "fasta", "csv", "tsv", "excel"],
        default="auto",
        help="Input format. Default: infer from extension.",
    )
    parser.add_argument("--seq-col", default="sequence", help="Sequence column for CSV/Excel inputs.")
    parser.add_argument("--id-col", default=None, help="Optional record id column for CSV/Excel inputs.")
    parser.add_argument("--sheet-name", default="0", help="Excel sheet name or index. Default: 0.")
    parser.add_argument(
        "--invalid-policy",
        choices=["replace", "error", "skip"],
        default="replace",
        help="How to handle invalid amino acid characters. Default: replace with X.",
    )
    parser.add_argument("--trunc-len", type=int, default=0, help="Truncate sequences to this length if > 0.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N records if > 0.")

    parser.add_argument("--model", default="esm2_t12_35M_UR50D", help="fair-esm pretrained loader name.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--repr-layer", type=int, default=-1, help="Representation layer. Default: final layer.")
    parser.add_argument("--toks-per-batch", type=int, default=4096, help="Approximate max tokens per batch.")

    parser.add_argument(
        "--edge-policy",
        choices=["threshold", "topk", "all"],
        default="threshold",
        help="How to choose graph edges from the contact map.",
    )
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for --edge-policy threshold.")
    parser.add_argument(
        "--top-k-factor",
        type=float,
        default=1.0,
        help="For --edge-policy topk, keep ceil(top_k_factor * sequence_length) edges before mirroring.",
    )
    parser.add_argument("--min-sep", type=int, default=0, help="Ignore edges with |i-j| < min_sep.")
    parser.add_argument("--directed", action="store_true", help="Do not mirror selected edges.")
    parser.add_argument("--include-self-loops", action="store_true", help="Add i->i edges.")

    parser.add_argument("--save-format", choices=["pt", "npz", "pkl"], default="pt")
    parser.add_argument(
        "--float-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Storage dtype for node_x and contact_map.",
    )
    parser.add_argument(
        "--no-contact-map",
        action="store_true",
        help="Do not save the dense [L,L] contact_map; save only graph edges.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    records = load_sequence_records(
        input_path=args.input,
        input_format=args.input_format,
        seq_col=args.seq_col,
        id_col=args.id_col,
        sheet_name=_parse_sheet_name(args.sheet_name),
        invalid_policy=args.invalid_policy,
        trunc_len=args.trunc_len if args.trunc_len > 0 else None,
        limit=args.limit if args.limit > 0 else None,
    )
    if not records:
        raise RuntimeError("No valid sequences were loaded from input.")

    graph_config = GraphBuildConfig(
        edge_policy=args.edge_policy,
        threshold=args.threshold,
        top_k_factor=args.top_k_factor,
        min_sep=args.min_sep,
        bidirectional=not args.directed,
        include_self_loops=args.include_self_loops,
    )

    manifest = extract_contact_graphs(
        records=records,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        repr_layer=args.repr_layer,
        toks_per_batch=args.toks_per_batch,
        graph_config=graph_config,
        save_format=args.save_format,
        float_dtype=args.float_dtype,
        save_contact_map=not args.no_contact_map,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Saved {len(manifest)} graph artifact(s) to {args.output_dir}")
    print(f"Manifest: {Path(args.output_dir) / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
