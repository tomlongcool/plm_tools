# ESM Contact Graph Tool

这个工具用于从蛋白质序列批量生成：

- residue embeddings: 每个残基一个 ESM 向量，保存为 `node_x`
- contact-like map: ESM 基于 attention/contact head 预测的 `[L, L]` 接触倾向矩阵
- contact-like graph: 从 contact map 过滤出的图边，保存为 `edge_index`、`edge_weight`

推荐保存格式是 `.pt`，因为它最适合后续接 PyTorch、DGL、PyG 或 ProCeSa 风格的 GNN。

## 安装依赖

Colab 中通常已经有 PyTorch，可以直接安装：

```bash
pip install fair-esm pandas openpyxl tqdm
```

本地环境建议先安装与你 CUDA 版本匹配的 PyTorch，然后安装：

```bash
pip install fair-esm pandas openpyxl tqdm
```

依赖说明：

- `torch`: 模型推理和 `.pt` 保存
- `fair-esm`: ESM/ESM-2 模型
- `pandas`: 读取 CSV、TSV、Excel
- `openpyxl`: 读取 `.xlsx`
- `tqdm`: 进度条，可选

## FASTA 输入

```bash
python tools/esm_contact_graph_tool.py proteins.fasta \
  --output-dir outputs/esm_graphs \
  --model esm2_t12_35M_UR50D \
  --edge-policy threshold \
  --threshold 0.2 \
  --min-sep 6 \
  --save-format pt
```

## CSV 输入

假设 CSV 里有 `protein_id` 和 `sequence` 两列：

```bash
python tools/esm_contact_graph_tool.py proteins.csv \
  --output-dir outputs/esm_graphs \
  --input-format csv \
  --id-col protein_id \
  --seq-col sequence \
  --model esm2_t12_35M_UR50D \
  --save-format pt
```

## Excel 输入

```bash
python tools/esm_contact_graph_tool.py proteins.xlsx \
  --output-dir outputs/esm_graphs \
  --input-format excel \
  --sheet-name 0 \
  --id-col protein_id \
  --seq-col sequence
```

## 输出文件结构

每条序列会保存一个文件，例如：

```text
outputs/esm_graphs/
  proteinA.pt
  proteinB.pt
  manifest.csv
```

`.pt` 文件中是一个 Python dict：

```python
{
    "id": str,
    "sequence": str,
    "length": int,
    "model_name": str,
    "repr_layer": int,
    "node_x": torch.Tensor,          # [L, hidden_dim]
    "contact_map": torch.Tensor,     # [L, L], unless --no-contact-map
    "edge_index": torch.LongTensor,  # [2, num_edges]
    "edge_weight": torch.Tensor,     # [num_edges], normalized contact weights
    "edge_raw_score": torch.Tensor,  # [num_edges], raw contact scores
    "graph_config": dict,
}
```

读取示例：

```python
import torch

artifact = torch.load("outputs/esm_graphs/proteinA.pt", map_location="cpu")

node_x = artifact["node_x"]
edge_index = artifact["edge_index"]
edge_weight = artifact["edge_weight"]
contact_map = artifact.get("contact_map")

print(node_x.shape)
print(edge_index.shape)
print(edge_weight.shape)
```

如果要构建 DGL 图：

```python
import dgl

src, dst = edge_index
graph = dgl.graph((src, dst), num_nodes=node_x.shape[0])
graph.ndata["x"] = node_x.float()
graph.edata["x"] = edge_weight.float()
```

## 保存格式选择

推荐：

- `.pt`: 最适合 PyTorch/DGL/GNN，保留 tensor 类型，推荐默认使用。

可选：

- `.npz`: 更适合 NumPy 生态或跨语言读取，metadata 会以 JSON 字符串保存。
- `.pkl`: 可以完整保存 Python 对象，但跨语言兼容性较差。

如果序列很长，`contact_map` 是 `[L, L]`，文件会变大。只想保存稀疏图边时可以加：

```bash
--no-contact-map
```

## 边过滤策略

`--edge-policy threshold`:

保留 `contact_map[i, j] >= threshold` 的边。适合做可控稀疏图。

```bash
--edge-policy threshold --threshold 0.2
```

`--edge-policy topk`:

每条序列保留大约 `top_k_factor * L` 条最高分边，之后默认镜像成双向边。

```bash
--edge-policy topk --top-k-factor 1.0
```

`--edge-policy all`:

保留所有候选残基对，通常会非常密集，不建议长序列使用。

## 常用参数

```bash
--trunc-len 800
```

把长序列截断到 800 aa，和 ProCeSa 常见设置一致。

```bash
--min-sep 6
```

过滤序列上距离太近的残基对，只保留 `|i-j| >= 6` 的边，常用于观察 long-range contacts。

```bash
--float-dtype float16
```

用半精度保存 `node_x` 和 `contact_map`，可以减小文件体积。

```bash
--device cuda
```

强制使用 GPU。默认 `auto` 会在可用时自动使用 CUDA。

## 作为 Python 函数使用

```python
from tools.esm_contact_graph_tool import (
    GraphBuildConfig,
    extract_contact_graphs,
    load_sequence_records,
)

records = load_sequence_records(
    "proteins.csv",
    input_format="csv",
    id_col="protein_id",
    seq_col="sequence",
)

graph_config = GraphBuildConfig(
    edge_policy="threshold",
    threshold=0.2,
    min_sep=6,
)

manifest = extract_contact_graphs(
    records=records,
    output_dir="outputs/esm_graphs",
    model_name="esm2_t12_35M_UR50D",
    graph_config=graph_config,
    save_format="pt",
)
```

## 和 ProCeSa 的关系

ProCeSa 的图大致是：

- 节点：氨基酸残基
- 节点特征：ESM residue embedding
- 边：contact-like map 中选出的残基对
- 边权：经过 `D^{-1/2} A D^{-1/2}` 归一化的 contact score

这个工具保存的 `node_x`、`edge_index`、`edge_weight` 可以直接转换成 DGL 图；如果需要完全复刻 ProCeSa 的 dense 构图，可以使用：

```bash
--edge-policy all --min-sep 0
```

但长序列时这个图会非常密。实际建模中更推荐 `threshold` 或 `topk`。
