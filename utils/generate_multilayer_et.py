"""
Generate a multi-layer transformer-like Chakra ET for quantization experiments.

Simulates a single transformer block with 8 sequential all-reduce collectives:
  0  embedding_ar       256 KB   (embedding table gradient sync)
  1  attn_q_proj_ar       1 MB   (Q projection weight gradient)
  2  attn_k_proj_ar       1 MB   (K projection weight gradient)
  3  attn_v_proj_ar       1 MB   (V projection weight gradient)
  4  attn_out_proj_ar     2 MB   (attention output projection)
  5  ffn_fc1_ar           4 MB   (FFN first linear layer)
  6  ffn_fc2_ar           4 MB   (FFN second linear layer)
  7  layernorm_ar       256 KB   (layer-norm parameter sync)

Each node data_dep-chains to the previous so Astra-sim runs them sequentially,
matching the backward-pass ordering of a real training iteration.
"""

import argparse
import os
import sys

# Ensure we can import Chakra modules from the workspace root
sys.path.insert(0, "/workspaces/astra-sim")

from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
    COMM_COLL_NODE,
    ALL_REDUCE,
    AttributeProto as ChakraAttr,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
    encodeMessage as encode_message,
)

# ---------------------------------------------------------------------------
# Layer definitions: (name, comm_size_bytes)
# ---------------------------------------------------------------------------
LAYERS = [
    ("embedding_ar",     256 * 1024),       # 256 KB
    ("attn_q_proj_ar",   1 * 1024 * 1024),  # 1 MB
    ("attn_k_proj_ar",   1 * 1024 * 1024),  # 1 MB
    ("attn_v_proj_ar",   1 * 1024 * 1024),  # 1 MB
    ("attn_out_proj_ar", 2 * 1024 * 1024),  # 2 MB
    ("ffn_fc1_ar",       4 * 1024 * 1024),  # 4 MB
    ("ffn_fc2_ar",       4 * 1024 * 1024),  # 4 MB
    ("layernorm_ar",     256 * 1024),        # 256 KB
]


def generate(npus_count: int, output_dir: str) -> None:
    et_dir = os.path.join(output_dir, "transformer_block_8npus")
    os.makedirs(et_dir, exist_ok=True)

    for npu in range(npus_count):
        et_path = os.path.join(et_dir, f"transformer_block.{npu}.et")
        with open(et_path, "wb") as f:
            encode_message(f, GlobalMetadata(version="0.0.4"))

            prev_id = None
            for node_id, (layer_name, comm_size) in enumerate(LAYERS):
                node = ChakraNode()
                node.id = node_id
                node.name = layer_name
                node.type = COMM_COLL_NODE
                node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
                node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
                if prev_id is not None:
                    node.data_deps.append(prev_id)
                encode_message(f, node)
                prev_id = node_id

        print(f"  Wrote {et_path}")

    print(f"Generated {npus_count} ET files in {et_dir}/")
    print(f"  {len(LAYERS)} layers: " + ", ".join(f"{n}({s//1024}KB)" for n, s in LAYERS))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a multi-layer transformer Chakra ET"
    )
    parser.add_argument("--npus-count", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default="examples/workload/microbenchmarks/",
        help="Directory under which transformer_block_8npus/ will be created",
    )
    args = parser.parse_args()
    generate(args.npus_count, args.output_dir)


if __name__ == "__main__":
    main()
