# -*- coding: utf-8 -*-
"""无头运行算子流。

用法:
  python -m flow.run --flow demo_flow.json --input D:/xxx/1.bmp
  python -m flow.run --build cls --input D:/xxx/1.bmp --project_dir D:/proj --out results.jsonl

--build 参数可快速构造内置流程模板（无需手写 JSON）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .frame import Frame
from .nodes import basic, classification, output  # noqa: F401  注册全部内置节点
from .registry import create_node
from .runner import Flow, Runner


def build_cls_flow(project_dir: str, out_path: str) -> Flow:
    """分类最小流程：文件 -> 缩放0.5 -> 分类推理 -> 写JSON。"""
    src = create_node("file_source", node_id="src")
    resize = create_node("resize", node_id="resize", scale_w=0.5, scale_h=0.5)
    cls = create_node("cls_infer", node_id="cls", project_dir=project_dir)
    out = create_node("result_to_json", node_id="out", out_path=out_path)
    return (Flow(name="分类最小流程")
            .add_node(src).add_node(resize).add_node(cls).add_node(out)
            .add_edge("src", "resize").add_edge("resize", "cls").add_edge("cls", "out"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="算子流无头运行器")
    ap.add_argument("--flow", default="", help="流程 JSON 文件路径")
    ap.add_argument("--build", default="", help="内置流程模板: cls")
    ap.add_argument("--input", default="", help="输入图像路径")
    ap.add_argument("--project_dir", default="", help="项目目录(分类推理用)")
    ap.add_argument("--out", default="results.jsonl", help="输出文件")
    ap.add_argument("--list", action="store_true", help="列出已注册节点")
    args = ap.parse_args(argv)

    if args.list:
        from .registry import list_nodes
        for info in list_nodes():
            print(f"[{info['category']}] {info['type']}: {info['name']}")
        return 0

    if args.build == "cls":
        flow = build_cls_flow(args.project_dir, args.out)
    elif args.flow:
        flow = Flow.load(args.flow)
    else:
        ap.error("必须提供 --flow 或 --build")
        return 2

    if not args.input:
        ap.error("必须提供 --input 图像路径")
        return 2

    # 输入 Frame：file_source 会按 params.path 或 frame.path 读取
    meta = {"project_dir": args.project_dir} if args.project_dir else {}
    in_frame = Frame(path=args.input, meta=meta)
    runner = Runner(flow)
    out_frame = runner.run(in_frame, on_node=lambda n, f: print(f"  -> {n.node_type}: {f.to_summary()}"))
    print(json.dumps(out_frame.result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())