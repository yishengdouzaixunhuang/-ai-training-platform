# -*- coding: utf-8 -*-
"""无头运行算子流。

用法:
  python -m flow.run --flow demo_flow.json --input D:/xxx/1.bmp
  python -m flow.run --build cls --input D:/xxx/1.bmp --project_dir D:/proj --out results.jsonl
  python -m flow.run --build seg --project_dir D:/proj --image_dir D:/proj/images --out outputs

--build 支持内置流程模板：cls / seg / det / ocr / ocv。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .frame import Frame
from .nodes import basic, classification, detection, segmentation, ocr, ocv, output  # noqa: F401  注册全部内置节点
from .registry import create_node, list_nodes
from .runner import Flow, Runner
from .batch import BatchRunner, build_cls_flow, build_seg_flow, build_det_flow, build_ocr_flow, build_ocv_flow


def build_flow(kind: str, args) -> Flow:
    if kind == "cls":
        return build_cls_flow(args.project_dir, args.model, top_k=getattr(args, "top_k", 1), out_path=args.out or "")
    if kind == "seg":
        return build_seg_flow(args.project_dir, args.model, out_dir=args.out or "outputs",
                              backend=args.backend, tiled=not args.no_tiled, scale=args.scale)
    if kind == "det":
        return build_det_flow(args.project_dir, args.model, out_dir=args.out or "outputs",
                              conf=args.conf, iou=args.iou, classes=getattr(args, "classes", ""))
    if kind == "ocr":
        return build_ocr_flow(out_dir=args.out or "outputs/ocr")
    if kind == "ocv":
        return build_ocv_flow(model_path=args.model, threshold=args.threshold, out_path=args.out or "outputs/ocv_results.jsonl")
    raise ValueError(f"未知模板: {kind}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="算子流无头运行器")
    ap.add_argument("--flow", default="", help="流程 JSON 文件路径")
    ap.add_argument("--build", default="", help="内置流程模板: cls/seg/det/ocr/ocv")
    ap.add_argument("--input", default="", help="输入图像路径(单张)")
    ap.add_argument("--image_dir", default="", help="输入图像目录(批量)")
    ap.add_argument("--project_dir", default="", help="项目目录(推理模型/结果相对路径基准)")
    ap.add_argument("--model", default="", help="模型文件(默认按模板)")
    ap.add_argument("--out", default="", help="输出目录或文件")
    ap.add_argument("--top_k", type=int, default=1, help="分类 Top-K")
    ap.add_argument("--backend", default="pytorch", help="分割推理后端")
    ap.add_argument("--no_tiled", action="store_true", help="分割不分块推理")
    ap.add_argument("--scale", type=float, default=1.0, help="分割缩放比例")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度")
    ap.add_argument("--iou", type=float, default=0.45, help="检测 IoU")
    ap.add_argument("--threshold", type=float, default=3.0, help="OCV 判定阈值")
    ap.add_argument("--list", action="store_true", help="列出已注册节点")
    args = ap.parse_args(argv)

    if args.list:
        for info in list_nodes():
            print(f"[{info['category']}] {info['type']}: {info['name']}")
        return 0

    if args.build:
        if not args.project_dir and args.build != "ocr":
            ap.error("--build 需要 --project_dir")
        flow = build_flow(args.build, args)
    elif args.flow:
        flow = Flow.load(args.flow)
    else:
        ap.error("必须提供 --flow 或 --build")
        return 2

    if not args.input and not args.image_dir:
        ap.error("必须提供 --input 图像路径或 --image_dir 目录")
        return 2

    meta = {"project_dir": args.project_dir} if args.project_dir else {}

    if args.image_dir:
        from .batch import BatchRunner
        images = sorted(
            os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif"))
        )
        if not images:
            print("image_dir 下无图片")
            return 1
        br = BatchRunner(flow, project_dir=args.project_dir)
        ok = err = 0
        for i, (path, frame, e) in enumerate(br.run(images), 1):
            if e:
                err += 1
                print(f"[{i}/{len(images)}] {os.path.basename(path)}: FAILED - {e}")
            else:
                ok += 1
                print(f"[{i}/{len(images)}] {os.path.basename(path)}: {frame.result}")
        print(f"批量完成: {ok} 成功, {err} 失败, 共 {len(images)}")
        return 0

    in_frame = Frame(path=args.input, meta=meta)
    runner = Runner(flow)
    out_frame = runner.run(in_frame, on_node=lambda n, f: print(f"  -> {n.node_type}: {f.to_summary()}"))
    print(json.dumps(_safe(out_frame.result), ensure_ascii=False, indent=2))
    return 0


def _safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    sys.exit(main())
