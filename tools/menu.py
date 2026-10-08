# -*- coding: utf-8 -*-
"""孚能 三视图语义分割 工具菜单

为什么菜单用 Python 而不是 .bat：
  .bat 里的中文必须存成 GBK 才能在 cmd 下正常显示，改一次就要转一次编码，
  极易踩坑（本工程已踩过一次）。这里让 .bat 保持纯 ASCII 只负责启动，
  所有中文界面放在 Python 里，UTF-8 原生，永不乱码。

启动：双击同目录的「孚能三视图工具.bat」
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <app>/tools
APP = HERE.parent                                # <app>
DEFAULT_SRC = r"\\192.168.1.10\04-图片\20260910\233-训练集导出"


def _datasets_dir() -> Path:
    for c in (Path(r"D:\01_Arthur\孚能\datasets"), APP.parent / "datasets", Path.home() / "fnt_datasets"):
        if c.exists():
            return c
    c = APP.parent / "datasets"
    c.mkdir(parents=True, exist_ok=True)
    return c


DS = _datasets_dir()


def clear():
    os.system("cls")


def hr(ch="=", n=68):
    print(ch * n)


def pick_dir(title: str, initial: str = "") -> str:
    ps = HERE / "pickdir.ps1"
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps), title]
    if initial:
        cmd.append(initial)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return (r.stdout or "").strip()
    except Exception as e:
        print(f"  [提示] 目录选择框不可用（{type(e).__name__}），请手动输入路径")
        return input("  路径: ").strip().strip('"')


def ask(prompt: str, default: str = "") -> str:
    s = input(f"  {prompt}" + (f"（回车={default}）: " if default else ": ")).strip()
    return s or default


def run(args, cwd=APP):
    print()
    hr("-")
    try:
        code = subprocess.call([sys.executable, "-m"] + args, cwd=str(cwd))
    except KeyboardInterrupt:
        print("\n  已取消")
        return 1
    hr("-")
    print(f"  退出码: {code}")
    return code


def list_ckpts(proj: Path):
    md = proj / "models"
    if not md.exists():
        return []
    return sorted(p.name for p in md.glob("*.pth"))


def choose_project() -> Path | None:
    p = pick_dir("选择工程目录（里面应有 project.json）", str(DS))
    if not p:
        print("  未选择")
        return None
    proj = Path(p)
    if not (proj / "project.json").exists():
        print(f"  [错误] 不是有效工程（缺 project.json）：{proj}")
        return None
    return proj


def show_project_info(proj: Path):
    try:
        d = json.loads((proj / "project.json").read_text(encoding="utf-8"))
        cn = d.get("class_cn", {})
        cls = " / ".join(f"{c}({cn.get(c, c)})" for c in d.get("classes", []))
        print(f"  工程: {proj.name}   切图: {d.get('tile_size')}   通道: {d.get('in_channels')}")
        print(f"  类别: {cls}")
    except Exception as e:
        print(f"  [提示] 读取工程信息失败: {e}")


# ─────────────────────── 功能 ───────────────────────
def do_import():
    clear()
    hr()
    print("  [1] 导入 LabelMe 三视图导出 → 可训练工程")
    hr()
    print("  弹出窗口请选择【导出根目录】，即同时含 原图_xxxx 与 Label_xxxx 两个子目录的那一层。")
    print(f"  默认: {DEFAULT_SRC}\n")
    src = pick_dir("选择 LabelMe 导出根目录", DEFAULT_SRC) or DEFAULT_SRC
    if not Path(src).exists():
        print(f"  [错误] 目录不存在: {src}")
        return
    print(f"  源目录: {src}\n")
    print("  切图尺寸决定能看多小的缺陷：128 保留最多细节（推荐小缺陷），512 上下文最多。")
    tile = ask("切图尺寸 [128 / 256 / 512]", "256")
    name = ask("输出工程名", f"farasis_tile{tile}")
    neg = ask("负样本比例（相对正样本）", "1.5")
    dst = DS / name
    print(f"\n  输出工程: {dst}")
    print("  开始导入（648 组约需 4 分钟，取决于网络）...\n")
    run(["tools.import_farasis_labelme", "--src", src, "--dst", str(dst),
         "--tile", tile, "--neg-ratio", neg, "--val-ratio", "0.2", "--preview", "24"])
    print(f"\n  人工抽检图: {dst / 'preview'}")


def do_train():
    clear()
    hr()
    print("  [2] 训练三图分割模型")
    hr()
    proj = choose_project()
    if not proj:
        return
    show_project_info(proj)
    print()
    print("  模型说明：")
    print("    unet       ★推荐。全分辨率跳连，专治小缺陷。CPU 上 128 图约 5 分钟/轮")
    print("    lraspp     mobilenet 骨干，最轻量。256 图约 8 分钟/轮")
    print("    fcn        带跳连，但 CPU 上 256 图要 75 分钟/轮 —— 需 GPU")
    print("    deeplabv3  resnet50，效果上限最高，但 CPU 上 512 图要 5.7 小时/轮 —— 需 GPU")
    print()
    model = ask("模型 [unet / lraspp / fcn / deeplabv3]", "unet")
    extra = []
    if model == "unet":
        base = ask("U-Net 首层通道数", "16")
        depth = ask("U-Net 下采样层数", "4")
        extra = ["--base", base, "--depth", depth]
    ep = ask("训练轮数", "20")
    bs = ask("批大小", "8")
    wrk = ask("数据加载进程数（CPU 训练填 0）", "0")
    th = ask("CPU 线程数（实测 6 最快，0=自动）", "6")
    name = ask("模型名", f"{model}{proj.name[-3:]}_{ep}ep")
    cks = list_ckpts(proj)
    resume = ""
    if cks:
        print(f"\n  已有模型（可选其一继续训练，留空=从头训练）：")
        for i, c in enumerate(cks):
            print(f"    [{i}] {c}")
        sel = ask("续训基座序号或文件名（回车=从头训练）", "")
        if sel:
            resume = cks[int(sel)] if (sel.isdigit() and int(sel) < len(cks)) else sel
    print(f"\n  开始训练 → models/{name}.pth" + (f"（续训自 {resume}）" if resume else ""))
    print("  训练中每轮会落盘 *_last.pth，中断后可再用续训接着跑。\n")
    run(["tools.train_farasis", "--project", str(proj), "--model", model, "--epochs", ep,
         "--batch", bs, "--workers", wrk, "--threads", th, "--name", name]
        + (["--resume", resume] if resume else []) + extra)
    print(f"\n  模型与报告: {proj / 'models'}")


def _choose_ckpt(proj: Path) -> str | None:
    cks = list_ckpts(proj)
    if not cks:
        print(f"  [错误] {proj / 'models'} 下没有 .pth 模型")
        return None
    print("  可用模型：")
    for i, c in enumerate(cks):
        print(f"    [{i}] {c}")
    sel = ask("序号或文件名", cks[-1])
    if sel.isdigit() and int(sel) < len(cks):
        return cks[int(sel)]
    return sel


def do_eval_tile():
    clear()
    hr()
    print("  [3] 评估：漏检 / 过杀 工作曲线（tile 级）")
    hr()
    print("  扫描 最小面积 × 概率阈值，并按缺陷尺寸分层给出漏检率。\n")
    proj = choose_project()
    if not proj:
        return
    ck = _choose_ckpt(proj)
    if not ck:
        return
    sp = ask("评估切分 [val / train / all]", "val")
    tta = ask("开启 TTA（更稳，耗时翻倍）[y/n]", "y").lower().startswith("y")
    med = ask("概率图中值滤波核 [0/3/5]（治碎点，推荐 3）", "3")
    run(["tools.eval_farasis", "--project", str(proj), "--ckpt", ck, "--split", sp]
        + (["--tta"] if tta else []) + (["--median", med] if med != "0" else [])
        + ["--min-areas", "0,16,64,256,1024", "--probs", "0.3,0.5,0.7"])
    print(f"\n  报告: {proj / 'models' / (Path(ck).stem + '_eval_report.md')}")


def do_eval_cell():
    clear()
    hr()
    print("  [4] 整图滑窗评估：电芯级 漏检 / 过杀（产线口径）")
    hr()
    print("  在整张原图上滑窗拼接后逐缺陷匹配 —— 只有这个口径能对接产线协议。")
    print("  需要能访问原始导出目录（读原图与 LabelMe 标注做 GT）。\n")
    proj = choose_project()
    if not proj:
        return
    ck = _choose_ckpt(proj)
    if not ck:
        return
    src = pick_dir("选择 LabelMe 导出根目录", DEFAULT_SRC) or DEFAULT_SRC
    sp = ask("评估切分 [val / train / all]", "val")
    lim = ask("最多评估多少颗电芯（0=全部）", "40")
    tta = ask("开启 TTA（更稳，耗时翻倍）[y/n]", "y").lower().startswith("y")
    med = ask("概率图中值滤波核 [0/3/5]（治碎点，推荐 3）", "3")
    print()
    run(["tools.infer_cell_farasis", "--project", str(proj), "--ckpt", ck, "--src", src,
         "--split", sp, "--limit", lim]
        + (["--tta"] if tta else []) + (["--median", med] if med != "0" else [])
        + ["--min-areas", "0,16,64,256", "--probs", "0.5"])
    print(f"\n  报告: {proj / 'models' / (Path(ck).stem + '_cell_cell_report.md')}")


def do_preview():
    clear()
    hr()
    print("  [5] 补生成人工抽检图（原图 + 掩膜叠加）")
    hr()
    print("  导入完成后想再看几张标注解析效果时用这个（不用重新切图）。\n")
    proj = choose_project()
    if not proj:
        return
    n = ask("生成张数", "24")
    run(["tools.import_farasis_labelme", "--src", "x", "--dst", str(proj),
         "--preview-only", "--preview", n])
    print(f"\n  抽检图: {proj / 'preview'}")


def do_open():
    p = pick_dir("选择要打开的目录", str(DS)) or str(DS)
    try:
        os.startfile(p)
    except Exception as e:
        print(f"  打开失败: {e}")


def do_rebuild_masks():
    clear()
    hr()
    print("  [7] 从标注 JSON 重建掩膜 + 补类别表")
    hr()
    print("  什么时候用：")
    print("    · Class 表一行都没有 / 画布选不到缺陷类别（工程 classes 是空的）")
    print("    · 预览里标注颜色或遮挡看着不对（掩膜像素值与类别表对不上）")
    print("    · 训练读到的类别张冠李戴")
    print("  原理：标注真源是 images/*.json，annotations/*_te_0_mask.png 只是派生结果。")
    print("        本工具用统一类别表重新光栅化，并保证 project.json 的类别表包含所有出现过的标签。\n")
    proj = choose_project()
    if not proj:
        return
    show_project_info(proj)
    print("\n  先跑一遍 DRY-RUN（不写任何文件）...\n")
    run(["tools.rebuild_masks_from_json", str(proj), "--dry-run"])
    ans = ask("确认按上面的报告重建并写盘？[y/N]", "N").lower()
    if not ans.startswith("y"):
        print("  已跳过写盘")
        return
    run(["tools.rebuild_masks_from_json", str(proj), "--apply"])
    print("\n  完成。回平台打开工程（或点 Class 表的「刷新」）即可看到类别与统计。")


MENU = [
    ("1", "导入 LabelMe 三视图导出 → 可训练工程（自动原分辨率切图）", do_import),
    ("2", "训练三图分割模型", do_train),
    ("3", "评估：漏检 / 过杀 工作曲线（tile 级，含尺寸分层）", do_eval_tile),
    ("4", "整图滑窗评估：电芯级 漏检 / 过杀（产线口径）", do_eval_cell),
    ("5", "补生成人工抽检图（原图 + 掩膜叠加）", do_preview),
    ("6", "打开目录", do_open),
    ("7", "从标注 JSON 重建掩膜 + 补类别表（修类别/掩膜不一致）", do_rebuild_masks),
]


def main():
    while True:
        clear()
        hr()
        print("    孚能 三视图（漫反射 / 形状图1 / 形状图2）语义分割 工具集")
        hr()
        print(f"  工程目录 : {DS}")
        print(f"  程序目录 : {APP}")
        print(f"  解释器   : {sys.executable}")
        hr("-")
        for k, desc, _ in MENU:
            print(f"  [{k}] {desc}")
        print("  [0] 退出")
        hr("-")
        ch = input("  请输入序号并回车: ").strip()
        if ch == "0":
            break
        for k, _, fn in MENU:
            if ch == k:
                try:
                    fn()
                except KeyboardInterrupt:
                    print("\n  已取消")
                except Exception as e:
                    print(f"\n  [异常] {type(e).__name__}: {e}")
                input("\n  按回车返回菜单...")
                break
    print("  再见")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
