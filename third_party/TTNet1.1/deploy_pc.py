#!/usr/bin/env python3
"""跨平台部署自检与启动入口；帮助、静态检查及命令预览只依赖标准库。"""

import argparse
import ast
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = "checkpoints/ttnet_3rd_phase/ttnet.pth"
DEFAULT_VIDEO = "dataset/test/videos/test_2.mp4"
DEPENDENCIES = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
    "opencv-python": "cv2", "easydict": "easydict",
    "scikit-learn": "sklearn", "tensorboard": "tensorboard", "tqdm": "tqdm",
    "matplotlib": "matplotlib", "Pillow": "PIL", "wget": "wget",
}


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须大于或等于 0")
    return number


def output_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("输出名称仅允许英文字母、数字、下划线和短横线")
    return value


def syntax_check():
    paths = [ROOT / "deploy_pc.py", ROOT / "make_video.py"]
    paths += sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "prepare_dataset").rglob("*.py"))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(f"语法检查通过：{len(paths)} 个 Python 文件（未写入字节码）")


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"文件缺失或为空：{path}；请先恢复独立资产包")


def environment_check(gpu, runtime=False):
    print(f"解释器：{sys.executable}，Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("运行模型的目标环境为 Python 3.11；其他版本仅用于 --static 或 --dry-run")
    missing = []
    for distribution in DEPENDENCIES:
        try:
            print(f"{distribution}=={importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError("缺少依赖：" + ", ".join(missing) + "；请按 DEPLOY_NOTES.md 安装")
    if not runtime:
        return
    for module in DEPENDENCIES.values():
        importlib.import_module(module)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；本基线演示/训练不支持纯 CPU，需 NVIDIA GPU、驱动及 CUDA 版 PyTorch")
    if gpu >= torch.cuda.device_count():
        raise RuntimeError(f"GPU 索引 {gpu} 超出可见设备范围")
    # 原 demo 使用 model.cuda()，通过子进程 CUDA_VISIBLE_DEVICES 将所选卡映射为 0。
    probe = torch.ones(1, device=f"cuda:{gpu}")
    torch.cuda.synchronize(gpu)
    print(f"CUDA 检查通过：{torch.cuda.get_device_name(gpu)}，torch CUDA {torch.version.cuda}，probe={probe.item()}")


def training_data_check():
    # main.py 即使传 --no-test 也会构造测试集，因此两部分都必须准备。
    for split, games in (("training", [f"game_{i}" for i in range(1, 6)]),
                         ("test", [f"test_{i}" for i in range(1, 8)])):
        for game in games:
            base = ROOT / "dataset" / split
            for annotation in ("ball_markup.json", "events_markup.json"):
                require_file(base / "annotations" / game / annotation)
            masks = base / "annotations" / game / "segmentation_masks"
            images = base / "images" / game
            if not any(masks.glob("*.png")) or not any(images.glob("img_*.jpg")):
                raise RuntimeError(f"训练所需标注掩码或抽帧缺失：{split}/{game}；资产包不含完整训练数据")
    print("训练/测试数据目录初检通过；不代表每个标注和帧均完整匹配")


def commands(args):
    common = ["--working-dir", str(ROOT), "--gpu_idx", "0"]
    if args.command == "demo":
        command = [sys.executable, str(ROOT / "src/demo.py"), *common,
                   "--saved_fn", args.saved_fn, "--arch", "ttnet",
                   "--pretrained_path", str(project_path(args.weights)),
                   "--video_path", str(project_path(args.video)),
                   "--seg_thresh", "0.5", "--event_thresh", "0.5",
                   "--thresh_ball_pos_mask", "0.05", "--save_demo_output",
                   "--output_format", "video", "--stream_video"]
        if args.max_frames is not None:
            command += ["--max_frames", str(args.max_frames)]
        return [command]
    if args.command == "verify":
        return [[sys.executable, str(ROOT / "src/verify.py")]]
    common += ["--no-val", "--batch_size", str(args.batch_size),
               "--num_workers", str(args.num_workers), "--num_epochs", "30",
               "--lr_type", "step_lr", "--lr_step_size", "10",
               "--smooth-labelling", "--checkpoint_freq", "30"]
    phases = [
        ["--saved_fn", "ttnet_1st_phase", "--lr", "0.001", "--lr_factor", "0.1",
         "--global_weight", "5.", "--seg_weight", "1.", "--no_local", "--no_event"],
        ["--saved_fn", "ttnet_2nd_phase", "--lr", "0.001", "--lr_factor", "0.1",
         "--global_weight", "0.", "--seg_weight", "0.", "--event_weight", "2.", "--local_weight", "1.",
         "--pretrained_path", str(ROOT / "checkpoints/ttnet_1st_phase/ttnet_1st_phase_epoch_30.pth"),
         "--overwrite_global_2_local", "--freeze_seg", "--freeze_global"],
        ["--saved_fn", "ttnet_3rd_phase", "--lr", "0.0001", "--lr_factor", "0.2",
         "--global_weight", "1.", "--seg_weight", "1.", "--event_weight", "1.", "--local_weight", "1.",
         "--pretrained_path", str(ROOT / "checkpoints/ttnet_2nd_phase/ttnet_2nd_phase_epoch_30.pth")],
    ]
    return [[sys.executable, str(ROOT / "src/main.py"), *common, *phase] for phase in phases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="检查语法、依赖元数据与默认演示资产")
    check.add_argument("--static", action="store_true", help="仅检查 Python 语法，不要求模型依赖/资产")
    check.add_argument("--runtime", action="store_true", help="额外实际导入依赖并执行 CUDA 张量探测，不加载权重")
    for name in ("demo", "train", "verify"):
        child = sub.add_parser(name, help={"demo": "启动 GPU 演示", "train": "串联三阶段 GPU 训练", "verify": "加载默认权重并验证 GPU 前向"}[name])
        child.add_argument("--dry-run", action="store_true", help="只打印命令，不导入依赖或执行模型")
        if name == "demo":
            child.add_argument("--saved-fn", default="demo", type=output_name)
            child.add_argument("--max-frames", type=positive_int)
        if name == "train":
            child.add_argument("--batch-size", default=8, type=positive_int)
            child.add_argument("--num-workers", default=4, type=nonnegative_int)
    for child in (check, sub.choices["demo"]):
        child.add_argument("--weights", default=DEFAULT_WEIGHTS, help="绝对路径或相对项目根目录的权重路径")
        child.add_argument("--video", default=DEFAULT_VIDEO, help="绝对路径或相对项目根目录的视频路径")
    for child in sub.choices.values():
        child.add_argument("--gpu", default=0, type=nonnegative_int, help="CUDA_VISIBLE_DEVICES 过滤前的可见设备索引")
    args = parser.parse_args()
    if args.command == "check" and args.static and args.runtime:
        parser.error("--static 与 --runtime 不可同时使用")
    try:
        if args.command == "check":
            syntax_check()
            if args.static:
                return 0
            environment_check(args.gpu, args.runtime)
            require_file(project_path(args.weights))
            require_file(project_path(args.video))
            print("演示资产路径检查通过；尚未验证权重内容/视频解码，请继续 verify 和短视频演示")
            return 0
        launch_commands = commands(args)
        if not args.dry_run:
            environment_check(args.gpu, runtime=True)
            if args.command == "train":
                training_data_check()
            else:
                require_file(project_path(args.weights if args.command == "demo" else DEFAULT_WEIGHTS))
                if args.command == "demo":
                    require_file(project_path(args.video))
        env = os.environ.copy()
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            devices = [value.strip() for value in visible.split(",") if value.strip()]
            if args.gpu >= len(devices):
                raise RuntimeError("--gpu 超出 CUDA_VISIBLE_DEVICES 范围")
            env["CUDA_VISIBLE_DEVICES"] = devices[args.gpu]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for command in launch_commands:
            print(f"工作目录：{ROOT / 'src'}", flush=True)
            print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
            print("参数：" + repr(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT / "src", env=env, check=True)
        return 0
    except (OSError, SyntaxError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"部署检查失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
