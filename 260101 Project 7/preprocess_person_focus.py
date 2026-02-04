"""
Luồng riêng: Object detection (phát hiện vật thể/chủ thể) -> cắt frame focus vào vùng có vật thể.
Mặc định: YOLO11 (Ultralytics), ưu tiên người. --backend torchvision: dùng Faster R-CNN.
Cần: pip install ultralytics (khi dùng yolo11).

Cách dùng:
  conda run -n pytorch_gpu --no-capture-output python preprocess_person_focus.py --input-dir ./kaggle_data/data --output-dir ./kaggle_data/data_person_focus

Sau đó train/inference với:
  --data-dir ./kaggle_data/data_person_focus

Cấu trúc kaggle_data:
  input_dir/data_train/<class>/<video_dir>/*.jpg
  input_dir/test/<video_id>/*.jpg
Output giữ nguyên cấu trúc trong output_dir.
"""

from __future__ import annotations

import os
import sys

# Tránh xung đột OpenMP trên Windows (nhiều bản libiomp5md.dll: PyTorch/MKL)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

# Windows: đường dẫn dài gây [Errno 22] Invalid argument; dùng long path sớm (200)
MAX_PATH_WIN = 200


def _path_for_io(p: Path) -> str:
    """Chuỗi path dùng cho open/mkdir/save; trên Windows dùng long path nếu cần."""
    s = str(p.resolve())
    if sys.platform == "win32" and len(s) >= MAX_PATH_WIN and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


# UTF-8 cho Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_frames(root: Path, subdir: str) -> list[tuple[Path, Path]]:
    """
    Thu thập (path_gốc, path_đích_tương_đối) cho mọi frame.
    root: thư mục gốc (vd: kaggle_data/data)
    subdir: 'data_train' hoặc 'test'
    Trả về [(src_path, rel_path), ...] với rel_path tương đối từ root (vd: data_train/class/video/frame.jpg).
    """
    base = root / subdir
    if not base.exists():
        return []
    out = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        # data_train: class / video_dir / frames; test: video_id / frames
        for video_dir in sorted([d for d in item.iterdir() if d.is_dir()]):
            for f in sorted(video_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rel = Path(subdir) / item.name / video_dir.name / f.name
                    out.append((f, rel))
    return out


def run(
    input_dir: Path,
    output_dir: Path,
    score_threshold: float = 0.15,
    padding_ratio: float = 0.15,
    skip_existing: bool = True,
    person_only: bool = False,
    backend: str = "yolo11",
    yolo_model: str = "yolo26m.pt",
    workers: int = 0,
) -> None:
    from PIL import Image
    from sota_training.person_detection import PersonFocusCrop

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="person_focus_"))
    logger.info("Ghi tạm vào: %s (sẽ copy vào %s khi xử lý xong hết)", work_dir, output_dir)
    logger.info("Backend detection: %s", backend)
    processor = PersonFocusCrop(
        score_threshold=score_threshold,
        padding_ratio=padding_ratio,
        fallback_full_frame=False,
        person_only=person_only,
        backend=backend,
        yolo_model=yolo_model,
    )

    for subdir in ("data_train", "test"):
        pairs = collect_frames(input_dir, subdir)
        if not pairs:
            logger.info("No frames under %s/%s", input_dir, subdir)
            continue
        all_folders = {rel.parent for _, rel in pairs}
        folder_written = {f: 0 for f in all_folders}
        mode = "chỉ người" if person_only else "vật thể (ưu tiên người)"
        logger.info(
            "Processing %s: %d frames, %d folders (chỉ ghi frame có %s)",
            subdir, len(pairs), len(all_folders), mode,
        )
        written = 0
        skipped = 0
        completed_skip_folders = 0  # Số folder đã xử lý xong và có 0 ảnh ghi
        prev_folder = None
        pbar = tqdm(pairs, desc=subdir, unit="frame", dynamic_ncols=True)
        for src, rel in pbar:
            out_in_final = output_dir / rel
            out_path = work_dir / rel
            folder = rel.parent
            if prev_folder is not None and folder != prev_folder:
                if folder_written[prev_folder] == 0:
                    completed_skip_folders += 1
            prev_folder = folder
            if skip_existing and out_in_final.exists():
                written += 1
                folder_written[folder] += 1
                pbar.set_postfix(written=written, skipped=skipped, skip_folders=completed_skip_folders)
                continue
            cropped, had_person = processor.process_frame_path(src)
            if had_person and cropped is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(cropped).save(str(out_path), quality=95)
                written += 1
                folder_written[folder] += 1
            else:
                skipped += 1
            pbar.set_postfix(written=written, skipped=skipped, skip_folders=completed_skip_folders)
        if prev_folder is not None and folder_written[prev_folder] == 0:
            completed_skip_folders += 1
        folders_skip = sum(1 for c in folder_written.values() if c == 0)
        skipped_folder_set = {f for f, c in folder_written.items() if c == 0}
        skipped_folder_list = sorted(str(f) for f in skipped_folder_set)
        logger.info(
            "Done %s: written %d, skipped %d, folders toàn bộ ảnh bị skip: %d",
            subdir, written, skipped, folders_skip,
        )
        if skipped_folder_set:
            logger.info("Giữ lại %d folder (copy ảnh gốc)...", len(skipped_folder_set))
            to_copy = [(src, rel) for src, rel in pairs if rel.parent in skipped_folder_set]
            for src, rel in tqdm(to_copy, desc="Copy folder giữ nguyên", unit="frame"):
                out_in_final = output_dir / rel
                out_path = work_dir / rel
                if skip_existing and out_in_final.exists():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_path_for_io(src), str(out_path))
            logger.info("Đã copy %d ảnh gốc từ %d folder (giữ nguyên folder).", len(to_copy), len(skipped_folder_set))
            if len(skipped_folder_list) <= 30:
                for f in skipped_folder_list:
                    logger.info("  [folder giữ nguyên] %s", f)
            else:
                for f in skipped_folder_list[:30]:
                    logger.info("  [folder giữ nguyên] %s", f)
                logger.info("  ... và %d folder khác", len(skipped_folder_list) - 30)
                skip_file = work_dir / f"kept_folders_{subdir}.txt"
                with open(skip_file, "w", encoding="utf-8") as fp:
                    fp.write("\n".join(skipped_folder_list))
                logger.info("  Danh sách: %s", skip_file)

    # Xử lý xong hết: copy toàn bộ từ work_dir vào output_dir, rồi xóa temp
    logger.info("Copy toàn bộ từ %s vào %s...", work_dir, output_dir)
    for p in work_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(work_dir)
            dest = output_dir / rel
            os.makedirs(_path_for_io(dest.parent), exist_ok=True)
            shutil.copy2(str(p), _path_for_io(dest))
    try:
        shutil.rmtree(work_dir)
        logger.info("Đã xóa thư mục tạm: %s", work_dir)
    except OSError as e:
        logger.warning("Không xóa được thư mục tạm %s: %s", work_dir, e)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess kaggle_data: person detection -> crop frames (focus on human action)."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./kaggle_data/data"),
        help="Thư mục gốc (có data_train, test)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./kaggle_data/data_person_focus"),
        help="Thư mục ghi dữ liệu đã xử lý (cùng cấu trúc)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="yolo11",
        choices=("yolo11", "torchvision"),
        help="Backend detection: yolo11 (mặc định, Ultralytics) hoặc torchvision (Faster R-CNN)",
    )
    p.add_argument(
        "--yolo-model",
        type=str,
        default="yolo26m.pt",
        help="Tên model YOLO khi backend=yolo11 (vd: yolo26n.pt, yolo26m.pt, yolo11m.pt)",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.15,
        help="Ngưỡng confidence detection (mặc định 0.15; càng thấp càng ít skip)",
    )
    p.add_argument(
        "--person-only",
        action="store_true",
        help="Chỉ phát hiện người. Mặc định: mọi vật thể (ưu tiên người).",
    )
    p.add_argument("--padding-ratio", type=float, default=0.15, help="Padding quanh bbox vật thể")
    p.add_argument("--no-skip-existing", action="store_true", help="Ghi đè frame đã tồn tại")
    p.add_argument("--workers", type=int, default=0, help="Số worker (0 = single process)")
    args = p.parse_args()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        padding_ratio=args.padding_ratio,
        skip_existing=not args.no_skip_existing,
        person_only=args.person_only,
        backend=args.backend,
        yolo_model=args.yolo_model,
        workers=args.workers,
    )
    logger.info("Preprocess xong. Dùng --data-dir %s để train/inference.", args.output_dir)


# Verification: sau khi chạy, kiểm tra output_dir có data_train và test với cùng cấu trúc;
# train/inference: python ... --data-dir <output_dir>
if __name__ == "__main__":
    main()
