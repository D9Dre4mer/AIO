"""
Xử lý riêng data TEST: Object detection -> cắt frame focus vào vùng có vật thể.
Hỗ trợ: test/<video_id>/*.jpg hoặc test/<video_id>/<subdir>/*.jpg
(Script gốc dùng cấu trúc 3 cấp nên không thu frame test khi test chỉ có 2 cấp.)

Cách dùng:
  conda run -n pytorch_gpu --no-capture-output python preprocess_person_focus_test.py \\
    --input-dir ./kaggle_data/data --output-dir ./kaggle_data/data_person_focus
Inference: --data-dir ./kaggle_data/data_person_focus
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MAX_PATH_WIN = 200


def _path_for_io(p: Path) -> str:
    s = str(p.resolve())
    if (
        sys.platform == "win32"
        and len(s) >= MAX_PATH_WIN
        and not s.startswith("\\\\?\\")
    ):
        return "\\\\?\\" + s
    return s


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


def collect_test_frames(root: Path) -> list[tuple[Path, Path]]:
    """
    Thu thập (path_gốc, path_đích_tương_đối) cho mọi frame trong test/.
    Hỗ trợ:
      - test/<video_id>/*.jpg (frame trực tiếp trong video_id)
      - test/<video_id>/<subdir>/*.jpg (một cấp con)
    rel_path dạng test/video_id/.../frame.jpg. Trả về [(src_path, rel_path), ...].
    """
    base = root / "test"
    if not base.exists():
        return []
    out = []
    for video_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for f in sorted(video_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            rel = Path("test") / video_dir.name / f.relative_to(video_dir)
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
    work_dir = Path(tempfile.mkdtemp(prefix="person_focus_test_"))
    logger.info("Ghi tạm vào: %s (sẽ copy vào %s khi xong)", work_dir, output_dir)
    logger.info("Backend detection: %s", backend)
    processor = PersonFocusCrop(
        score_threshold=score_threshold,
        padding_ratio=padding_ratio,
        fallback_full_frame=False,
        person_only=person_only,
        backend=backend,
        yolo_model=yolo_model,
    )

    pairs = collect_test_frames(input_dir)
    if not pairs:
        logger.warning("Không có frame nào dưới %s/test", input_dir)
        return
    all_folders = {rel.parent for _, rel in pairs}
    folder_written = {f: 0 for f in all_folders}
    mode = "chỉ người" if person_only else "vật thể (ưu tiên người)"
    logger.info(
        "Processing test: %d frames, %d folders (chỉ ghi frame có %s)",
        len(pairs), len(all_folders), mode,
    )
    written = 0
    skipped = 0
    completed_skip_folders = 0
    prev_folder = None
    pbar = tqdm(pairs, desc="test", unit="frame", dynamic_ncols=True)
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
            pbar.set_postfix(
                written=written, skipped=skipped, skip_folders=completed_skip_folders,
            )
            continue
        cropped, had_person = processor.process_frame_path(src)
        if had_person and cropped is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cropped).save(str(out_path), quality=95)
            written += 1
            folder_written[folder] += 1
        else:
            skipped += 1
        pbar.set_postfix(
            written=written, skipped=skipped, skip_folders=completed_skip_folders,
        )
    if prev_folder is not None and folder_written[prev_folder] == 0:
        completed_skip_folders += 1
    folders_skip = sum(1 for c in folder_written.values() if c == 0)
    skipped_folder_set = {f for f, c in folder_written.items() if c == 0}
    skipped_folder_list = sorted(str(f) for f in skipped_folder_set)
    logger.info(
        "Done test: written %d, skipped %d, folders toàn bộ ảnh bị skip: %d",
        written, skipped, folders_skip,
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
        logger.info(
            "Đã copy %d ảnh gốc từ %d folder (giữ nguyên folder).",
            len(to_copy), len(skipped_folder_set),
        )
        if len(skipped_folder_list) <= 30:
            for f in skipped_folder_list:
                logger.info("  [folder giữ nguyên] %s", f)
        else:
            for f in skipped_folder_list[:30]:
                logger.info("  [folder giữ nguyên] %s", f)
            logger.info("  ... và %d folder khác", len(skipped_folder_list) - 30)
        skip_file = work_dir / "kept_folders_test.txt"
        with open(skip_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(skipped_folder_list))
        logger.info("  Danh sách: %s", skip_file)

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
        description="Preprocess chỉ data TEST: person detection -> crop frames.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./kaggle_data/data"),
        help="Thư mục gốc (có thư mục test)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./kaggle_data/data_person_focus"),
        help="Thư mục ghi dữ liệu đã xử lý (test/ giữ cấu trúc)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="yolo11",
        choices=("yolo11", "torchvision"),
        help="Backend detection: yolo11 hoặc torchvision",
    )
    p.add_argument(
        "--yolo-model",
        type=str,
        default="yolo26m.pt",
        help="Tên model YOLO khi backend=yolo11",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.15,
        help="Ngưỡng confidence detection",
    )
    p.add_argument(
        "--person-only",
        action="store_true",
        help="Chỉ phát hiện người",
    )
    p.add_argument(
        "--padding-ratio", type=float, default=0.15, help="Padding quanh bbox",
    )
    p.add_argument(
        "--no-skip-existing", action="store_true", help="Ghi đè frame đã tồn tại",
    )
    p.add_argument(
        "--workers", type=int, default=0, help="Số worker (0 = single process)",
    )
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
    logger.info(
        "Preprocess test xong. Dùng --data-dir %s để inference.", args.output_dir,
    )


if __name__ == "__main__":
    main()
