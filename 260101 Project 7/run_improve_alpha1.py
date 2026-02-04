"""
Script gọi tự động improve model Alpha1 (cải thiện các expert yếu).

Mặc định:
- Checkpoint: checkpoints/videomae_model_alpha1_best.pt
- Bật --retrain-experts (train lại expert heads của weak groups)

Chạy:
  conda run -n pytorch_gpu python run_improve_alpha1.py

Sửa checkpoint Alpha1 (ghi config['label_subsets'] khớp state_dict, lần sau không fallback):
  python run_improve_alpha1.py --fix-config [--checkpoint PATH]

Tùy chọn (forward sang improve_weak_heads_alpha):
  --checkpoint PATH    Checkpoint khác (mặc định: videomae_model_alpha1_best.pt)
  --fix-config         Chỉ sửa config checkpoint rồi thoát (không evaluate/retrain)
  --no-retrain         Chỉ evaluate + báo weak groups, không retrain
  --data-dir DIR       Thư mục data (mặc định: ./kaggle_data/data)
  --output-dir DIR     Thư mục lưu checkpoint improved (mặc định: ./checkpoints)
  --output-name NAME   Tên file improved
                       (mặc định: videomae_model_alpha_improved_heads.pt)
  --group-threshold F  Group acc < F coi là yếu (mặc định: 0.85)
  --improve-bottom-k-groups K  Luôn retrain K group thấp nhất (mặc định: 0)
  --expert-epochs N    Epochs mỗi expert (mặc định: 30)
  --expert-lr LR       Learning rate expert (mặc định: 2e-4)
  --batch-size N       Batch size (mặc định: 24)
"""

import sys
from pathlib import Path


def _get_default_checkpoint():
    base = Path(__file__).resolve().parent
    return base / 'checkpoints' / 'videomae_model_alpha1_best.pt'


def _get_checkpoint_from_argv():
    argv = sys.argv[1:]
    if '--checkpoint' in argv:
        i = argv.index('--checkpoint')
        if i + 1 < len(argv):
            return Path(argv[i + 1])
    return _get_default_checkpoint()


def _inject_defaults():
    base = Path(__file__).resolve().parent
    argv = sys.argv[1:]
    if '--checkpoint' not in argv:
        sys.argv.extend([
            '--checkpoint',
            str(base / 'checkpoints' / 'videomae_model_alpha1_best.pt'),
        ])
    if '--no-retrain' not in argv and '--retrain-experts' not in argv:
        sys.argv.append('--retrain-experts')


if __name__ == '__main__':
    if '--fix-config' in sys.argv:
        sys.argv.remove('--fix-config')
        from sota_training.utils import setup_logging
        from improve_weak_heads_alpha import fix_alpha1_checkpoint_config
        ckpt = _get_checkpoint_from_argv()
        log_dir = ckpt.resolve().parent.parent / 'logging'
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging('alpha', log_dir)
        fix_alpha1_checkpoint_config(ckpt)
        print("Alpha1 checkpoint đã sửa config:", ckpt)
    else:
        _inject_defaults()
        from improve_weak_heads_alpha import main
        main()
