import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import diagnose_cross_domain as diagnosis
import train_p2p as p2p
from utils import cleanup, get_rank, init_distributed_mode

CHECKPOINT = Path('/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth')
DATASET = Path('/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets') / '\u51b0\u51bb-2025'
SOURCE_MEAN_STD = Path('/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets') / '\u77f3\u8721-2025' / 'mean_std.npy'
OUTPUT_DIR = ROOT / 'reports' / 'strict_source_only_baseline_eval'
REPORT = ROOT / 'reports' / 'STRICT_SOURCE_ONLY_BASELINE_REPORT.md'

def main():
    for path in (CHECKPOINT, DATASET, SOURCE_MEAN_STD):
        if not path.exists():
            raise FileNotFoundError(path)
    parser = diagnosis.get_parser()
    args = parser.parse_args([
        '--checkpoint', str(CHECKPOINT),
        '--case_name', 'strict_source_only_baseline',
        '--dataset', str(DATASET),
        '--output_dir', str(OUTPUT_DIR),
        '--num_workers', '0',
        '--match_dis', '15',
        '--dedup_interval', '15',
        '--skip_empty_gt_in_eval',
        '--strict_load',
    ])
    args.mean_std_path = str(SOURCE_MEAN_STD)
    args.test_mean_std_path = str(SOURCE_MEAN_STD)
    p2p.args = args
    init_distributed_mode(args)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    print(f'[STRICT-SOURCE] dataset={DATASET}', flush=True)
    print(f'[STRICT-SOURCE] mean_std={SOURCE_MEAN_STD}', flush=True)
    print(f'[STRICT-SOURCE] checkpoint={CHECKPOINT}', flush=True)
    diagnosis.run_diagnosis(args)
    if getattr(args, 'distributed', False):
        cleanup()
    summary = json.loads((OUTPUT_DIR / 'summary.json').read_text(encoding='utf-8'))
    metrics = {'P': summary['precision'], 'R': summary['recall'], 'F1': summary['f1'], 'TP': int(summary['tp']), 'FP': int(summary['fp']), 'FN': int(summary['fn'])}
    REPORT.write_text(
        '# Strict Source-Only Baseline\n\n'
        '| P | R | F1 | TP | FP | FN |\n'
        '|---:|---:|---:|---:|---:|---:|\n'
        f"| {metrics['P']:.6f} | {metrics['R']:.6f} | {metrics['F1']:.6f} | {metrics['TP']} | {metrics['FP']} | {metrics['FN']} |\n",
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f'report={REPORT}', flush=True)

if __name__ == '__main__':
    main()
