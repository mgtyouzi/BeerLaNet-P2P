import json
import os

import torch
from torch.utils.data import DataLoader

import train_p2p as base
import train_p2p_no_empty_v2 as no_empty
from models.beerlanet_p2p import build_beerlanet_p2p
from train_beerlanet_p2p import configure_protocol, get_args_parser, set_reproducibility


def get_eval_parser():
    parser = get_args_parser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_name", required=True)
    return parser


def main():
    args = get_eval_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the unchanged P2P evaluator")

    args.distributed = False
    args.rank = 0
    args.gpu = 0
    args.world_size = 1
    torch.cuda.set_device(0)
    configure_protocol(args)
    set_reproducibility(args)

    dataset = base.build_dataset(args, "test")
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=base.collate_fn_pad,
    )
    model = build_beerlanet_p2p(args, initialize_p2p=False).cuda(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    evaluator = base.Evaluator(data_loader)
    metrics = evaluator.calculate_metrics(
        model,
        effective_matching_dis=args.match_dis,
        rank=0,
    )
    precision, recall, f1 = metrics[no_empty.KEY_DET]
    counts = metrics["eval_counts"]
    summary = {
        "case": args.case_name,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "beerlanet_mode": args.beerlanet_mode,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": counts["det_tp"],
        "pred": counts["det_pred"],
        "gt": counts["det_gt"],
        "fp": counts["det_pred"] - counts["det_tp"],
        "fn": counts["det_gt"] - counts["det_tp"],
        "mae": metrics["MAE"],
        "mse": metrics["MSE"],
        "checkpoint_selection_role": (
            "source_ffpe_selection" if args.case_name.startswith("ffpe_")
            else "diagnostic_only_no_checkpoint_selection"
        ),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[BeerLaNet-P2P eval] output={output_path}", flush=True)


if __name__ == "__main__":
    main()
