import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import train_p2p as base
import train_p2p_no_empty_v2 as no_empty
from models.beerlanet_p2p import build_beerlanet_p2p


def get_args_parser():
    parser = no_empty.get_args_parser()
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--beerlanet_mode", choices=("frozen", "joint"), required=True)
    parser.add_argument("--beerlanet_r", default=8, type=int)
    parser.add_argument("--beerlanet_n_iter", default=10, type=int)
    parser.add_argument(
        "--beerlanet_gradient_checkpointing",
        action="store_true",
        help="recompute BeerLaNet during backward to reduce joint-mode activation memory",
    )
    return parser


def configure_protocol(args):
    base.args = args
    base._original_build_dataset = base.build_dataset
    base.build_dataset = no_empty.build_dataset_no_empty
    if args.skip_empty_eval:
        base.Evaluator.calculate_metrics = no_empty.calculate_metrics_skip_empty


def install_model_builder(args, initialize_p2p=True):
    def builder(_):
        model = build_beerlanet_p2p(args, initialize_p2p=initialize_p2p)
        if no_empty.is_main():
            print(
                "[BeerLaNet-P2P] "
                f"mode={args.beerlanet_mode}, r={args.beerlanet_r}, "
                f"n_iter={args.beerlanet_n_iter}, checkpointing={args.beerlanet_gradient_checkpointing}, "
                f"initialized_p2p_epoch={model.initialized_p2p_epoch}",
                flush=True,
            )
        return model

    base.build_model = builder


def set_reproducibility(args):
    seed = args.seed + base.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    args = get_args_parser().parse_args()
    configure_protocol(args)
    base.init_distributed_mode(args)
    set_reproducibility(args)
    install_model_builder(args, initialize_p2p=True)

    if no_empty.is_main():
        print(
            "[Stage-C protocol] FFPE-train only; FFPE test selects checkpoints; "
            "Frozen is not accessed by this training entry.",
            flush=True,
        )
        print(
            f"[Stage-C protocol] seed={args.seed}, batch_size={args.batch_size}, "
            f"epochs={args.epochs}, lr={args.lr}, dataset={args.dataset}",
            flush=True,
        )
    base.train()


if __name__ == "__main__":
    main()
