import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_p2p as base
from models.beerlanet_p2p import build_beerlanet_p2p
from train_beerlanet_p2p import configure_protocol, get_args_parser, set_reproducibility


def main():
    parser = get_args_parser()
    parser.add_argument("--smoke_output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real P2P forward smoke")

    args.distributed = False
    args.rank = 0
    args.gpu = 0
    args.world_size = 1
    torch.cuda.set_device(0)
    configure_protocol(args)
    set_reproducibility(args)

    dataset = base.build_dataset(args, "test")
    sample = dataset[0]
    image = sample[0]
    images = image.unsqueeze(0).cuda(0)
    model = build_beerlanet_p2p(args, initialize_p2p=True).cuda(0).eval()

    with torch.no_grad():
        rgb, x0, stain_matrix, density, adapted = model.forward_components(images)
        outputs = model.p2p(adapted)

    height, width = images.shape[-2:]
    expected_anchors = math.ceil(width / 32.0) * math.ceil(height / 32.0) * 5
    actual_anchors = int(outputs["pnt_coords"].shape[1])
    tensors = {
        "rgb": rgb,
        "x0": x0,
        "stain_matrix": stain_matrix,
        "density": density,
        "adapted": adapted,
        "pnt_coords": outputs["pnt_coords"],
        "cls_logits": outputs["cls_logits"],
    }
    finite = {name: bool(torch.isfinite(value).all().item()) for name, value in tensors.items()}
    if not all(finite.values()):
        raise RuntimeError(f"non-finite tensors in forward smoke: {finite}")
    if tuple(density.shape) != (1, args.beerlanet_r, height, width):
        raise RuntimeError(f"unexpected density shape: {tuple(density.shape)}")
    if tuple(adapted.shape) != (1, 3, height, width):
        raise RuntimeError(f"unexpected adapted shape: {tuple(adapted.shape)}")
    if actual_anchors != expected_anchors:
        raise RuntimeError(f"anchor mismatch: actual={actual_anchors}, expected={expected_anchors}")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise RuntimeError("restored RGB is outside [0,1]")

    report = {
        "status": "PASS",
        "mode": args.beerlanet_mode,
        "input_shape": list(images.shape),
        "rgb_shape": list(rgb.shape),
        "x0_shape": list(x0.shape),
        "stain_matrix_shape": list(stain_matrix.shape),
        "density_shape": list(density.shape),
        "adapted_shape": list(adapted.shape),
        "pnt_coords_shape": list(outputs["pnt_coords"].shape),
        "cls_logits_shape": list(outputs["cls_logits"].shape),
        "expected_anchors": expected_anchors,
        "actual_anchors": actual_anchors,
        "ranges": {
            name: [float(value.min()), float(value.max())]
            for name, value in tensors.items()
        },
        "finite": finite,
        "p2p_output_keys": sorted(outputs.keys()),
        "beerlanet_r": args.beerlanet_r,
        "beerlanet_n_iter": args.beerlanet_n_iter,
        "initialized_p2p_epoch": model.initialized_p2p_epoch,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.smoke_output)), exist_ok=True)
    with open(args.smoke_output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
