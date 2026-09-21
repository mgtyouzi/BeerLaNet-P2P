import gc
import hashlib
import importlib.util
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PATH = ROOT / "third_party" / "BeerLaNet" / "src" / "BeerLaNet.py"
REPORT_PATH = ROOT / "reports" / "BEERLANET_AUTOGRAD_REPORT.md"
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("official_beerlanet", OFFICIAL_PATH)
official_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_module)
OfficialBeerLaNet = official_module.BeerLaNet

from compat.beerlanet_autograd import BeerLaNetAutogradCompat


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_mutations():
    records = []
    for number, line in enumerate(OFFICIAL_PATH.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if ".data" in stripped:
            category = "initialization-only .data" if number == 66 else "forward parameter .data mutation"
            records.append((number, category, stripped))
        elif re.search(r"\b[A-Za-z_]\w*\s*[+\-*/]=", stripped):
            records.append((number, "autograd-unsafe tensor augmented assignment", stripped))
    return records


def same_state_dict(left, right):
    return list(left) == list(right) and all(torch.equal(left[key], right[key]) for key in left)


def grad_table(model, input_tensor):
    rows = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        rows.append({
            "name": name,
            "requires_grad": parameter.requires_grad,
            "grad_none": grad is None,
            "finite": False if grad is None else bool(torch.isfinite(grad).all().item()),
            "norm": None if grad is None else float(grad.detach().norm().item()),
        })
    input_grad = input_tensor.grad
    rows.append({
        "name": "INPUT",
        "requires_grad": True,
        "grad_none": input_grad is None,
        "finite": False if input_grad is None else bool(torch.isfinite(input_grad).all().item()),
        "norm": None if input_grad is None else float(input_grad.detach().norm().item()),
    })
    return rows


def run_compat_backward(state, base_input, loss_name):
    gc.collect()
    torch.cuda.empty_cache()
    model = BeerLaNetAutogradCompat(r=8, c=3, learn_S_init=True, calc_tau=True).cuda()
    model.load_state_dict(state, strict=True)
    image = base_input.detach().clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    x0, spectra, density = model(image, n_iter=10, unit_norm_S=True)
    if loss_name == "D.mean()":
        loss = density.mean()
    else:
        loss = density.mean() + 0.01 * spectra.mean() + 0.01 * x0.mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "loss": float(loss.detach().item()),
        "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "seconds": elapsed,
        "grads": grad_table(model, image),
        "outputs_finite": all(bool(torch.isfinite(value).all().item()) for value in (x0, spectra, density)),
    }
    del model, image, x0, spectra, density, loss
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested compatibility test")
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    source_hash_before = sha256(OFFICIAL_PATH)
    base_input = torch.rand(1, 3, 256, 256, device=device)

    official = OfficialBeerLaNet(r=8, c=3, learn_S_init=True, calc_tau=True).to(device)
    original_state = {key: value.detach().clone() for key, value in official.state_dict().items()}
    compat = BeerLaNetAutogradCompat(r=8, c=3, learn_S_init=True, calc_tau=True).to(device)
    compat.load_state_dict(original_state, strict=True)
    state_equal = same_state_dict(original_state, compat.state_dict())

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        original_outputs = tuple(value.detach().cpu() for value in official(base_input.clone(), n_iter=10, unit_norm_S=True))
        compat_outputs = tuple(value.detach().cpu() for value in compat(base_input.clone(), n_iter=10, unit_norm_S=True))
    torch.cuda.synchronize()
    forward_allocated = torch.cuda.max_memory_allocated() / 2**20
    forward_reserved = torch.cuda.max_memory_reserved() / 2**20
    labels = ("x0", "S", "D")
    differences = {
        label: float((original - fixed).abs().max().item())
        for label, original, fixed in zip(labels, original_outputs, compat_outputs)
    }

    official_backward = "PASSED"
    try:
        probe = OfficialBeerLaNet(r=8, c=3, learn_S_init=True, calc_tau=True).to(device)
        probe.load_state_dict(original_state, strict=True)
        probe_input = base_input.detach().clone().requires_grad_(True)
        _, _, probe_density = probe(probe_input, n_iter=10, unit_norm_S=True)
        probe_density.mean().backward()
    except RuntimeError as error:
        official_backward = "FAILED: " + str(error).splitlines()[0]
    finally:
        for name in ("probe", "probe_input", "probe_density"):
            if name in locals():
                del locals()[name]
        torch.cuda.empty_cache()

    backward_results = {
        "D.mean()": run_compat_backward(original_state, base_input, "D.mean()"),
        "D.mean()+0.01*S.mean()+0.01*x0.mean()": run_compat_backward(
            original_state, base_input, "composite"
        ),
    }
    source_hash_after = sha256(OFFICIAL_PATH)
    mutations = locate_mutations()
    gradients_ok = all(
        result["outputs_finite"]
        and all(not row["grad_none"] and row["finite"] for row in result["grads"] if row["requires_grad"])
        for result in backward_results.values()
    )
    forward_ok = state_equal and all(value <= 1e-6 for value in differences.values())
    source_unchanged = source_hash_before == source_hash_after
    passed = forward_ok and gradients_ok and source_unchanged

    lines = [
        "# BeerLaNet Autograd Compatibility Report",
        "",
        f"**Status:** {'PASS' if passed else 'FAIL'}",
        "",
        "## Configuration",
        "",
        "- Input: random Uniform(0,1), seed=0, shape `[1,3,256,256]`",
        "- Device: CUDA, dtype: float32",
        "- BeerLaNet: `r=8`, `n_iter=10`, `unit_norm_S=True`",
        f"- Official source SHA256 before/after: `{source_hash_before}` / `{source_hash_after}`",
        f"- Official source unchanged: `{source_unchanged}`",
        "",
        "## Located Mutations",
        "",
        "| Line | Classification | Official operation | Compat treatment |",
        "|---:|---|---|---|",
    ]
    treatment = {
        66: "Initialization only; replace `.data =` with `no_grad()` + `copy_()`.",
        136: "Do not mutate parameter; use local `gamma = abs(self.gamma)`.",
        137: "Do not mutate parameter; use local `lam = abs(self.lam)`.",
        140: "Do not mutate parameter; use local `tau = abs(self.tau)` when applicable.",
        164: "Replace augmented assignment with mathematically equivalent out-of-place update.",
    }
    for number, category, operation in mutations:
        lines.append(f"| {number} | {category} | `{operation}` | {treatment.get(number, 'Classified; unchanged unless required.')} |")
    lines += [
        "",
        "## Minimal Compatibility Diff",
        "",
        "```diff",
        "- self.gamma.data = torch.abs(self.gamma.data)",
        "- self.lam.data   = torch.abs(self.lam.data)",
        "+ gamma = torch.abs(self.gamma)",
        "+ lam = torch.abs(self.lam)",
        "- Dt += -tau_D*(S.T@SDt + S.T@X - S.T@x_0)",
        "+ Dt = Dt - tau_D*(S.T@SDt + S.T@X - S.T@x_0)",
        "```",
        "",
        "## Forward Equivalence",
        "",
        f"- State dict keys and tensors exactly equal before forward: `{state_equal}`",
        "",
        "| Output | Max absolute error |",
        "|---|---:|",
    ]
    for label in labels:
        lines.append(f"| {label} | {differences[label]:.12g} |")
    lines += [
        "",
        f"- Forward peak allocated: {forward_allocated:.2f} MiB",
        f"- Forward peak reserved: {forward_reserved:.2f} MiB",
        "",
        "## Official Backward Control",
        "",
        f"`{official_backward}`",
        "",
        "## Compat Backward",
    ]
    for loss_name, result in backward_results.items():
        lines += [
            "",
            f"### `{loss_name}`",
            "",
            f"- Loss: {result['loss']:.12g}",
            f"- Outputs finite: `{result['outputs_finite']}`",
            f"- Peak allocated: {result['allocated_mib']:.2f} MiB",
            f"- Peak reserved: {result['reserved_mib']:.2f} MiB",
            f"- Forward + backward time: {result['seconds']:.4f} s",
            "",
            "| Tensor | requires_grad | grad is None | finite | grad norm |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in result["grads"]:
            norm = "" if row["norm"] is None else f"{row['norm']:.12g}"
            lines.append(
                f"| {row['name']} | {row['requires_grad']} | {row['grad_none']} | {row['finite']} | {norm} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        f"- Forward equivalence: `{'PASS' if forward_ok else 'FAIL'}`",
        f"- Compat gradients present and finite: `{'PASS' if gradients_ok else 'FAIL'}`",
        f"- Official source unchanged: `{'PASS' if source_unchanged else 'FAIL'}`",
        f"- Final: `{'PASS' if passed else 'FAIL'}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={REPORT_PATH}")
    print(f"status={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError("BeerLaNet autograd compatibility gate failed; inspect report")


if __name__ == "__main__":
    main()
