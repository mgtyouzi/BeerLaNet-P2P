# BeerLaNet Autograd Compatibility Report

**Status:** PASS

## Configuration

- Input: random Uniform(0,1), seed=0, shape `[1,3,256,256]`
- Device: CUDA, dtype: float32
- BeerLaNet: `r=8`, `n_iter=10`, `unit_norm_S=True`
- Official source SHA256 before/after: `c22bb5294211eea63b3149e839b5610604a3ccc73eb155a631e8945663451900` / `c22bb5294211eea63b3149e839b5610604a3ccc73eb155a631e8945663451900`
- Official source unchanged: `True`

## Located Mutations

| Line | Classification | Official operation | Compat treatment |
|---:|---|---|---|
| 66 | initialization-only .data | `self.S_init.data = self.S_init.data/self._S_norm(self.S_init.data)` | Initialization only; replace `.data =` with `no_grad()` + `copy_()`. |
| 136 | forward parameter .data mutation | `self.gamma.data = torch.abs(self.gamma.data)` | Do not mutate parameter; use local `gamma = abs(self.gamma)`. |
| 137 | forward parameter .data mutation | `self.lam.data   = torch.abs(self.lam.data)` | Do not mutate parameter; use local `lam = abs(self.lam)`. |
| 140 | forward parameter .data mutation | `self.tau.data   = torch.abs(self.tau.data)` | Do not mutate parameter; use local `tau = abs(self.tau)` when applicable. |
| 164 | autograd-unsafe tensor augmented assignment | `Dt += -tau_D*(S.T@SDt + S.T@X - S.T@x_0)` | Replace augmented assignment with mathematically equivalent out-of-place update. |

## Minimal Compatibility Diff

```diff
- self.gamma.data = torch.abs(self.gamma.data)
- self.lam.data   = torch.abs(self.lam.data)
+ gamma = torch.abs(self.gamma)
+ lam = torch.abs(self.lam)
- Dt += -tau_D*(S.T@SDt + S.T@X - S.T@x_0)
+ Dt = Dt - tau_D*(S.T@SDt + S.T@X - S.T@x_0)
```

## Forward Equivalence

- State dict keys and tensors exactly equal before forward: `True`

| Output | Max absolute error |
|---|---:|
| x0 | 0 |
| S | 0 |
| D | 0 |

- Forward peak allocated: 10.26 MiB
- Forward peak reserved: 26.00 MiB

## Official Backward Control

`FAILED: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [65536, 8]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).`

## Compat Backward

### `D.mean()`

- Loss: 0.103513732553
- Outputs finite: `True`
- Peak allocated: 195.80 MiB
- Peak reserved: 224.00 MiB
- Forward + backward time: 0.0280 s

| Tensor | requires_grad | grad is None | finite | grad norm |
|---|---:|---:|---:|---:|
| gamma | True | False | True | 6.2790636548e-06 |
| lam | True | False | True | 0.00300023169257 |
| S_init | True | False | True | 0.0121529763564 |
| INPUT | True | False | True | 0.000973757647444 |

### `D.mean()+0.01*S.mean()+0.01*x0.mean()`

- Loss: 0.115384325385
- Outputs finite: `True`
- Peak allocated: 195.80 MiB
- Peak reserved: 224.00 MiB
- Forward + backward time: 0.0291 s

| Tensor | requires_grad | grad is None | finite | grad norm |
|---|---:|---:|---:|---:|
| gamma | True | False | True | 6.49740923109e-06 |
| lam | True | False | True | 0.00311507075094 |
| S_init | True | False | True | 0.0120201641694 |
| INPUT | True | False | True | 0.00100614374969 |

## Decision

- Forward equivalence: `PASS`
- Compat gradients present and finite: `PASS`
- Official source unchanged: `PASS`
- Final: `PASS`
