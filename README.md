# BeerLaNet-P2P

Research codebase for source-only FFPE-to-frozen domain-generalized signet-ring cell point detection.

This repository starts from the clean P2P detector implementation. BeerLaNet integration will be developed here while keeping datasets, model weights, checkpoints, and experiment outputs outside Git.

## Intended pipeline

```text
FFPE image -> BeerLaNet stain normalization -> P2P backbone/FPN -> point detection head
```

Training uses FFPE source data only; frozen tissue is reserved for out-of-domain evaluation.

## Data policy

No pathology images, annotations, checkpoints, generated samples, or experiment logs are tracked in this repository. Configure dataset and checkpoint paths locally on the training server.
