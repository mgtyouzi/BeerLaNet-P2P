import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.beerlanet_p2p import BeerLaNetP2P


class DummyP2P(nn.Module):
    def forward(self, images):
        score = images.mean(dim=(1, 2, 3), keepdim=False)
        logits = torch.stack((score, -score), dim=-1).unsqueeze(1)
        points = torch.stack((score, score), dim=-1).unsqueeze(1)
        return {"pnt_coords": points, "cls_logits": logits}


def standardized_input(mean, std, size=32):
    raw = torch.rand(1, 3, size, size)
    normalized = (raw - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    return raw, normalized


def assert_finite_gradients(parameters):
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_frozen_beerlanet_preserves_contract_and_only_trains_adapter_and_p2p():
    torch.manual_seed(0)
    mean = torch.tensor([0.7, 0.5, 0.6])
    std = torch.tensor([0.1, 0.2, 0.15])
    raw, normalized = standardized_input(mean, std)
    model = BeerLaNetP2P(
        p2p=DummyP2P(),
        mean=mean,
        std=std,
        r=8,
        n_iter=2,
        train_beerlanet=False,
        gradient_checkpointing=False,
    )

    restored, _, _, d_matrix, adapted = model.forward_components(normalized)
    outputs = model(normalized)

    assert torch.allclose(restored, raw, atol=1e-6, rtol=0)
    assert d_matrix.shape == (1, 8, 32, 32)
    assert adapted.shape == (1, 3, 32, 32)
    assert torch.isfinite(d_matrix).all()
    assert torch.isfinite(adapted).all()
    assert set(outputs) == {"pnt_coords", "cls_logits"}
    assert not any(parameter.requires_grad for parameter in model.beerlanet.parameters())

    outputs["cls_logits"].sum().backward()
    assert model.adapt_conv.weight.grad is not None
    assert torch.isfinite(model.adapt_conv.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.beerlanet.parameters())


def test_joint_beerlanet_receives_finite_gradients_with_checkpointing():
    torch.manual_seed(0)
    mean = torch.tensor([0.7, 0.5, 0.6])
    std = torch.tensor([0.1, 0.2, 0.15])
    _, normalized = standardized_input(mean, std)
    model = BeerLaNetP2P(
        p2p=DummyP2P(),
        mean=mean,
        std=std,
        r=8,
        n_iter=2,
        train_beerlanet=True,
        gradient_checkpointing=True,
    )
    model.train()

    outputs = model(normalized)
    outputs["cls_logits"].sum().backward()

    assert_finite_gradients(model.beerlanet.parameters())
    assert model.adapt_conv.weight.grad is not None
    assert torch.isfinite(model.adapt_conv.weight.grad).all()


if __name__ == "__main__":
    test_frozen_beerlanet_preserves_contract_and_only_trains_adapter_and_p2p()
    test_joint_beerlanet_receives_finite_gradients_with_checkpointing()
    print("BeerLaNet-P2P integration tests passed")
