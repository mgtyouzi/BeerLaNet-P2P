import torch
import torch.nn as nn
import torch.nn.functional as F


class BeerLaNetAutogradCompat(nn.Module):
    """Autograd-safe implementation of the official BeerLaNet equations."""

    def __init__(self, r, c=3, learn_S_init=False, calc_tau=True):
        super().__init__()
        self.r = r
        self.c = c
        self.calc_tau = calc_tau
        self.gamma = nn.Parameter(torch.rand(1) * 1e-5)
        self.lam = nn.Parameter(torch.rand(1) * 1e-5)
        if not calc_tau:
            self.tau = nn.Parameter(torch.rand(1) * 1e-5)
        if learn_S_init:
            self.S_init = nn.Parameter(torch.rand(self.c, self.r))
            with torch.no_grad():
                self.S_init.copy_(self.S_init / self._S_norm(self.S_init))
        else:
            self.S_init = None

    def forward(self, X, S=None, D=None, n_iter=1, unit_norm_S=True):
        n, c_in, p1, p2 = X.shape
        p = p1 * p2
        if c_in != self.c:
            raise ValueError(f"expected {self.c} channels, got {c_in}")
        X = X.view(-1, self.c, p)
        if S is None:
            if self.S_init is None:
                raise ValueError("S must be provided when learn_S_init=False")
            S = self.S_init.clone().to(X.device)
        if D is None:
            Dt = torch.zeros(n, self.r, p, device=X.device, dtype=X.dtype)
        else:
            Dt = D.view(-1, self.r, p)

        gamma = torch.abs(self.gamma)
        lam = torch.abs(self.lam)
        tau_parameter = None if self.calc_tau else torch.abs(self.tau)

        for _ in range(n_iter):
            SDt = S @ Dt
            x_0 = torch.mean(X + SDt, dim=2, keepdims=True)
            if self.calc_tau:
                tau_D = 1.0 / torch.linalg.matrix_norm(S, ord="fro") ** 2
            else:
                tau_D = tau_parameter
            Dt = Dt - tau_D * (S.T @ SDt + S.T @ X - S.T @ x_0)

            S_nrm = self._S_norm(S).view(1, self.r, 1)
            Dt = F.relu(Dt - lam * gamma * tau_D * S_nrm)
            Dt_L2 = self._Dt_Lp(Dt, 2)
            scl = F.relu(Dt_L2 - lam * tau_D * S_nrm)
            scl = scl / Dt_L2 + 1e-10
            Dt = Dt * scl

            SDt = S @ Dt
            x_0 = torch.mean(X + SDt, dim=2, keepdims=True)
            Dt_sum = Dt.sum(dim=2, keepdim=True)
            if self.calc_tau:
                tau_S = 1.0 / torch.mean(torch.linalg.matrix_norm(Dt, ord="fro") ** 2)
            else:
                tau_S = tau_parameter
            S_grad = S - tau_S * torch.mean(
                SDt @ Dt.permute(0, 2, 1)
                + X @ Dt.permute(0, 2, 1)
                - x_0 @ Dt_sum.permute(0, 2, 1),
                dim=0,
                keepdims=False,
            )

            Dt_nrm = self._Dt_norm(Dt, gamma).mean(dim=0, keepdims=False)
            S_nrm = self._S_norm(S_grad)
            scl_S = F.relu(S_nrm - lam * tau_S * Dt_nrm.T)
            scl_S = scl_S / (S_nrm + 1e-10)
            S = S_grad * scl_S
            if unit_norm_S:
                S_nrm = self._S_norm(S)
                S = S / (S_nrm + 1e-10)
                Dt = Dt * (S_nrm.view(1, self.r, 1) + 1e-10)

        return x_0, S, Dt.view(n, self.r, p1, p2)

    @staticmethod
    def _S_norm(S):
        return torch.linalg.vector_norm(S, ord=2, dim=0, keepdim=True)

    @staticmethod
    def _Dt_Lp(Dt, nrm_ord):
        return torch.linalg.vector_norm(Dt, ord=nrm_ord, dim=2, keepdim=True)

    def _Dt_norm(self, Dt, gamma):
        return gamma * self._Dt_Lp(Dt, 1) + self._Dt_Lp(Dt, 2)
