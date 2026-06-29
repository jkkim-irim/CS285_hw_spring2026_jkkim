"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        ### 직접 구현한 부분
        output_dim = chunk_size * action_dim
        dims = [state_dim, *hidden_dims, output_dim]

        layers = []
        for i in range(len(dims) - 1):
             layers.append(nn.Linear(dims[i], dims[i+1]))
             if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        ### 직접 구현한 부분

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        ### 직접 구현한 부분
        pred = self.net(state)
        pred = pred.reshape(-1, self.chunk_size, self.action_dim)
        return ((pred-action_chunk)**2).mean()
        ### 직접 구현한 부분

    

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        ### 직접 구현한 부분    
        pred = self.net(state)
        pred = pred.reshape(-1, self.chunk_size, self.action_dim)
        return pred
        ### 직접 구현한 부분


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        ### 직접 구현한 부분
        input_dim = state_dim + (chunk_size*action_dim) + 1
        output_dim = chunk_size * action_dim
        dims = [input_dim, *hidden_dims, output_dim]

        layers = []
        for i in range(len(dims) - 1):
             layers.append(nn.Linear(dims[i], dims[i+1]))
             if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        ### 직접 구현한 부분        

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        A = action_chunk
        A_0 = torch.randn_like(action_chunk)
        batch = action_chunk.shape[0]

        tau = torch.rand(batch, 1, 1, device=action_chunk.device)
        tau_2d = tau.reshape(batch, 1)
        A_tau = (tau*A) + (1-tau)*A_0
        A_tau_flat = A_tau.reshape(batch, -1)
        input = torch.cat([state,A_tau_flat,tau_2d], dim=-1)

        target = A - A_0
        v_pred = self.net(input)
        v_pred = v_pred.reshape(-1, self.chunk_size, self.action_dim)

        return ((v_pred-target)**2).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch = state.shape[0]
        A = torch.randn(batch, self.chunk_size, self.action_dim, device=state.device)
        dt = 1 / num_steps
        for i in range(num_steps):
            tau_val = i * dt
            tau = torch.full((batch, 1), tau_val, device=state.device)
            input = torch.cat([state, A.reshape(batch, -1), tau], dim=-1)
            v = self.net(input)
            v = v.reshape(batch, self.chunk_size, self.action_dim)
            A = A + dt * v
        return A


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
