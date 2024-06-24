import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformer import Encoder


class PretrainingModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_type = "PretrainingModel"
        self.encoder = Encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.linear = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        output = self.encoder(src)
        output = self.linear(output)
        return output


class FineTunedModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        d_out: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        pretrained_weights: dict | None = None,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.model_type = "FineTunedModel"
        self.encoder = Encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.linear = nn.Linear(d_model, d_out)
        self.activation = nn.ReLU()
        self.init_weights(pretrained_weights)

    def init_weights(self, pretrained_weights: dict | None) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        if pretrained_weights is not None:
            self.encoder.load_state_dict(pretrained_weights)
        else:
            self.encoder.init_weights()

    def forward(self, src: Tensor) -> Tensor:
        output = self.encoder(src)
        output = torch.movedim(output, (1, 2), (2, 1))
        output = F.avg_pool1d(output, kernel_size=output.size(2))
        output = output.squeeze(2)
        if self.layer_norm:
            output = self.layer_norm(output)
        output = self.linear(output)
        output = self.activation(output)
        return output
