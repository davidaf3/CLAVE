import copy
import torch
from torch import nn, Tensor
from tokenizer import PADDING_TOK
from config import MAX_SEQ_LENGTH


class Encoder(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Encoder"
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LENGTH, d_model)
        self.embedding_layer_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        padding_mask = src.eq(PADDING_TOK)
        not_padding = (~padding_mask).int()
        position_idxs = (torch.cumsum(not_padding, dim=1).int() - 1) * not_padding
        embeddings = self.embedding(src) + self.pos_embedding(position_idxs)
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        output = self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        output = src
        use_nested = not self.layers[0].training
        if use_nested:
            output = torch._nested_tensor_from_mask(
                output, src_key_padding_mask.logical_not(), mask_check=False
            )
            src_key_padding_mask = None

        for mod in self.layers:
            output = mod(
                output, is_causal=False, src_key_padding_mask=src_key_padding_mask
            )

        return (
            output.to_padded_tensor(float(PADDING_TOK), src.size())
            if use_nested
            else output
        )
