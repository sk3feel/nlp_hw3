from torch import nn

def share_layers(model, blocks=3):
    layers = model.bert.encoder.layer
    shared = []
    for i in range(len(layers)):
        shared.append(layers[i % blocks])

    model.bert.encoder.layer = nn.ModuleList(shared)
    return model

from copy import deepcopy

class LoRA(nn.Module):
    def __init__(self, linear, rank=8):
        super().__init__()
        self.linear = linear
        self.A = nn.Linear(linear.in_features, rank, bias=False)
        self.B = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.linear(x) + self.B(self.A(x))

def add_lora(model, rank=8):
    model.requires_grad_(False)
    layers = []

    for layer in model.bert.encoder.layer:
        weights = {id(p): p for p in layer.parameters()}
        layer = deepcopy(layer, weights)
        attn = layer.attention.self
        attn.query = LoRA(attn.query, rank)
        attn.value = LoRA(attn.value, rank)
        layers.append(layer)

    model.bert.encoder.layer = nn.ModuleList(layers)
    return model
