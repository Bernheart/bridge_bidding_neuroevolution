import copy
import torch
import torch.nn as nn
import src.utils.globals as g


class SoftmaxWithTempMask(nn.Module):
    def __init__(self, temp: float = 2.0):
        super().__init__()
        self.temp = temp

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: same shape as logits, True = available
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask, logits, neg_inf)
        # temperature scaling
        scaled = masked_logits / self.temp
        # subtract max for numerical stability
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        exps = torch.exp(scaled) * mask.to(logits.dtype)
        return exps / exps.sum(dim=-1, keepdim=True)


class NeuralNetTorch(nn.Module):
    def __init__(self,
                 input_size: int = g.INPUT_SIZE,
                 hidden_layers: list[int] = g.HIDDEN_LAYERS,
                 output_size: int = g.OUTPUT_SIZE,
                 temp: float = 1.0):
        super().__init__()
        layer_sizes = [input_size] + hidden_layers + [output_size]

        # build a ModuleList of linears
        self.linears = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            lin = nn.Linear(in_f, out_f)
            # Xavier‐uniform initialization
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.softmax_mask = SoftmaxWithTempMask(temp)

    def forward(self, x: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
        """
        x: float tensor of shape (input_size,)
        availability_mask: boolean tensor of shape (output_size,)
        returns one‐hot action tensor of shape (output_size,)
        """
        a = x
        for lin in self.linears[:-1]:
            a = self.activation(lin(a))
        logits = self.linears[-1](a)
        probs = self.softmax_mask(logits, availability_mask)
        # sample an action
        action = torch.multinomial(probs, num_samples=1).item()
        one_hot = torch.zeros_like(probs)
        one_hot[action] = 1.0
        return one_hot

    def clone(self) -> "NeuralNetTorch":
        # deep copy parameters into a fresh instance
        new = NeuralNetTorch(
            input_size=self.linears[0].in_features,
            hidden_layers=[L.out_features for L in self.linears[:-1]],
            output_size=self.linears[-1].out_features,
            temp=self.softmax_mask.temp
        )
        new.load_state_dict(copy.deepcopy(self.state_dict()))
        return new

    def mutate(self, rate: float = g.MUTATION_RATE, sigma: float = g.MUTATION_SCALE):
        """
        Mutate each parameter with probability `rate`, adding N(0, sigma^2) noise.
        """
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < rate
                noise = torch.randn_like(param) * sigma
                param.add_(mask.to(param.dtype) * noise)

    def crossover(self, other: "NeuralNetTorch") -> "NeuralNetTorch":
        """
        Create a child by randomly picking each parameter element from self or other.
        """
        child = self.clone()
        child_state = child.state_dict()
        other_state = other.state_dict()
        new_state = {}
        for key in child_state:
            p1 = child_state[key]
            p2 = other_state[key]
            mask = torch.rand_like(p1) < 0.5
            new_state[key] = torch.where(mask, p1, p2)
        child.load_state_dict(new_state)
        return child

    def get_parameters(self) -> dict:
        # return CPU‐numpy lists
        return {
            'weights': [p.detach().cpu().numpy().tolist()
                        for name, p in self.named_parameters() if "weight" in name],
            'biases': [p.detach().cpu().numpy().tolist()
                       for name, p in self.named_parameters() if "bias" in name]
        }

    def set_parameters(self, params: dict):
        # assumes same ordering as get_parameters
        with torch.no_grad():
            idx_w = idx_b = 0
            for name, p in self.named_parameters():
                if "weight" in name:
                    tensor = torch.tensor(params['weights'][idx_w], dtype=p.dtype)
                    p.copy_(tensor)
                    idx_w += 1
                elif "bias" in name:
                    tensor = torch.tensor(params['biases'][idx_b], dtype=p.dtype)
                    p.copy_(tensor)
                    idx_b += 1
