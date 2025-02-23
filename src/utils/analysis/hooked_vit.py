from collections import defaultdict
import torch
import logging

logging.basicConfig(level=logging.INFO)


def get_layer(model, layer_name: str):
    layers = layer_name.split(".")
    for layer in layers:
        if layer.isnumeric():
            model = model[int(layer)]
        elif hasattr(model, layer):
            model = getattr(model, layer)
        else:
            raise ValueError(f"Layer {layer} not found in {model}")
    return model


def clean_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, tuple):
        return tuple(clean_tensor(t) for t in tensor)
    elif isinstance(tensor, list):
        return [clean_tensor(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: clean_tensor(v) for k, v in tensor.items()}
    else:
        return tensor


class HookedViT(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, head_name: str="head"):
        super().__init__()
        self.clear()
        self.hooks = {}
        self.model = model.eval()
        self.head_name = head_name
        self.logger = logging.getLogger(__name__)

        self.hook(model)

    def forward(self, x):
        self.clear()
        with torch.no_grad():
            self.model(x)

        return {
            "attn_fts": self.get_attn_fts(),
            "attns": self.get_attns(),
            "zs": self.get_zs(),
        }

    def clear(self):
        self.cache = defaultdict(dict)

    def _hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            self.cache[layer_name]["input"] = clean_tensor(input)
            self.cache[layer_name]["output"] = clean_tensor(output)

        return hook_fn

    def hook_layer(self, model, layer_name: str):
        layer = get_layer(model, layer_name)
        self.logger.debug(f"Hooking {layer_name}: {layer.__class__}")
        hook = layer.register_forward_hook(self._hook_fn(layer_name))
        return hook

    def hook(self, model):
        # get Attention params: H, D
        self.H = model.blocks[0].attn.num_heads
        self.D = model.blocks[0].attn.proj.weight.shape[0]  # (D, D)

        # hook the layers
        self.n_blocks = len(model.blocks)
        for i in range(self.n_blocks):
            # deactivate fused_attn to get access to the individual components
            model.blocks[i].attn.fused_attn = False
            for layer_name in self._hooked_layers_per_block(i):
                self.hooks[layer_name] = self.hook_layer(model, layer_name)

        # hook the head 
        self.hooks[self.head_name] = self.hook_layer(model, self.head_name)
        # self.hooks[''] = model.register_forward_hook(self._hook_fn(''))

    def get_attn_ft(self, block_idx: int):
        proj_input = self.cache[f"blocks.{block_idx}.attn.proj"]["input"][0]
        B, N, _ = proj_input.shape
        attn_ft = proj_input.reshape(B, N, self.H, self.D // self.H).transpose(1, 2)
        return attn_ft

    def get_attn(self, block_idx: int):
        return self.cache[f"blocks.{block_idx}.attn.attn_drop"]["output"]
    
    def get_attn_fts(self):
        attn_fts = []
        for idx in range(self.n_blocks):
            attn_fts.append(self.get_attn_ft(idx))
        return torch.stack(attn_fts)
    
    def get_attns(self):
        attns = []
        for idx in range(self.n_blocks):
            attns.append(self.get_attn(idx))
        return torch.stack(attns)

    def get_zs(self):
        # zs: 0 = input of first block, same as input of first norm1
        # zs: i = input of norm2 of block i // 2 + 1
        # zs: i+1 = output of block i // 2 + 2, same as input of norm1 of the next block :0
        # zs: n = output of head, same as output of model
        zs = []
        zs.append(self.cache["blocks.0"]["input"][0])
        for idx in range(self.n_blocks):
            """
            class Block:
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                    ---> hook on this x
                    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                    ---> hook on this x
                    return x
            """
            z1 = self.cache[f"blocks.{idx}.norm2"]["input"][0]
            # zs.append(self.cache[f'blocks.{idx}']['output'])
            #   This works too for standard timm ViTs,
            #   but not for custom Blocks that output multiple tensors,
            #   like the cl_vs_mim ViT
            mlp_output = self.cache[f"blocks.{idx}.mlp"]["output"]
            z2 = mlp_output + z1
            zs.extend([z1, z2])

        # TODO: careful, this could change depending on the model
        # zs.append(self.cache['']['output'])
        try: 
            zs.append(self.cache[self.head_name]["output"])
        except KeyError:
            self.logger.warning(f"Head output not found in cache. Appending None.")
            zs.append(None)
        return zs

    @staticmethod
    def _hooked_layers_per_block(i):
        return [
            f"blocks.{i}",  # input -> zs[0]
            f"blocks.{i}.norm2",  # input -> zs[i]
            f"blocks.{i}.mlp",  # output + zs[i] -> zs[i+1]
            f"blocks.{i}.attn.attn_drop",  # output -> attn
            f"blocks.{i}.attn.proj",  # input -> attn_ft
        ]

