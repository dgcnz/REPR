import json
import torch
import torch.nn as nn

from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
import math


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_ind):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        if key_ind is not None:
            xkv = x[torch.arange(B).unsqueeze(1), key_ind]
            Nkv = key_ind.size(1)
        else:
            xkv = x
            Nkv = N
        k, v = (
            self.kv(xkv)
            .reshape(B, Nkv, self.num_heads, 2 * C // self.num_heads)
            .permute(0, 2, 1, 3)
            .chunk(2, dim=-1)
        )

        # q = F.normalize(q, dim=-1) * 20.
        # k = F.normalize(k, dim=-1)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, key_ind):
        dp = self.drop_path
        x = x + dp(self.attn(self.norm1(x), key_ind))
        x = x + dp(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, embed_dim, num_channels, num_patches, num_heads, query_type="positional"
    ):
        super().__init__()

        def positionalencoding2d(d_model, height, width):
            """
            :param d_model: dimension of the model
            :param height: height of the positions
            :param width: width of the positions
            :return: d_model*height*width position matrix
            """
            if d_model % 4 != 0:
                raise ValueError(
                    "Cannot use sin/cos positional encoding with "
                    "odd dimension (got dim={:d})".format(d_model)
                )
            pe = torch.zeros(d_model, height, width)
            # Each dimension use half of d_model
            d_model = int(d_model / 2)
            div_term = torch.exp(
                torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
            )
            pos_w = torch.arange(0.0, width).unsqueeze(1)
            pos_h = torch.arange(0.0, height).unsqueeze(1)
            pe[0:d_model:2, :, :] = (
                torch.sin(pos_w * div_term)
                .transpose(0, 1)
                .unsqueeze(1)
                .repeat(1, height, 1)
            )
            pe[1:d_model:2, :, :] = (
                torch.cos(pos_w * div_term)
                .transpose(0, 1)
                .unsqueeze(1)
                .repeat(1, height, 1)
            )
            pe[d_model::2, :, :] = (
                torch.sin(pos_h * div_term)
                .transpose(0, 1)
                .unsqueeze(2)
                .repeat(1, 1, width)
            )
            pe[d_model + 1 :: 2, :, :] = (
                torch.cos(pos_h * div_term)
                .transpose(0, 1)
                .unsqueeze(2)
                .repeat(1, 1, width)
            )

            return pe.permute(1, 2, 0)

        def positionalencoding1d(d_model, length):
            """
            :param d_model: dimension of the model
            :param length: length of positions
            :return: length*d_model position matrix
            """
            if d_model % 2 != 0:
                raise ValueError(
                    "Cannot use sin/cos positional encoding with "
                    "odd dim (got dim={:d})".format(d_model)
                )
            pe = torch.zeros(length, d_model)
            position = torch.arange(0, length).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, d_model, 2, dtype=torch.float)
                    * -(math.log(10000.0) / d_model)
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)

            return pe

        # num_patches, num_patches, embed_dim
        self.positional_embeddings2d = positionalencoding2d(
            embed_dim, num_patches, num_patches
        )  # [64, 64, 384]
        # num_patches, embed_dim
        self.positional_embeddings1d = positionalencoding1d(
            embed_dim, num_patches
        )  # [64, 384]

        self.num_channels = num_channels
        self.query_type = query_type
        self.input_project = nn.Linear(embed_dim * 2, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.output_project = nn.Linear(embed_dim, num_channels)
        self.num_heads = num_heads

    def forward(self, x, query):
        """
        x: [bs, num_patches, embedding_dim]
        query: [bs, num_pairs, 2]
        """
        bs, num_patches, embedding_dim = x.shape
        query = query.to(x.device)
        self.positional_embeddings2d = self.positional_embeddings2d.to(x.device)
        self.positional_embeddings1d = self.positional_embeddings1d.to(x.device)

        if self.query_type == "positional":
            # transformed_queries: [bs, num_pairs, embedding_dim]
            transformed_queries = self.positional_embeddings2d[
                query[:, :, 0], query[:, :, 1], :
            ].to(x.device)
            # transformed_queries is the same for the same i and j across all images in the batch
            # key_pos: [bs, num_patches, embedding_dim]
            key_pos = (
                self.positional_embeddings1d[range(num_patches), :]
                .repeat(bs, 1, 1)
                .to(x.device)
            )
            x = x + key_pos  # [256, 64, 384]
        elif self.query_type == "patch_cat":
            # transformed_queries_0: [bs, num_pairs, embedding_dim]
            num_pairs = query.shape[1]
            batch_indice = torch.arange(bs)[..., None].repeat(1, num_pairs)
            transformed_queries_0 = x[batch_indice, query[:, :, 0], :]
            transformed_queries_1 = x[batch_indice, query[:, :, 1], :]

            # transformed_queries: [bs, num_pairs, embedding_dim*2]
            transformed_queries = torch.cat(
                [transformed_queries_0, transformed_queries_1], dim=2
            )
            transformed_queries = self.input_project(transformed_queries)

        # q: [256, 4096, 384], x: [256, 64, 384]
        attn_output, _ = self.multihead_attn(
            query=transformed_queries, key=x, value=x
        )  # [bs, num_pairs, embedding_dim]
        output = self.output_project(attn_output)  # [bs, num_pairs, num_channels]
        return output
        # return transformed_queries


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        num_channels=2,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        mask_prob=0.0,
        pretrain=False,
        linear_probe=False,
        global_pool=False,
        use_pe=True,
        loss="CrossEntropyLoss",
        use_ce=False,
        num_pairs=64,
        with_replacement=0,
        ce_op="concat",
        column_embedding="scalar",
        head_type="mlp",
        debug_pairwise_mlp=False,
        debug_cross_attention=False,
        cross_attention_query_type="positional",
        cross_attention_num_heads=1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.depth = depth
        self.mask_prob = mask_prob
        self.pretrain = pretrain
        self.linear_probe = linear_probe
        self.global_pool = global_pool
        self.use_pe = use_pe
        self.loss = loss
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size
        self.head_type = head_type
        self.use_ce = use_ce
        self.ce_op = ce_op
        self.column_embedding = column_embedding
        # self.pairwise_mlp = pairwise_mlp
        self.num_pairs = num_pairs
        self.with_replacement = with_replacement
        self.debug_pairwise_mlp = debug_pairwise_mlp
        self.debug_cross_attention = debug_cross_attention

        embed_img_dim = embed_dim
        if self.use_ce and self.ce_op == "concat":
            embed_img_dim = embed_dim - 1

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_img_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_img_dim,
            )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        if self.use_ce:
            if self.ce_op == "add":
                if self.column_embedding == "learned":
                    self.col_embed = nn.Parameter(
                        torch.zeros(1, num_patches + 1, embed_dim)
                    )  # ([1, 65, 384])
                elif self.column_embedding == "random":
                    self.col_embed = torch.rand(
                        1, num_patches + 1, embed_dim
                    )  # ([1, 65, 384]) numbers are 0-1
                elif self.column_embedding == "one-hot":
                    self.col_embed = torch.nn.functional.one_hot(
                        torch.arange(num_patches + 1), num_classes=embed_dim
                    )  # ([65, 384])
                elif self.column_embedding == "x":
                    pass
                # else:
                #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            elif self.ce_op == "concat":
                if self.column_embedding == "random":
                    self.col_embed = torch.rand(
                        1, num_patches, 1
                    )  # shape: ([1, 64, 1]) between 0-1
                elif self.column_embedding == "one-hot":
                    pass  # not implemented
                    self.col_embed = torch.nn.functional.one_hot(
                        torch.arange(num_patches)
                    )  # shape: ([64, 64])
                elif self.column_embedding == "scalar":
                    self.col_embed = torch.arange(num_patches)[
                        :, None
                    ]  # shape: ([64, 1]) from 0-64
                elif self.column_embedding == "scalar_norm":
                    self.col_embed = (torch.arange(num_patches) / num_patches)[
                        :, None
                    ]  # shape: ([64, 1]) from 0-1
                elif self.column_embedding == "x":
                    pass

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.clf = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if self.pretrain:
            # if mp3: #TODO(change the name)
            if loss == "CrossEntropyLoss":
                self.head = nn.Linear(embed_dim, num_patches)
            elif self.use_ce:
                self.head = nn.Linear(
                    embed_dim, num_channels * num_patches
                )  # dx,dy,dw,dh
            elif self.head_type == "mlp":
                # give all patches to the linear layer
                self.head = nn.Linear(
                    num_patches * embed_dim, num_channels * num_patches * num_patches
                )
            elif self.head_type == "pairwise_mlp":
                self.head = nn.Linear(
                    embed_dim * 2, num_channels
                )  # 2 (patches) x embed_dim - > dx,dy,dw,dh
            elif self.head_type == "cross_attention":
                self.head = CrossAttention(
                    embed_dim=embed_dim,
                    num_channels=num_channels,
                    num_patches=num_patches,
                    num_heads=cross_attention_num_heads,
                    query_type=cross_attention_query_type,
                )

        self.register_buffer("targets", torch.arange(num_patches))

        if self.use_ce and self.column_embedding == "learned":
            trunc_normal_(self.col_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_param_index(self, var_name):
        """depth + 2 levels in total, return in top down order"""
        if var_name in ("cls_token", "mask_token"):
            i = 0
        elif var_name == "pos_embed":
            i = self.depth + 1
        elif var_name.startswith("patch_embed"):
            i = 0
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split(".")[1])
            i = layer_id + 1
        else:
            i = self.depth + 1

        return self.depth + 1 - i

    def get_parameter_groups(self, base_lr, weight_decay=1e-5, layer_lr_decay=1.0):
        parameter_group_names = {}  # for debugging
        parameter_group_vars = {}
        no_weight_decay = self.no_weight_decay()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (
                len(param.shape) == 1
                or name.endswith(".bias")
                or name in no_weight_decay
            ):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            layer_id = self.get_param_index(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_group_vars:
                lr = base_lr * (layer_lr_decay**layer_id)

                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": lr,
                }
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": lr,
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        return list(parameter_group_vars.values())

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        """in the pretraining mode, we remove the positions embeddings, and select a subset of patches as context.
        Only the context pacthes server as keys and values. All patches attented to the context patches and make
        predictions about their positions.
        """
        if self.use_ce and self.column_embedding == "x":
            assert False  # not implemented

        B = x.shape[0]
        x = self.patch_embed(x)
        # x.shape: torch.Size([256, 64, 383])
        # batch_ind = torch.arange(B).unsqueeze(1)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        if self.pretrain:
            B = x.size(0)
            # we first generate a list of indices per example, key_ind, of shape (B, n_keys)
            n_keys = int(self.num_patches * (1 - self.mask_prob))
            _, ind = torch.randn(B, self.num_patches).to(x.device).sort(dim=-1)
            # ind = torch.stack([torch.randperm(self.num_patches) for _ in range(B)], dim=0).to(x.device)
            cls_ind = (torch.ones(B, 1) * self.num_patches).to(x.device).long()
            # the class token is always included in key_ind, treated as the last token
            key_ind = torch.cat(
                [ind[:, :n_keys], cls_ind], dim=1
            )  # this line is masking the rest of keys with the num_patches
            if self.use_ce:
                if self.ce_op == "add":
                    x = x + self.col_embed.to(
                        x.device
                    )  # self.col_embed: torch.Size([65, 384])
                elif self.ce_op == "concat":
                    x = torch.cat(
                        (x, self.col_embed.to(x.device).repeat(B, 1, 1)), dim=2
                    )  # Size([256, 64, 384])
            x = torch.cat((x, cls_tokens), dim=1)  # x.shape: torch.Size([256, 65, 384])
            # query_ind = ind[:, n_keys:]
            # pos_embed = self.pos_embed.expand(B, -1, -1)
            # x[batch_ind, key_ind] += pos_embed[batch_ind, key_ind]

            # we do not use position embedding during pretraining
        else:
            if self.use_ce and self.ce_op == "concat":
                x = torch.cat(
                    (x, self.col_embed.to(x.device).repeat(B, 1, 1)), dim=2
                )  # Size([256, 65, 385])

            # x = x + pos_embed
            x = torch.cat((x, cls_tokens), dim=1)
            # during finetuning, we add the position embeddings back
            if self.use_pe:
                x = x + self.pos_embed
            key_ind = None
            # query_ind = None

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, key_ind)

        if self.pretrain:
            x = self.norm(x[:, :-1])
            # x = self.norm(x[batch_ind, query_ind])
            # x = self.norm(x[batch_ind.squeeze(1), :-1, :])
            # targets = query_ind
            # return x, targets
        elif self.global_pool:
            x = self.norm(x[:, :-1].mean(dim=1))
        elif self.no_pool:
            x = self.norm(x[:, :-1].flatten(1))
        else:
            x = self.norm(x[:, -1])
        return x
    
    def forward_features_pretrain(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, None)
        x = self.norm(x)
        return x

    def forward(self, x):
        if self.pretrain:
            if self.loss == "CrossEntropyLoss":
                targets = self.targets.repeat(
                    x.size(0)
                )  # x.size(0) = batch_size, torch.Size([16384])
                # x, targets = self.forward_features(x)  # x.shape = torch.Size([256, 64, 384])
                x = self.forward_features(x)  # x.shape = torch.Size([256, 64, 384])
                x = self.head(x).flatten(0, 1)
                # targets = targets.flatten().to(x.device)
                return x, targets
            # all other regression losses
            else:
                # # targets_x.shape and targets_y.shape = [256, 64, 64]
                # targets_x, targets_y = self.get_targets()
                # # repeating the target along batch size
                # targets_x, targets_y = targets_x.repeat(x.size(0), 1, 1), targets_y.repeat(x.size(0), 1, 1)
                # # (targets_y + torch.transpose(targets_y, dim0=2, dim1=1) == 0).all() == True
                #
                # # concatenating the target_x and target_y into the same matrix with shape [256, 2, 64, 64] delta x & y
                # targets = torch.cat((targets_x[:, None, :, :], targets_y[:, None, :, :]), dim=1)

                x = self.forward_features(x)  # x.shape = torch.Size([256, 64, 384])
                b, num_patches, embed_dim = x.shape
                # reshape and randomize x and give it to head
                if self.head_type == "pairwise_mlp":
                    # choose N indexes of patches to concat to the first N patches
                    N = self.num_pairs  # num_pairs

                    if self.debug_pairwise_mlp:
                        # todo: a specific case for debugging pairwise mlp
                        N = 64
                        indices = torch.arange(64)
                        indices = torch.cat(
                            (
                                indices.repeat_interleave(N)[None, :],
                                indices.repeat(N)[None, :],
                            ),
                            dim=0,
                        )
                        # patch_pairs with these indices-> [256, 4096, 768]
                    else:
                        # with replacement:
                        if self.with_replacement:
                            indices = torch.randint(num_patches, (N,))
                            indices = torch.cat(
                                (
                                    indices[None, :],
                                    torch.randint(num_patches, (N,))[None, :],
                                ),
                                dim=0,
                            )  # [2, N]
                        # without replacement:
                        else:
                            indices = torch.arange(N)  # [N]
                            indices = torch.cat(
                                (
                                    indices[None, :],
                                    torch.randperm(num_patches)[:N][None, :],
                                ),
                                dim=0,
                            )  # [2, N]

                    x = torch.cat(
                        (x[:, indices[0], :], x[:, indices[1], :]), dim=2
                    )  # ([256, N, 2*384])
                    # todo: another variation is to return 4 x 4 for (i,i), (i,j), (j,i), (j,j) for N=2
                    x = self.head(x)  # [256, N, 4]
                    x = x.permute(0, 2, 1)  # x.shape = [256, num_channels, num_pairs]
                    return (
                        x,
                        indices,
                    )  # indices will index the targets for the loss indices: [2, num_pairs]
                elif self.head_type == "cross_attention":
                    N = self.num_pairs  # num_pairs
                    # todo: a specific case for debugging cross attention with v1 unshuffled
                    if self.debug_cross_attention:
                        indices = torch.arange(num_patches * num_patches)[
                            :, None
                        ]  # [64x64 ,1]
                    else:
                        indices = torch.randperm(num_patches * num_patches)[:N][
                            :, None
                        ]  # [N, 1]

                    x_indices, y_indices = indices % num_patches, indices // num_patches
                    # same index across all batches (patches are sampled randomly anyway)
                    indices = torch.cat((y_indices, x_indices), dim=1).repeat(
                        b, 1, 1
                    )  # [bs, num_pairs, 2]
                    # x:[b, num_pairs, 2] indices: [b, num_pairs, 2]
                    x = self.head(x, indices)  # x: [256, num_pairs, num_channels]
                    # bs, num_patches, embedding_dim = x.shape
                    # query = indices.to(x.device)
                    # num_pairs = query.shape[1]
                    # batch_indice = torch.arange(bs)[..., None].repeat(1, num_pairs)
                    # transformed_queries_0 = x[batch_indice, query[:, :, 0], :]
                    # transformed_queries_1 = x[batch_indice, query[:, :, 1], :]

                    # transformed_queries: [bs, num_pairs, embedding_dim*2]
                    # x = torch.cat([x[:, query[:, 0], :], x[:, query[:, 1], :]], dim=2)
                    # transformed_queries = nn.Linear(embedding_dim * 2, embedding_dim, device=x.device)(transformed_queries)
                    # x = nn.Linear(embedding_dim * 2, self.num_channels, device=x.device)(x)

                    x = x.permute(0, 2, 1)  # [b, num_channels, num_pairs]
                    return x, indices.permute(0, 2, 1)[
                        0, :, :
                    ]  # x: [b, num_channels, num_pairs], indices: [2, num_pairs]

                else:
                    if self.use_ce:
                        x = self.head(
                            x
                        )  # self.head(x).shape = torch.Size([256, 64, 64*num_channels])
                    else:
                        x = self.head(x.reshape(x.shape[0], -1))

                    x = x.reshape(
                        b, num_patches, num_patches, self.num_channels
                    )  # [256, 64, 64, num_channels]
                    x = x.permute(0, 3, 1, 2)  # [256, num_channels, 64, 64]

                    return x

        else:
            if self.linear_probe:
                with torch.no_grad():
                    x = self.forward_features(x)
            else:
                x = self.forward_features(x)
            x = self.clf(x)
        return x

    def get_targets_pairwise_x(self):
        # t is the number of patches per width or height of an image
        t = int(math.sqrt(self.num_patches))
        a1 = torch.arange(t)
        return (
            torch.cat([a1.repeat(t) - i for i in range(t)], dim=0)
            .view(t, t**2)
            .repeat(t, 1)
        )

    def get_targets_pairwise_y(self):
        t = int(math.sqrt(self.num_patches))
        a1 = torch.arange(t)
        return torch.repeat_interleave(
            torch.repeat_interleave(
                torch.cat([a1 - i for i in range(t)], dim=0).view(t, t), t, dim=0
            ),
            t,
            dim=1,
        )

    def get_targets(self):
        targets_x = self.get_targets_pairwise_x().to(
            self.targets.device, non_blocking=True, dtype=torch.float
        )
        targets_y = self.get_targets_pairwise_y().to(
            self.targets.device, non_blocking=True, dtype=torch.float
        )
        return targets_x, targets_y


def partvit_base_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = VisionTransformer(
        **model_args,
        **kwargs,
    )
    return model


def partvit_small_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = VisionTransformer(
        **model_args,
        **kwargs,
    )
    return model



if __name__ == "__main__":
    # Example usage
    import timm
    # model = timm.create_model(
    #     "partvit_base_patch16_224",
    #     pretrained=False,
    model = partvit_base_patch16_224(
        num_channels=2,
        num_classes=1000,
        use_ce=True,
        ce_op="concat",
        pretrain=True,
        column_embedding="scalar",
        head_type="cross_attention",
        cross_attention_query_type="positional",
        cross_attention_num_heads=1,
    )

    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Forward pass
    output = model.forward_features(input_tensor)
    print(output.shape)  # Should print the shape of the output tensor