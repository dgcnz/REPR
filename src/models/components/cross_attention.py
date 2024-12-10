import torch.nn as nn
import torch
import math


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        num_patches: int,
        num_heads: int,
        query_type: str = "positional",
    ):
        super().__init__()

        def positionalencoding2d(d_model: int, height: int, width: int):
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

    def forward(self, x: torch.Tensor, query: torch.Tensor):
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
