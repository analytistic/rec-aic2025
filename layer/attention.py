import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, att_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.att_units = att_units
        self.num_heads = num_heads
        self.head_dim = att_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, att_units)
        self.k_linear = torch.nn.Linear(hidden_units, att_units)
        self.v_linear = torch.nn.Linear(hidden_units, att_units)
        self.out_linear = torch.nn.Linear(att_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.att_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)


        return output, None
    
class FlashGroupedAttention(torch.nn.Module):
    """
    GQA实现
    同时按照qwen2025， gated attention for language models， 在concat后添加门控

    """
    def __init__(self, hidden_units, att_units, num_heads, dropout_rate, num_kv_heads=None):
        super(FlashGroupedAttention, self).__init__()

        self.hidden_units = hidden_units
        self.att_units = att_units
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads // 2
        self.head_dim = att_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.n_repeat = self.num_heads // self.num_kv_heads

        self.q_linear = torch.nn.Linear(hidden_units, att_units)
        self.k_linear = torch.nn.Linear(hidden_units, att_units//self.n_repeat)
        self.v_linear = torch.nn.Linear(hidden_units, att_units//self.n_repeat)
        self.out_linear = torch.nn.Linear(att_units, hidden_units)
        self.gate_linear = torch.nn.Linear(att_units, att_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        K = K.repeat_interleave(self.n_repeat, dim=1)
        V = V.repeat_interleave(self.n_repeat, dim=1)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.att_units)

        # 门控非线性映射
        attn_output = attn_output * torch.sigmoid(self.gate_linear(attn_output))

        # 最终的线性变换
        output = self.out_linear(attn_output)


        return output, None
    
class TimeIntervalAwareSelfAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, embeddings={}):
        super(TimeIntervalAwareSelfAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        self.E_PK = embeddings["E_PK"]
        self.E_PV = embeddings["E_PV"]
        self.E_RK = embeddings["E_RK"]
        self.E_RV = embeddings["E_RV"]

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, poss_seq, interval_seq, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query) 
        K = self.k_linear(key) + self.E_PK(poss_seq)
        V = self.v_linear(value) + self.E_PV(poss_seq)

        rel_emb_k = self.E_RK(interval_seq).view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, H, L, L, D]
        rel_emb_v = self.E_RV(interval_seq).view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算位置注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        # 计算时间间隔注意力
        rel_attn = torch.einsum('bhld,bhlkd->bhlk', Q, rel_emb_k)
        attn_scores = (attn_scores + rel_attn) / (self.head_dim ** 0.5)


        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1).masked_fill(~(attn_mask.unsqueeze(1)), 0.0)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)

        # 计算V
        attn_output = torch.matmul(attn_weights, V)
        rel_attn_output = torch.einsum('bhlk,bhlkd->bhld', attn_weights, rel_emb_v)

        attn_output = (attn_output + rel_attn_output).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        attn_output = self.out_linear(attn_output)
        attn_output = torch.where(attn_output.isnan(), 0, attn_output)  # 处理NaN
        return attn_output, None
  







class PointwiseAggregatedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TODO: add relative attention bias based on time
        # self.rab_p = RelativeAttentionBias(num_heads, relative_attention_num_buckets=32,
        #                                    relative_attention_max_distance=128)
        self.rab_t = MatrixBasedAttentionBias(num_heads, relative_attention_num_buckets=20)
        # self.rab_r = MatrixBasedAttentionBias(num_heads, relative_attention_num_buckets=5)
        # self.rab_p = MatrixBasedAttentionBias(num_heads, relative_attention_num_buckets=128)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None, diff_matrix=None, renew_seq=None, pos_diff_matrix=None):
        batch_size = q.shape[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)


        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        rab_t = self.rab_t(diff_matrix, device=q.device)
        # rab_p = self.rab_p(pos_diff_matrix, device=q.device)
        # rab_r = self.rab_r(renew_seq, device=q.device)


        # att_w_bias = attention_scores
        att_w_bias = attention_scores + rab_t

        att_w_bias = F.silu(att_w_bias).masked_fill(mask.unsqueeze(1).logical_not(), float(0) )

        av = (att_w_bias @ v)
        return av.transpose(1, 2).flatten(2)
    

class MatrixBasedAttentionBias(nn.Module):
    def __init__(self, num_heads, relative_attention_num_buckets, relative_attention_max_distance=128):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    def forward(self, bucket_matrix, device=None):

        if device is None:
            device = self.relative_attention_bias.weight.device

        bucket_matrix = bucket_matrix.to(device)
        
        values = self.relative_attention_bias(bucket_matrix)  # 形状 [bs, seq_len, seq_len, num_heads]
        values = values.permute([0, 3, 1, 2])  # 形状 [bs, num_heads, seq_len, seq_len]
        return values


class RelativeAttentionBias(nn.Module):
    def __init__(self, num_heads, relative_attention_num_buckets, relative_attention_max_distance=128):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    def forward(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    # https://github.com/huggingface/transformers/blob/6cdbd73e01a9719bfaec07d91fd108e8d932bbbb/src/transformers/models/t5/modeling_t5.py#L384
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
class HSTUBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_model * 4)  # Transform and split
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, num_heads)
        self.f2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def split(self, x):
        u, v, q, k = x.chunk(4, dim=-1)
        return u, v, q, k

    def forward(self, x, mask=None, diff_matrix=None, renew_seq=None, pos_diff_matrix=None):
        # Pointwise Projection
        x_proj = F.silu(self.f1(x))
        u, v, q, k = self.split(x_proj)

        # Spatial Aggregation
        av = self.pointwise_attn(v, k, q, mask=mask, diff_matrix=diff_matrix, renew_seq=renew_seq, pos_diff_matrix=pos_diff_matrix)

        # Pointwise Transformation
        y = self.f2(self.norm(av * u))

        return y











