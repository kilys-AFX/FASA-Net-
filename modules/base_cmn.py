from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .att_model_cmn import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix, context_vec=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix, context_vec=context_vec)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None, context_vec=None):
        embeddings = self.tgt_embed(tgt)  # batchsize, sequence_length, 512

        # memory_matrix: 2048, 512

        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0), memory_matrix.size(1))

        # dummy_memory_matrix: batchsize, 2048, 512
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)

        # responses: batchsize, sequence_length, 512
        embeddings = embeddings + responses
        # Memory querying and responding for textual features

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past, context_vec=context_vec)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask): # x: batchsize, 98, 512  mask: batchsize, 1, 98
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None, context_vec=None):
        # x: batchsize, sequence_length, 512
        # memory: batchsize, 98, 512
        # src_mask: batchsize, 1, 98
        # tgt_mask: batchsize, sequence_length, sequence_length
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            # 判断是否是CCADecoderLayer
            if hasattr(layer, 'ctx_attn'):
                x = layer(x, memory, src_mask, tgt_mask, context_vec=context_vec, layer_past=layer_past)
            else:
                x = layer(x, memory, src_mask, tgt_mask, layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class ContextEncoder(nn.Module):
    """
    CCA模块的上下文编码器 (来自DSCI论文)
   报告中提取全局上下文 用于从医学向量，帮助解码器更好地理解整体语义
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(ContextEncoder, self).__init__()
        self.d_model = d_model
        # 标准Transformer encoder层用于处理文本嵌入
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 投影层用于将文本嵌入映射到上下文空间
        self.text_proj = nn.Linear(d_model, d_model)

    def forward(self, text_embeddings):
        """
        text_embeddings: (batch_size, seq_len, d_model) 文本嵌入序列
        返回: (batch_size, d_model) 全局上下文向量
        """
        # 对文本序列进行编码
        encoded = self.text_encoder(text_embeddings)

        # 计算全局上下文向量 (对所有token取平均)
        context = encoded.mean(dim=1)  # (batch_size, d_model)

        # 投影并返回
        return self.text_proj(context)


class CCADecoderLayer(nn.Module):
    """
    支持CCA的解码器层
    在标准解码器层基础上增加了对全局上下文向量的注意力
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(CCADecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

        # CCA: 新增的上下文注意力层
        self.ctx_attn = MultiHeadedAttention(size // 4, size, dropout)
        self.ctx_proj = nn.Linear(size, size)

    def forward(self, x, memory, src_mask, tgt_mask, context_vec=None, layer_past=None):
        m = memory
        if layer_past is None:
            # 标准自注意力
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            # 跨模态注意力 (对图像特征)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

            # CCA: 融入全局上下文向量
            if context_vec is not None:
                # 将上下文向量扩展到与x相同的序列长度
                ctx_expanded = context_vec.unsqueeze(1).expand(-1, x.size(1), -1)
                # 上下文注意力
                ctx_attn_output = self.ctx_attn(x, ctx_expanded, ctx_expanded)
                ctx_attn_output = self.ctx_proj(ctx_attn_output)
                x = x + ctx_attn_output

            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaseCMN(AttModel):

    def make_model(self, tgt_vocab, cmn, use_cca=False):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)

        # 创建解码器层
        if use_cca:
            decoder_layer = CCADecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout)
        else:
            decoder_layer = DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout)

        decoder = Decoder(decoder_layer, self.num_layers)

        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            decoder,
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        # CCA模块开关
        self.use_cca = getattr(args, 'use_cca', False)

        tgt_vocab = self.vocab_size + 1

        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)  # CMN

        # 初始化CCA上下文编码器
        if self.use_cca:
            self.context_encoder = ContextEncoder(
                d_model=args.d_model,
                num_heads=args.num_heads,
                dropout=args.dropout
            )
            # 用于推理阶段的可学习默认上下文向量
            self.default_context = nn.Parameter(torch.randn(1, args.d_model))
            # 用于存储训练时的文本嵌入（在forward中动态生成）
            self.text_embedding_cache = None

        self.model = self.make_model(tgt_vocab, self.cmn, use_cca=self.use_cca)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.cmm_size, args.cmm_dim))  # CMN: 2048, 512
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        # 获取context_vec (在训练时从forward传入，推理时使用默认上下文)
        context_vec = getattr(self, '_current_context_vec', None)
        if context_vec is None and self.use_cca:
            context_vec = self.default_context.expand(fc_feats.size(0), -1)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks, context_vec=context_vec)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask, context_vec = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        # 存储context_vec供推理时使用
        if self.use_cca:
            self._current_context_vec = context_vec

        # 保持与父类兼容，只返回4个值
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        # Memory querying and responding for visual features
        # Start
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0), self.memory_matrix.size(1))
        # dummy_memory_matrix: batchsize, 2048, 512
        responses = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix)

        # responses: batchsize, 98, 512
        att_feats = att_feats + responses
        # Memory querying and responding for visual features
        # End

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            # CCA: 生成上下文向量
            if self.use_cca:
                # 使用ground truth文本嵌入来生成全局上下文
                text_embeddings = self.model.tgt_embed[0].lut(seq) * math.sqrt(self.d_model)
                text_embeddings = text_embeddings + self.model.tgt_embed[1].pe[:, :text_embeddings.size(1)]
                context_vec = self.context_encoder(text_embeddings)
            else:
                context_vec = None
        else:
            seq_mask = None
            # 推理时：使用可学习的默认上下文向量
            if self.use_cca:
                context_vec = self.default_context.expand(att_feats.size(0), -1)
            else:
                context_vec = None

        return att_feats, seq, att_masks, seq_mask, context_vec

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask, context_vec = self._prepare_feature_forward(att_feats, att_masks, seq)

        # 存储context_vec供get_logprobs_state使用
        self._current_context_vec = context_vec

        out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix, context_vec=context_vec)
        outputs = F.log_softmax(self.logit(out), dim=-1)

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, context_vec=None):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past,
                                      memory_matrix=self.memory_matrix, context_vec=context_vec)
        return out[:, -1], [ys.unsqueeze(0)] + past