from __future__ import annotations
import numpy as np
import tensorflow as tf
import keras
from keras import layers

def moore8_adj_unweighted(k: int) -> np.ndarray:
    N = k * k
    A = np.zeros((N, N), np.float32)
    def idx(i, j): return i * k + j
    for i in range(k):
        for j in range(k):
            u = idx(i, j)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < k and 0 <= nj < k:
                        A[u, idx(ni, nj)] = 1.0
    return A

def normalize_adj(A: np.ndarray) -> np.ndarray:
    D = np.sum(A, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-9))
    A_norm = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
    return A_norm.astype(np.float32)

class GraphConv(layers.Layer):
    def __init__(self, out_dim: int, A_norm: np.ndarray, **kw):
        super().__init__(**kw)
        self.A = tf.constant(A_norm, dtype=tf.float32)
        self.proj = layers.Dense(out_dim, use_bias=False)
    def call(self, x):
        ax = tf.einsum("ij,bjf->bif", self.A, x)
        return self.proj(ax)

class GNNBlock(layers.Layer):
    def __init__(self, dim: int, A_norm: np.ndarray, dropout: float = 0.1, mlp_ratio: int = 4, **kw):
        super().__init__(**kw)
        self.n1 = layers.LayerNormalization(epsilon=1e-5)
        self.gc = GraphConv(dim, A_norm)
        self.d1 = layers.Dropout(dropout)
        self.n2 = layers.LayerNormalization(epsilon=1e-5)
        self.ff = keras.Sequential([
            layers.Dense(dim * mlp_ratio, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])
        self.d2 = layers.Dropout(dropout)
    def call(self, x, training=False):
        h = self.gc(self.n1(x))
        x = x + self.d1(h, training=training)
        h = self.ff(self.n2(x), training=training)
        x = x + self.d2(h, training=training)
        return x

def build_patch_gnn(k: int, c: int, d_hidden: int, n_layers: int, dropout: float, use_pos_emb: bool = True):
    assert k % 2 == 1, "k_patch phải lẻ"
    N = k * k
    center_idx = N // 2
    A_norm = normalize_adj(moore8_adj_unweighted(k))

    inp = layers.Input(shape=(k, k, c), name="patch")
    x = layers.Reshape((N, c), name="flatten_nodes")(inp)
    x = layers.Dense(d_hidden, name="node_proj")(x)

    if use_pos_emb:
        pos_emb_layer = layers.Embedding(input_dim=N, output_dim=d_hidden, name="pos_emb")
        def _add_pos(t):
            pe = pos_emb_layer(tf.range(N))
            pe = tf.expand_dims(pe, axis=0)
            return t + pe
        x = layers.Lambda(_add_pos, name="add_pos_emb")(x)

    for i in range(n_layers):
        x = GNNBlock(d_hidden, A_norm, dropout=dropout, name=f"gnn_blk_{i}")(x)

    center = layers.Lambda(lambda t: t[:, center_idx, :], name="center")(x)
    gmean  = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name="gmean")(x)
    h = layers.Concatenate(name="readout")([center, gmean])

    h = layers.Dense(d_hidden, activation="gelu")(h)
    h = layers.Dropout(dropout)(h)
    logit = layers.Dense(1)(h)
    logit = layers.Lambda(lambda t: tf.squeeze(t, -1), name="logit")(logit)
    return keras.Model(inp, logit, name=f"PatchGNN_k{k}")

class WeightedBCEFromLogits(keras.losses.Loss):
    def __init__(self, pos_weight=1.0, name="wbce_logits"):
        super().__init__(reduction="none", name=name)
        self.pos_weight = float(pos_weight)
    def call(self, y_true, y_logit):
        y_true  = tf.cast(y_true, tf.float32)
        y_logit = tf.cast(y_logit, tf.float32)
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_logit, pos_weight=self.pos_weight)
