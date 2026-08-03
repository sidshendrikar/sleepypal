"""
observer.py — The "Observation" engine: a CNN-LSTM seizure-state classifier
(Section 2.4 of the paper) plus the machinery that fixes the paper's headline
weakness (12.5% ictal sensitivity from class imbalance).

Architecture (matches the paper):
    Input (64,64,1)
    Conv2D(32,3x3,ReLU) -> MaxPool(2x2)          -> (32,32,32)
    Conv2D(64,3x3,ReLU) -> MaxPool(2x2)          -> (16,16,64)
    Reshape (16, 1024)   [rows as time steps]
    LSTM(64)
    Dense(3, softmax)  -> P(interictal), P(ictal), P(preictal)

Imbalance remedies added on top (all toggled from config.TrainConfig):
    * inverse-frequency class weights
    * categorical focal loss (down-weights easy majority epochs)
    * minority oversampling + optional Jansen-Rit synthetic augmentation
    * decision-threshold tuning on validation to hit a target sensitivity
    * BatchNorm + dropout for regularization (improvement over the paper)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from config import IMG_SIZE, TrainConfig, LABEL_NAMES

tf.get_logger().setLevel("ERROR")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(n_classes=3, in_channels=1, lstm_units=64, dropout=0.3):
    m = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, in_channels)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(2),
        layers.Reshape((IMG_SIZE // 4, (IMG_SIZE // 4) * 64)),   # (16, 1024)
        layers.LSTM(lstm_units),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return m


# ----------------------------------------------------------------------------
# Categorical focal loss  FL = -alpha * (1-p)^gamma * log(p)
# ----------------------------------------------------------------------------
def focal_loss(gamma=2.0, alpha=None):
    alpha = tf.constant(alpha if alpha is not None else [1.0, 1.0, 1.0],
                        dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _class_weights(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return w


def _oversample(X, y, n_classes=3, seed=0, cap=1.0):
    """Random oversample minority classes up to cap*majority count."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=n_classes)
    target = int(counts.max() * cap)
    Xs, ys = [X], [y]
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0 or len(idx) >= target:
            continue
        extra = rng.choice(idx, size=target - len(idx), replace=True)
        Xs.append(X[extra]); ys.append(y[extra])
    X2 = np.concatenate(Xs); y2 = np.concatenate(ys)
    perm = rng.permutation(len(y2))
    return X2[perm], y2[perm]


def train_observer(X, y, cfg: TrainConfig = None, X_syn=None, y_syn=None,
                   n_classes=3, verbose=0):
    """Train the CNN-LSTM. Returns (model, history, norm_stats).

    X_syn/y_syn : optional Jansen-Rit synthetic epochs (spectrograms) added to
                  the TRAIN split only to enrich the rare ictal class.
    """
    if cfg is None:
        cfg = TrainConfig()
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # stratified train/val split
    rng = np.random.default_rng(cfg.seed)
    tr_idx, va_idx = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * cfg.val_frac))
        va_idx.extend(idx[:k]); tr_idx.extend(idx[k:])
    tr_idx = np.array(tr_idx); va_idx = np.array(va_idx)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    # normalization stats from the REAL train split only (no val/synthetic leak)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8

    # add Jansen-Rit synthetic epochs to the train split only
    if X_syn is not None and len(X_syn):
        if X_syn.shape[-1] != Xtr.shape[-1]:        # tile 1-ch synth to C-ch
            X_syn = np.repeat(X_syn, Xtr.shape[-1], axis=-1)
        Xtr = np.concatenate([Xtr, X_syn])
        ytr = np.concatenate([ytr, y_syn])

    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    # ---- exactly ONE imbalance strategy (see TrainConfig.strategy) ----
    class_weight = None
    loss = "categorical_crossentropy"
    if cfg.strategy == "class_weight":
        cw = _class_weights(ytr, n_classes)
        class_weight = {c: float(cw[c]) for c in range(n_classes)}
    elif cfg.strategy == "oversample":
        Xtr, ytr = _oversample(Xtr, ytr, n_classes, seed=cfg.seed,
                               cap=cfg.oversample_cap)
    elif cfg.strategy == "oversample_focal":
        Xtr, ytr = _oversample(Xtr, ytr, n_classes, seed=cfg.seed,
                               cap=cfg.oversample_cap)
        loss = focal_loss(cfg.focal_gamma)          # uniform alpha
    else:
        raise ValueError(f"unknown strategy {cfg.strategy}")

    Ytr = tf.keras.utils.to_categorical(ytr, n_classes)
    Yva = tf.keras.utils.to_categorical(yva, n_classes)

    model = build_model(n_classes, in_channels=Xtr.shape[-1])
    model.compile(optimizer=optimizers.Adam(cfg.lr), loss=loss,
                  metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping(
               monitor="val_loss", patience=8, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(
               monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]

    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                     epochs=cfg.epochs, batch_size=cfg.batch_size,
                     class_weight=class_weight, verbose=verbose, callbacks=cbs)
    return model, hist, (mu, sd)


def predict_proba(model, X, norm_stats):
    mu, sd = norm_stats
    return model.predict((X - mu) / sd, verbose=0)
