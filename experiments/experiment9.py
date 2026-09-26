"""Round I: recipe comparison on the split that includes all four new
out-of-scope downloads. Trains on the repo's data/train, scores with
evaluate_model.evaluate (app image loader) on data/val and data/test, and saves
per-photo probabilities so decision rules can be measured without retraining.
Selection is on val only. (Recreated from round G/H's experiment8.py after the
temporary folder was wiped by a restart; training code is unchanged.)"""
import os, sys, json, time, argparse
import numpy as np
EXP = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(EXP)
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import train_model as t
from model import unfreeze_top_layers, load_gate_model
from evaluate_model import evaluate, predict_directory

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--seed", type=int, default=1)
p.add_argument("--label-smoothing", type=float, default=0.0)
p.add_argument("--b3-weight", type=float, default=2.0)
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--finetune-lr", type=float, default=1e-5)
p.add_argument("--finetune-layers", type=int, default=100)
p.add_argument("--mixup", type=float, default=0.0, help="Beta(a, a) mixup within each batch; 0 = off")
p.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetv2b0", "efficientnetv2s"])
p.add_argument("--train-dir", default=None, help="training folder (default: the repo's data/train)")
p.add_argument("--zoom-out", type=float, default=0.0, help="RandomZoom upper bound: how far training photos may be zoomed OUT")
p.add_argument("--val-dir", default=None, help="early-stopping folder (default: the repo's data/val); evaluation always uses data/val and data/test")
p.add_argument("--min-fill", type=float, default=1.0, help="smallest part of the frame a training photo may be shrunk into (1.0 = off)")
a = p.parse_args()
tf.keras.utils.set_random_seed(a.seed)

TRAIN = a.train_dir or f"{WT}/data/train"
names = sorted(d for d in os.listdir(TRAIN) if os.path.isdir(f"{TRAIN}/{d}"))
def load(d, shuffle):
    return tf.keras.utils.image_dataset_from_directory(d, image_size=(224, 224), batch_size=32, label_mode="categorical",
                                                       class_names=names, shuffle=shuffle, seed=a.seed, follow_links=True)
aug = t.make_augmenter(zoom_out=a.zoom_out, min_frame_fill=a.min_fill)
train_ds = load(TRAIN, True).map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0, 255), y),
                                              num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
if a.mixup > 0:
    def mix(x, y):
        g1 = tf.random.gamma([], a.mixup); g2 = tf.random.gamma([], a.mixup)
        lam = g1 / (g1 + g2)
        perm = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        return lam * x + (1 - lam) * tf.gather(x, perm), lam * y + (1 - lam) * tf.gather(y, perm)
    train_ds = train_ds.map(mix, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = load(a.val_dir or f"{WT}/data/val", False).map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

counts = np.array([len(os.listdir(f"{TRAIN}/{c}")) for c in names], dtype=float)
w = counts.sum() / (len(names) * counts)
w[names.index("burn_3rd_degree")] *= a.b3_weight
cw = {i: float(v) for i, v in enumerate(w)}

inputs = layers.Input(shape=(224, 224, 3))
if a.backbone in ("efficientnetv2b0", "efficientnetv2s"):
    ctor = tf.keras.applications.EfficientNetV2B0 if a.backbone == "efficientnetv2b0" else tf.keras.applications.EfficientNetV2S
    base = ctor(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    x = layers.Rescaling(127.5, offset=127.5)(inputs)   # app preprocessing gives [-1, 1]; this backbone expects [0, 255]
else:
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet", alpha=a.alpha)
    x = inputs
base.trainable = False
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
model = models.Model(inputs, layers.Dense(len(names), activation="softmax")(x))
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=a.label_smoothing)

t0 = time.time()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
h1 = model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=cw, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
unfreeze_top_layers(model, a.finetune_layers)
model.compile(optimizer=tf.keras.optimizers.Adam(a.finetune_lr), loss=loss, metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=40, class_weight=cw, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])

def summary(r):
    b, o, ap = r["burn"], r.get("out_of_scope", {}), r["app"]
    return dict(acc=r["accuracy"], bal=r["balanced_accuracy"], wound_class_acc=r["wound_class_accuracy"],
                answered=ap["in_scope_fraction_answered"], acc_answered=ap["in_scope_accuracy_when_answered"],
                rejected=ap["in_scope_fraction_rejected_as_out_of_scope"], ood_labelled=o.get("fraction_confidently_labelled"),
                b3_recall=b["burn_3rd_recall"], b3_app_1st=b["burn_3rd_app_says_1st"], b3_app_other=b["burn_3rd_app_says_other_wound"],
                b3_app_unknown=b["burn_3rd_app_says_unknown"], b2_recall=r["per_class"]["burn_2nd_degree"]["recall"],
                recall={c: v["recall"] for c, v in r["per_class"].items()})
gate = load_gate_model()   # the shipped gate, unchanged, so numbers are app-level
val, test = evaluate(model, names, f"{WT}/data/val", gate), evaluate(model, names, f"{WT}/data/test", gate)
framed = {f"frame{pct}_{split}": summary(evaluate(model, names, f"{EXP}/data_frame{pct}/{split}", gate))
          for pct in (70, 50) for split in ("val", "test")}
tag = f"{a.name}_s{a.seed}"
yv, pv, fv = predict_directory(model, f"{WT}/data/val", names)
yt, pt, ft = predict_directory(model, f"{WT}/data/test", names)
np.savez(f"{EXP}/runs/{tag}_probs.npz", names=names, y_val=yv, p_val=pv, f_val=fv, y_test=yt, p_test=pt, f_test=ft)
out = dict(config=vars(a), epochs=[len(h1.history["loss"]), len(h2.history["loss"])], minutes=(time.time() - t0) / 60,
           val=summary(val), test=summary(test), framed=framed, val_full=val, test_full=test)
json.dump(out, open(f"{EXP}/runs/{tag}.json", "w"), indent=1)
v, te = out["val"], out["test"]
print(f"RESULT {tag}: VAL bal {v['bal']:.3f} ood {v['ood_labelled']:.3f} rej {v['rejected']:.3f} b3 other {v['b3_app_other']} 1st {v['b3_app_1st']} | "
      f"TEST acc {te['acc']:.3f} bal {te['bal']:.3f} ood {te['ood_labelled']:.3f} | "
      f"FRAMED50 val answered {framed['frame50_val']['answered']:.3f} test answered {framed['frame50_test']['answered']:.3f} | {out['minutes']:.1f} min")
