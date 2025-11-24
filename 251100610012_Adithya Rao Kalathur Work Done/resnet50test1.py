"""
FINAL FULL RESNET50 PIPELINE — WITH:
- tf.data memory-efficient loading
- Permanent poor augmentation folder (not in results)
- Automatic reuse of augmented images (no duplicates)
- LR Range Test (Fast.ai style)
- Automatic LR suggestion
- Smoothed Grad-CAM (SmoothGrad CAM)
- 3 Grad-CAMs per class → 3×4 grid
- CSV training log with learning rate
- Very memory-efficient
"""

# ============================================================
# IMPORTS
# ============================================================
import os, math, gc, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Mixed Precision
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print(" Mixed precision enabled")
except:
    print(" Mixed precision unavailable")

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Optional rotation support
try:
    import tensorflow_addons as tfa
    TFA = True
except:
    TFA = False

# ============================================================
# PATHS + CONSTANTS
# ============================================================
base_path = "/data/vaishnav25/Data_cuts1"
output_dir = "/data/vaishnav25/ADITHYA/Results_resnet50_test1"
os.makedirs(output_dir, exist_ok=True)

# NEW: Permanent augmentation folder
global_aug_dir = "/data/vaishnav25/ADITHYA/Poor_Augmented_Global"
os.makedirs(global_aug_dir, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1500
INITIAL_LR = 1e-4
MIN_LR = 1e-6
class_names = ["well", "mod", "poor"]
AUTOTUNE = tf.data.AUTOTUNE

print("Device:", "GPU" if tf.config.list_physical_devices("GPU") else "CPU")

# ============================================================
# 1️ LOAD + CAP DATA
# ============================================================
class_dirs = [
    os.path.join(base_path, "Well"),
    os.path.join(base_path, "Mod"),
    os.path.join(base_path, "Poor"),
]

all_paths, all_labels = [], []

for idx, cdir in enumerate(class_dirs):
    imgs = [os.path.join(cdir, f)
            for f in os.listdir(cdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Cap well+mod at 800
    if idx in [0, 1]:
        imgs = imgs[:800]

    all_paths.extend(imgs)
    all_labels.extend([idx] * len(imgs))

# ============================================================
# 2️ PERMANENT POOR AUGMENTATION (REUSE ACROSS RUNS)
# ============================================================
poor_paths = [p for p, l in zip(all_paths, all_labels) if l == 2]
original_poor = len(poor_paths)
target_poor = 800
need = target_poor - original_poor

# Check already augmented images
existing_aug = [
    os.path.join(global_aug_dir, f)
    for f in os.listdir(global_aug_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

existing_count = len(existing_aug)
print(f"Found {original_poor} original poor, {existing_count} augmented exist.")

# -----------------------------
# Case 1: Already enough → reuse
# -----------------------------
if existing_count >= need:
    print(" Reusing existing augmented images.")
    reuse = existing_aug[:need]
    all_paths.extend(reuse)
    all_labels.extend([2] * len(reuse))

# -----------------------------
# Case 2: Need to generate some
# -----------------------------
else:
    to_generate = need - existing_count
    print(f" Need {to_generate} more augmented images.")

    gen = ImageDataGenerator(brightness_range=[0.7, 1.3], channel_shift_range=30)
    rng = np.random.RandomState(42)
    created = 0

    while created < to_generate:
        src = rng.choice(poor_paths)
        pil = load_img(src, target_size=IMG_SIZE)
        x = img_to_array(pil).reshape((1,) + IMG_SIZE + (3,))

        for batch in gen.flow(x, batch_size=1):
            save_path = os.path.join(global_aug_dir, f"aug_{existing_count + created}.png")
            array_to_img(batch[0]).save(save_path)

            all_paths.append(save_path)
            all_labels.append(2)

            created += 1
            break

    print(f" Created {to_generate} new augmentations.")

    # Also reuse existing augmented
    all_paths.extend(existing_aug)
    all_labels.extend([2] * existing_count)

# ============================================================
# Shuffle dataset
# ============================================================
all_paths = np.array(all_paths)
all_labels = np.array(all_labels)

shuf = np.random.permutation(len(all_paths))
all_paths = all_paths[shuf]
all_labels = all_labels[shuf]

print("Final counts:")
for i, c in enumerate(class_names):
    print(c, np.sum(all_labels == i))

# ============================================================
# 3️ TRAIN/VAL SPLIT
# ============================================================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# ============================================================
# 4️ TF.DATA PIPELINE
# ============================================================
def decode_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32)/255., label

def train_aug(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.3)

    if TFA:
        angle = tf.random.uniform([], -20, 20) * math.pi/180
        img = tfa.image.rotate(img, angle)

    return img, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(2000)
    .map(decode_resize, AUTOTUNE)
    .map(train_aug, AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(decode_resize, AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ============================================================
# 5️ MODEL — RESNET50 (frozen)
# ============================================================
from tensorflow.keras.regularizers import l2

def build_resnet50():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    out = Dense(3, activation="softmax", kernel_regularizer=l2(1e-5), dtype="float32")(x)

    model = Model(base.input, out)
    model.compile(
        optimizer=Adam(INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_resnet50()
model.summary()

# ============================================================
# 6️ LR RANGE TEST
# ============================================================
def lr_range_test(model, train_ds, start_lr, end_lr, steps=120,
                  stop_multiplier=4.0, output_path=None):

    ds = train_ds.repeat().prefetch(1)
    it = iter(ds)

    w0 = model.get_weights()
    try:
        wopt0 = model.optimizer.get_weights()
    except:
        wopt0 = None

    lrs, losses = [], []
    best_loss = 1e9
    mult = (end_lr/start_lr)**(1/(steps-1))
    lr = start_lr

    for step in range(steps):
        tf.keras.backend.set_value(model.optimizer.lr, lr)

        xb, yb = next(it)
        out = model.train_on_batch(xb, yb)
        loss = float(out[0] if isinstance(out, (list, tuple)) else out)

        lrs.append(lr)
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
        if loss > best_loss * stop_multiplier:
            print("Loss exploded → stopping LR test.")
            break

        lr *= mult

    # Restore model
    model.set_weights(w0)
    if wopt0:
        try: model.optimizer.set_weights(wopt0)
        except: pass

    # Plot
    if output_path:
        plt.figure(figsize=(8,5))
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("LR")
        plt.ylabel("Loss")
        plt.title("LR Finder")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    return np.array(lrs), np.array(losses)

def suggest_lr(lrs, losses):
    idx = np.argmin(losses[10:-10])
    return float(lrs[10:-10][idx] * 0.1)

lrs, losses = lr_range_test(
    model, train_ds,
    start_lr=1e-7, end_lr=1e-1,
    steps=150,
    output_path=os.path.join(output_dir, "lr_finder.png")
)

best_lr = suggest_lr(lrs, losses)
print("Suggested LR:", best_lr)
tf.keras.backend.set_value(model.optimizer.lr, best_lr)

# ============================================================
# 7️ TRAINING + CSV LOG
# ============================================================
class CSVLoggerLR(Callback):
    def __init__(self, path):
        self.path = path
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except:
            lr = None

        logs = logs or {}
        row = {
            "epoch": epoch+1,
            "loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "accuracy": logs.get("accuracy"),
            "val_accuracy": logs.get("val_accuracy"),
            "learning_rate": lr
        }

        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

csv_path = os.path.join(output_dir, "training_log.csv")

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=MIN_LR, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    CSVLoggerLR(csv_path)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 8️ SAVE ACCURACY/LOSS CURVES
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train","Val"])
plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train","Val"])
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

# ============================================================
# 9️ CONFUSION MATRIX + ROC-AUC
# ============================================================
preds = model.predict(val_ds)
y_pred = np.argmax(preds, axis=1)
y_true = val_labels

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.savefig(os.path.join(output_dir, "conf_matrix.png"))
plt.close()

# ROC
y_bin = label_binarize(y_true, classes=[0,1,2])
plt.figure(figsize=(8,6))
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_bin[:,i], preds[:,i])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, label=f"{cname} AUC={roc_auc:.2f}")

plt.legend()
plt.title("ROC-AUC")
plt.savefig(os.path.join(output_dir, "roc_auc_curve.png"))
plt.close()

# ============================================================
# 10 SMOOTH GRAD-CAM (SmoothGrad CAM)
# ============================================================
def get_last_conv(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            return layer.name
    return None

def gradcam_smoothed(model, img_path, class_index, layer_name,
                     n_samples=16, noise_level=0.12):

    pil = load_img(img_path, target_size=IMG_SIZE)
    x0 = img_to_array(pil)/255.0
    x0 = x0.astype("float32")

    grad_model = Model(model.inputs,
                       [model.get_layer(layer_name).output, model.output])

    cams = []

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_level, x0.shape).astype("float32")
        x_noisy = np.clip(x0 + noise, 0, 1)
        x_batch = np.expand_dims(x_noisy,0)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x_batch)
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)[0].numpy()
        conv = conv_out[0].numpy()

        w = grads.mean(axis=(0,1))
        cam = np.dot(conv, w)
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max()+1e-8)
        cams.append(cam)

    cam_avg = np.mean(cams, axis=0)
    cam_avg = cam_avg / (cam_avg.max()+1e-8)
    cam_avg = cv2.resize(cam_avg, IMG_SIZE)

    heatmap = cv2.applyColorMap((cam_avg*255).astype("uint8"), cv2.COLORMAP_JET)
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, IMG_SIZE)

    overlay = cv2.addWeighted(orig,0.6,heatmap,0.4,0)
    embossed = cv2.addWeighted(orig,0.5,heatmap,0.5,0)

    return orig[:,:,::-1], heatmap[:,:,::-1], overlay[:,:,::-1], embossed[:,:,::-1]

# Generate 3×4 Grad-CAM grids
layer_name = get_last_conv(model)

for idx, cname in enumerate(class_names):
    sample_paths = [p for p,l in zip(val_paths,val_labels) if l==idx][:3]

    figs = []
    rows = []

    for p in sample_paths:
        rows.append(gradcam_smoothed(model, p, idx, layer_name))

    fig, ax = plt.subplots(3,4, figsize=(16,12))
    titles = ["Original","Heatmap","Overlay","Embossed"]

    for r in range(3):
        for c in range(4):
            ax[r,c].imshow(rows[r][c])
            if r==0:
                ax[r,c].set_title(titles[c])
            ax[r,c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"gradcam_{cname}_grid.png"))
    plt.close()

# ============================================================
# SAVE MODEL
# ============================================================
model.save(os.path.join(output_dir, "resnet50_final.keras"))
print("\nALL RESULTS SAVED IN:", output_dir)
print("Permanent poor augmentation folder:", global_aug_dir)
