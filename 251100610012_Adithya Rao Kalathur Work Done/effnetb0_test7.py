"""
=======================================================================
PROJECT: SCC Multi-class Classification (EfficientNetB0 + Grad-CAM)
FINAL: 1000 images/class, robust flat EfficientNet detection, BN freeze,
      Dense(256) head, Dropout=0.35, L2=2e-4, phase1 LR=7e-5, phase2 LR=2e-6
=======================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
from tensorflow.keras.regularizers import l2

# ---------------------------
# PATHS + CONFIG
# ---------------------------
base_path = "/data/vaishnav25/Data_cuts1"
output_dir = "/data/vaishnav25/ADITHYA/Results_effnetb0_final7"
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2000   # phase2 max epochs (phase1 uses 10)
TARGET_PER_CLASS = 1000

print("Using:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# ---------------------------
# LOAD EXACTLY 1000 IMAGES PER CLASS (augment poor to reach 1000)
# ---------------------------
class_names = ["well", "mod", "poor"]
class_dirs = {
    0: os.path.join(base_path, "Well"),
    1: os.path.join(base_path, "Mod"),
    2: os.path.join(base_path, "Poor")
}

# --- Augmenter used to create extra poor images (keeps your augmentation style) ---
augmenter_for_augmentation = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.7, 1.3],
    channel_shift_range=40,
    horizontal_flip=True,
    vertical_flip=False
)

all_image_paths = []
all_labels = []

print("\n=== Loading / preparing 1000 images per class ===")
for cls_idx, cls_dir in class_dirs.items():
    imgs = [
        os.path.join(cls_dir, f)
        for f in sorted(os.listdir(cls_dir))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    found = len(imgs)
    print(f"{class_names[cls_idx]}: found {found} images in {cls_dir}")

    if cls_idx in [0, 1]:
        # Well & Mod: cap to TARGET_PER_CLASS (if fewer than target, take all)
        if found >= TARGET_PER_CLASS:
            selected = imgs[:TARGET_PER_CLASS]
            all_image_paths.extend(selected)
            all_labels.extend([cls_idx] * TARGET_PER_CLASS)
            print(f" -> taking first {TARGET_PER_CLASS}")
        else:
            # fewer than target -> take all (you can opt to augment them too; currently we take all)
            selected = imgs
            all_image_paths.extend(selected)
            all_labels.extend([cls_idx] * len(selected))
            print(f" -> only {found} available, taking all {found} (not augmenting)")
    else:
        # Poor: if < TARGET_PER_CLASS, augment to reach TARGET_PER_CLASS
        if found >= TARGET_PER_CLASS:
            selected = imgs[:TARGET_PER_CLASS]
            all_image_paths.extend(selected)
            all_labels.extend([cls_idx] * TARGET_PER_CLASS)
            print(f" -> taking first {TARGET_PER_CLASS}")
        else:
            needed = TARGET_PER_CLASS - found
            print(f" -> augmenting poor: need {needed} additional images")
            aug_dir = os.path.join(base_path, "Poor_Augmented_1000")
            os.makedirs(aug_dir, exist_ok=True)
            augmented_paths = []
            aug_idx = 0
            # Keep original poor images
            all_image_paths.extend(imgs)
            all_labels.extend([cls_idx] * len(imgs))
            # Generate augmented images until we reach target
            while len(augmented_paths) < needed:
                pick = np.random.choice(imgs)
                img = load_img(pick, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                for batch in augmenter_for_augmentation.flow(x, batch_size=1):
                    aug_img = array_to_img(batch[0])
                    save_path = os.path.join(aug_dir, f"aug_{aug_idx}.png")
                    aug_img.save(save_path)
                    augmented_paths.append(save_path)
                    aug_idx += 1
                    break
            all_image_paths.extend(augmented_paths)
            all_labels.extend([cls_idx] * len(augmented_paths))
            print(f" -> saved {len(augmented_paths)} augmented poor images to {aug_dir}")

# Sanity print
all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

print("\nFinal counts (before shuffle):")
for i, name in enumerate(class_names):
    print(f"{name}: {np.sum(all_labels == i)}")
print("Total:", len(all_image_paths))

# Shuffle
all_image_paths, all_labels = shuffle(all_image_paths, all_labels, random_state=42)
print("Dataset shuffled.\n")

# ---------------------------
# DATA GENERATOR (training uses same augmentation style as before)
# ---------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.7, 1.3],
    channel_shift_range=40,
    horizontal_flip=True,
    vertical_flip=False
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

train_df = pd.DataFrame({"filename": train_paths, "label": train_labels})
val_df = pd.DataFrame({"filename": val_paths, "label": val_labels})

train_gen = datagen.flow_from_dataframe(
    train_df, x_col="filename", y_col="label",
    target_size=IMG_SIZE, class_mode="raw",
    batch_size=BATCH_SIZE, shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    val_df, x_col="filename", y_col="label",
    target_size=IMG_SIZE, class_mode="raw",
    batch_size=BATCH_SIZE, shuffle=False
)

# ---------------------------
# BUILD MODEL (EfficientNetB0 base, Dense(256) head, dropout, L2)
# ---------------------------
def build_model(l2_reg=2e-4, dropout=0.35):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout)(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    outputs = Dense(3, activation="softmax", kernel_regularizer=l2(l2_reg))(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model

model = build_model()
initial_lr = 7e-5
model.compile(optimizer=Adam(initial_lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------
# CALLBACKS
# ---------------------------
checkpoint_path = os.path.join(output_dir, "best_effnetb0.h5")
callbacks_phase1 = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
]

# ---------------------------
# PHASE 1: train head
# ---------------------------
initial_epochs = 10
history_phase1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epochs,
    callbacks=callbacks_phase1,
    verbose=1
)

# ---------------------------
# PHASE 2: FINE-TUNING — flat EfficientNet detection, BN freeze, unfreeze last 40%
# ---------------------------
# Find indices of EfficientNet layers in the flattened model
eff_indices = []
for i, layer in enumerate(model.layers):
    n = layer.name.lower()
    # these prefixes are part of EfficientNet naming
    if n.startswith(("stem", "block", "top", "expand", "project")):
        eff_indices.append(i)

if len(eff_indices) == 0:
    # fallback broader search
    for i, layer in enumerate(model.layers):
        n = layer.name.lower()
        if "efficientnet" in n or n.startswith(("block", "stem", "top")):
            eff_indices.append(i)

if len(eff_indices) == 0:
    raise ValueError("EfficientNet layers not found in model.layers — print model.summary() to debug")

start_idx = eff_indices[0]
end_idx = eff_indices[-1]
num_eff_layers = len(eff_indices)
print(f"Detected EfficientNet flattened layers: {num_eff_layers}, indices {start_idx}..{end_idx}")

# Freeze BatchNorm layers
for idx in eff_indices:
    if isinstance(model.layers[idx], tf.keras.layers.BatchNormalization):
        model.layers[idx].trainable = False

# Unfreeze last 40% of EfficientNet layers (except BatchNorm)
to_unfreeze = max(1, int(num_eff_layers * 0.40))
for idx in eff_indices[-to_unfreeze:]:
    if not isinstance(model.layers[idx], tf.keras.layers.BatchNormalization):
        model.layers[idx].trainable = True

print(f"Unfreezing last {to_unfreeze} EfficientNet layers (BatchNorm kept frozen)")

# Recompile with low LR for fine-tuning
fine_tune_lr = 2e-6
model.compile(optimizer=Adam(fine_tune_lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks_phase2 = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
]

history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    initial_epoch=initial_epochs,
    callbacks=callbacks_phase2,
    verbose=1
)

# load best weights if saved
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

# ---------------------------
# MERGE HISTORIES + PLOTS
# ---------------------------
def merge_histories(h1, h2):
    out = {}
    for k in h1.history.keys():
        out[k] = h1.history[k] + h2.history.get(k, [])
    for k in h2.history.keys():
        if k not in out:
            out[k] = h2.history[k]
    return out

history = merge_histories(history_phase1, history_phase2)

plt.figure(figsize=(10,4))
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], "--", label="Val Acc")
plt.legend(); plt.title("Accuracy Curve")
plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], "--", label="Val Loss")
plt.legend(); plt.title("Loss Curve")
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

# ---------------------------
# CONFUSION MATRIX + ROC
# ---------------------------
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = val_labels

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(6,6)); disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix"); plt.savefig(os.path.join(output_dir, "confusion_matrix.png")); plt.close()

y_true_bin = label_binarize(y_true, classes=[0,1,2])
plt.figure(figsize=(8,6))
for i, c in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_pred_probs[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1], 'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC-AUC Curve")
plt.legend(); plt.grid(True); plt.savefig(os.path.join(output_dir, "roc_auc_curve.png")); plt.close()

# ---------------------------
# GRAD-CAM (overlay)
# ---------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if "conv" in layer.name.lower():
            return layer.name
    return None

last_conv_layer_name = find_last_conv_layer(model)
print("Using last conv layer for Grad-CAM:", last_conv_layer_name)

def generate_gradcam(model, img_path, class_index=None, layer_name=None):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)[None, ...]
    x = preprocess_input(x.copy())

    preds = model.predict(x)
    if class_index is None:
        class_index = np.argmax(preds[0])

    if layer_name is None:
        layer_name = last_conv_layer_name

    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0].numpy()
    weights = np.mean(grads.numpy(), axis=(0,1))
    cam = np.dot(conv_outputs, weights)
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, IMG_SIZE)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    return overlay[:, :, ::-1], heatmap[:, :, ::-1], overlay[:, :, ::-1], overlay[:, :, ::-1]

num_images_per_class = 5
for i, class_name in enumerate(class_names):
    sample_imgs = val_df[val_df['label'] == i]['filename'].iloc[:num_images_per_class]
    for idx, sample_img in enumerate(sample_imgs):
        try:
            orig, heatmap, overlay, embossed = generate_gradcam(model, sample_img, class_index=i)
            fig, ax = plt.subplots(1, 4, figsize=(16,5))
            titles = [f"{class_name.capitalize()} - Original", "Heatmap", "Overlay", "Embossed"]
            for j, image in enumerate([orig, heatmap, overlay, embossed]):
                ax[j].imshow(image); ax[j].set_title(titles[j]); ax[j].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gradcam_{class_name}_{idx}.png"))
            plt.close()
        except Exception as e:
            print("Grad-CAM error for", sample_img, ":", e)

# ---------------------------
# SAVE MODEL
# ---------------------------
model_save_path = os.path.join(output_dir, "effnetb0_final7.h5")
model.save(model_save_path)
print("Model saved at:", model_save_path)
