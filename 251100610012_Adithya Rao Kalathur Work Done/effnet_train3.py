"""
=======================================================================
PROJECT: Lung Cancer Multi-class Classification (EfficientNetV2B1 + Auto Smoothed Grad-CAM)
DESCRIPTION:
  - 3 Classes: well, mod, poor
  - well & mod capped at 500 images
  - poor augmented to 300 using SMOTE-like image augmentation
  - Two-phase training: Frozen base → Fine-tuning (last 30 layers)
  - Adam optimizer with learning rate scheduling + early stopping
  - Dropout(0.3) + L2 regularization to reduce overfitting
  - Smoothed Grad-CAM visualization (auto per-class)
  - Saves best weights for both phases
=======================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2

# ============================================================
# 1 PATH SETUP
# ============================================================
base_path = "/data/vaishnav25/Data"
output_dir = "/data/vaishnav25/ADITHYA/Results3"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2 GPU CHECK
# ============================================================
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f" Using {device}")

# ============================================================
# 3 LOAD DATA (limit 500 per well/mod; augment poor → 300)
# ============================================================
class_names = ["well", "mod", "poor"]
class_paths = [os.path.join(base_path, "Well"),
               os.path.join(base_path, "Mod"),
               os.path.join(base_path, "Poor")]

all_image_paths, all_labels = [], []
max_images_per_class = [500, 500, 300]

for idx, class_dir in enumerate(class_paths):
    imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if idx < 2:
        imgs = imgs[:500]
    all_image_paths.extend(imgs)
    all_labels.extend([idx] * len(imgs))

# ===============================
# Data augmentation for poor class
# ===============================
IMG_SIZE = (260, 260)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

poor_indices = [i for i, l in enumerate(all_labels) if l == 2]
if len(poor_indices) < 300:
    poor_imgs = [all_image_paths[i] for i in poor_indices]
    augmented_imgs = []
    while len(poor_imgs) + len(augmented_imgs) < 300:
        img_path = np.random.choice(poor_imgs)
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        for batch in datagen.flow(x, batch_size=1):
            augmented_imgs.append(array_to_img(batch[0]))
            break

    temp_aug_dir = os.path.join(base_path, "Poor_Augmented")
    os.makedirs(temp_aug_dir, exist_ok=True)
    for i, img in enumerate(augmented_imgs):
        aug_path = os.path.join(temp_aug_dir, f"aug_{i}.png")
        img.save(aug_path)
        all_image_paths.append(aug_path)
        all_labels.append(2)

all_image_paths, all_labels = shuffle(np.array(all_image_paths), np.array(all_labels), random_state=42)

print(" Class distribution (after augmentation):")
for i, c in enumerate(class_names):
    print(f"{c}: {np.sum(all_labels == i)}")

# ============================================================
# 4 DATA GENERATOR
# ============================================================
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 100

train_df, val_df, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

train_df = pd.DataFrame({'filename': train_df, 'label': train_labels})
val_df = pd.DataFrame({'filename': val_df, 'label': val_labels})

train_gen = datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='label',
    target_size=IMG_SIZE, class_mode='raw',
    batch_size=BATCH_SIZE, shuffle=True
)
val_gen = datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='label',
    target_size=IMG_SIZE, class_mode='raw',
    batch_size=BATCH_SIZE, shuffle=False
)

# ============================================================
# 5 MODEL DEFINITION
# ============================================================
def build_effnetv2b1_model(input_shape=(260, 260, 3), num_classes=3):
    base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

model, base_model = build_effnetv2b1_model()

# ============================================================
# 6 PHASE 1 — FROZEN BASE
# ============================================================
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=3e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks_phase1 = [
    ModelCheckpoint(os.path.join(output_dir, "best_phase1_model.h5"),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

print("\n Phase 1: Training with frozen base...")
history1 = model.fit(train_gen, validation_data=val_gen,
                     epochs=EPOCHS_PHASE1, callbacks=callbacks_phase1, verbose=1)

# ============================================================
# 7 PHASE 2 — FINE-TUNING
# ============================================================
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks_phase2 = [
    ModelCheckpoint(os.path.join(output_dir, "best_phase2_model.h5"),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
]

print("\n Phase 2: Fine-tuning last 30 layers...")
history2 = model.fit(train_gen, validation_data=val_gen,
                     epochs=EPOCHS_PHASE2, callbacks=callbacks_phase2, verbose=1)

# ============================================================
# 8 PLOTS
# ============================================================
def plot_history():
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    plt.figure(figsize=(10, 4))
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, '--', label='Val Acc')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle=':', label='Phase 2 Start')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, '--', label='Val Loss')
    plt.axvline(x=len(history1.history['loss']), color='r', linestyle=':', label='Phase 2 Start')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

plot_history()

# ============================================================
# 9 CONFUSION MATRIX & ROC-AUC
# ============================================================
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = val_labels

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(6,6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

y_true_bin = label_binarize(y_true, classes=[0,1,2])
plt.figure(figsize=(8,6))
for i, c in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC-AUC")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "roc_auc_curve.png"))
plt.close()

# ============================================================
# 10 SMOOTHED GRAD-CAM (AUTO PER CLASS)
# ============================================================
def generate_smooth_gradcam(model, img_path, layer_name=None, n_samples=8, noise_level=0.02):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    class_index = np.argmax(preds[0])
    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
    heatmaps = []
    for _ in range(n_samples):
        noisy = x + np.random.normal(0, noise_level, x.shape)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(noisy)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, conv_out)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = np.dot(conv_out[0], pooled_grads.numpy())
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-8)
        if cam.ndim == 2:
            heatmaps.append(cam)
    smooth_cam = np.mean(heatmaps, axis=0)
    smooth_cam = cv2.GaussianBlur(smooth_cam, (5,5), 0)
    smooth_cam /= (np.max(smooth_cam) + 1e-8)
    return smooth_cam

def overlay_gradcam(img_path, heatmap, out_path):
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, IMG_SIZE)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(out_path, overlay)

print(" Generating top-3 Grad-CAMs per class...")
correct_mask = y_true == y_pred_classes
correct_files = np.array(val_df['filename'])[correct_mask]
correct_labels = np.array(y_true)[correct_mask]
correct_preds = np.array(y_pred_probs)[correct_mask]

for c_idx, c_name in enumerate(class_names):
    cls_mask = correct_labels == c_idx
    if not np.any(cls_mask): continue
    cls_probs = correct_preds[cls_mask, c_idx]
    cls_files = correct_files[cls_mask]
    top_idx = np.argsort(cls_probs)[-3:]
    out_cls_dir = os.path.join(output_dir, f"GradCAM_{c_name}")
    os.makedirs(out_cls_dir, exist_ok=True)
    for i in top_idx:
        img_path = cls_files[i]
        heatmap = generate_smooth_gradcam(model, img_path)
        out_path = os.path.join(out_cls_dir, f"{os.path.basename(img_path)}_gradcam.png")
        overlay_gradcam(img_path, heatmap, out_path)
        print(f" Saved Grad-CAM for {c_name}: {out_path}")

# ============================================================
#  SAVE MODEL SAFELY
# ============================================================
try:
    model.save(os.path.join(output_dir, "effnetv2b1_finetuned_gradcam_final.h5"))
    print(f"\n Model saved successfully at {output_dir}")
except Exception as e:
    print(f" Model save failed: {e}")
