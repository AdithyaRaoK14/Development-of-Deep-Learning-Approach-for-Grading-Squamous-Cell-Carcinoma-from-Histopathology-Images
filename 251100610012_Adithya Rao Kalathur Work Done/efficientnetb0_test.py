"""
=======================================================================
PROJECT: SCC Multi-class Classification (EfficientNetB0 + Grad-CAM)
EXPERIMENT ENGINE VERSION — clean, modular, per-experiment results
6 SYSTEMATIC EXPERIMENT SETUP
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
from tensorflow.keras.regularizers import l1, l2
import cv2

# ======================================================================
# ======================================================================
#  EXPERIMENT CONFIG — EDIT ONLY THIS (choose 1 of the 6 below)
# ======================================================================
# ======================================================================

# ----------------- COPY ONE BLOCK AT A TIME HERE ----------------------

# 1) Baseline
# EXP_ID = "exp1_baseline_l2_2e4_dropout35"
# LR_PHASE1 = 7e-5
# LR_PHASE2 = 2e-6
# DROPOUT = 0.35
# REG_TYPE = "l2"
# REG_VALUE = 2e-4

# 2) No dropout, no regularization
# EXP_ID = "exp2_nodropout_noreg"
# LR_PHASE1 = 7e-5
# LR_PHASE2 = 2e-6
# DROPOUT = 0.0
# REG_TYPE = None
# REG_VALUE = 0

# 3) High dropout 0.50 with L2
# EXP_ID = "exp3_dropout50_l2_2e4"
# LR_PHASE1 = 7e-5
# LR_PHASE2 = 2e-6
# DROPOUT = 0.50
# REG_TYPE = "l2"
# REG_VALUE = 2e-4

# 4) Low LR + Dropout 0.20
# EXP_ID = "exp4_lowlr_dropout20"
# LR_PHASE1 = 3e-5
# LR_PHASE2 = 1e-6
# DROPOUT = 0.20
# REG_TYPE = "l2"
# REG_VALUE = 2e-4

# 5) L1 Regularization
# EXP_ID = "exp5_l1_1e5_dropout35"
# LR_PHASE1 = 7e-5
# LR_PHASE2 = 2e-6
# DROPOUT = 0.35
# REG_TYPE = "l1"
# REG_VALUE = 1e-5

# 6) Strong L2 + High Dropout
EXP_ID = "exp6_strongl2_5e4_dropout50"
LR_PHASE1 = 5e-5
LR_PHASE2 = 5e-7
DROPOUT = 0.50
REG_TYPE = "l2"
REG_VALUE = 5e-4

# ---------------------------------------------------------------------

MAIN_OUTPUT = "/data/vaishnav25/ADITHYA/EFFNET_EXPERIMENTS"
os.makedirs(MAIN_OUTPUT, exist_ok=True)

output_dir = os.path.join(MAIN_OUTPUT, EXP_ID)
os.makedirs(output_dir, exist_ok=True)

print(f"\n===== RUNNING EXPERIMENT: {EXP_ID} =====")
print(f"Saving all outputs to: {output_dir}\n")

# ======================================================================
# ======================================================================

# PATHS
base_path = "/data/vaishnav25/Data_cuts1"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2000
TARGET_PER_CLASS = 1000

print("Using:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# =====================================================
# LOAD 1000 IMAGES PER CLASS
# =====================================================
class_names = ["well", "mod", "poor"]
class_dirs = {
    0: os.path.join(base_path, "Well"),
    1: os.path.join(base_path, "Mod"),
    2: os.path.join(base_path, "Poor")
}

augmenter_for_augmentation = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.7,1.3],
    channel_shift_range=40,
    horizontal_flip=True,
    vertical_flip=False
)

all_image_paths = []
all_labels = []

print("\n=== Loading / preparing 1000 images per class ===")
for cls_idx, cls_dir in class_dirs.items():

    imgs = [os.path.join(cls_dir,f) for f in sorted(os.listdir(cls_dir))
            if f.lower().endswith(('.png','.jpg','.jpeg'))]

    found = len(imgs)
    print(f"{class_names[cls_idx]}: found {found}")

    if cls_idx in [0,1]:
        if found >= TARGET_PER_CLASS:
            use = imgs[:TARGET_PER_CLASS]
        else:
            use = imgs

        all_image_paths.extend(use)
        all_labels.extend([cls_idx]*len(use))

    else:
        # Poor class
        if found >= TARGET_PER_CLASS:
            use = imgs[:TARGET_PER_CLASS]
            all_image_paths.extend(use)
            all_labels.extend([cls_idx]*TARGET_PER_CLASS)

        else:
            needed = TARGET_PER_CLASS - found
            aug_dir = os.path.join(base_path, "Poor_Augmented_1000")
            os.makedirs(aug_dir, exist_ok=True)

            all_image_paths.extend(imgs)
            all_labels.extend([cls_idx]*found)

            augmented = []
            aug_idx = 0
            while len(augmented) < needed:
                pick = np.random.choice(imgs)
                img = load_img(pick, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                for batch in augmenter_for_augmentation.flow(x, batch_size=1):
                    aug_img = array_to_img(batch[0])
                    save_path = os.path.join(aug_dir, f"aug_{aug_idx}.png")
                    aug_img.save(save_path)
                    augmented.append(save_path)
                    aug_idx += 1
                    break

            all_image_paths.extend(augmented)
            all_labels.extend([cls_idx]*len(augmented))

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)
all_image_paths, all_labels = shuffle(all_image_paths, all_labels, random_state=42)

# =====================================================
# DATA GENERATORS
# =====================================================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.7,1.3],
    channel_shift_range=40,
    horizontal_flip=True,
    vertical_flip=False
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

train_df = pd.DataFrame({"filename":train_paths,"label":train_labels})
val_df = pd.DataFrame({"filename":val_paths,"label":val_labels})

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

# =====================================================
# MODEL BUILDER
# =====================================================
def build_model(dropout, reg_type, reg_value):

    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    for layer in base.layers:
        layer.trainable = False

    if reg_type == "l2":
        reg = l2(reg_value)
    elif reg_type == "l1":
        reg = l1(reg_value)
    else:
        reg = None

    x = GlobalAveragePooling2D()(base.output)

    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Dense(256, activation="relu", kernel_regularizer=reg)(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    outputs = Dense(3, activation="softmax", kernel_regularizer=reg)(x)

    return Model(inputs=base.input, outputs=outputs)

model = build_model(DROPOUT, REG_TYPE, REG_VALUE)
model.compile(optimizer=Adam(LR_PHASE1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# =====================================================
# CALLBACKS
# =====================================================
checkpoint_path = os.path.join(output_dir, "best_model.h5")

callbacks_phase1 = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
]

# =====================================================
# TRAIN PHASE 1
# =====================================================
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks_phase1,
    verbose=1
)

# =====================================================
# FINE TUNE (PHASE 2)
# =====================================================
eff_indices = []
for i, layer in enumerate(model.layers):
    if "block" in layer.name.lower() or "stem" in layer.name.lower():
        eff_indices.append(i)

# Freeze BN
for idx in eff_indices:
    if isinstance(model.layers[idx], tf.keras.layers.BatchNormalization):
        model.layers[idx].trainable = False

# Unfreeze last 40%
N = len(eff_indices)
to_unfreeze = max(1, int(N*0.40))
for idx in eff_indices[-to_unfreeze:]:
    if not isinstance(model.layers[idx], tf.keras.layers.BatchNormalization):
        model.layers[idx].trainable = True

model.compile(optimizer=Adam(LR_PHASE2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks_phase2 = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    initial_epoch=10,
    callbacks=callbacks_phase2,
    verbose=1
)

model.load_weights(checkpoint_path)

# =====================================================
# PLOT GRAPHS
# =====================================================
def merge_histories(h1,h2):
    out={}
    for k in h1.history.keys():
        out[k] = h1.history[k] + h2.history.get(k,[])
    return out

history = merge_histories(history1, history2)

plt.figure(figsize=(10,4))
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["train","val"])
plt.savefig(os.path.join(output_dir,"accuracy_curve.png"))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Loss")
plt.legend(["train","val"])
plt.savefig(os.path.join(output_dir,"loss_curve.png"))
plt.close()

# =====================================================
# CONFUSION MATRIX + ROC
# =====================================================
val_gen.reset()
y_pred = model.predict(val_gen)
y_pred_labels = np.argmax(y_pred,axis=1)
y_true = val_labels

cm = confusion_matrix(y_true, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
plt.figure(figsize=(6,6))
disp.plot(cmap=plt.cm.Blues,values_format='d')
plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))
plt.close()

# ROC Curve
y_true_bin = label_binarize(y_true, classes=[0,1,2])
plt.figure(figsize=(8,6))
for i,c in enumerate(class_names):
    fpr,tpr,_ = roc_curve(y_true_bin[:,i], y_pred[:,i])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label=f"{c} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.savefig(os.path.join(output_dir,"roc_auc_curve.png"))
plt.close()

# =====================================================
# GRAD-CAM
# =====================================================
def find_last_conv(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    return None

last_conv = find_last_conv(model)

def generate_gradcam(model,img_path,class_idx,layer_name):

    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)[None]
    x = preprocess_input(x)

    grad_model = Model(model.inputs, [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:,class_idx]

    grads = tape.gradient(loss, conv_out)[0].numpy()
    conv_out = conv_out[0].numpy()

    weights = grads.mean(axis=(0,1))
    cam = np.dot(conv_out, weights)
    cam = np.maximum(cam,0)
    cam = cam/(cam.max()+1e-8)
    cam = cv2.resize(cam, IMG_SIZE)

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, IMG_SIZE)

    overlay = cv2.addWeighted(orig,0.6,heatmap,0.4,0)
    return orig[:,:,::-1], heatmap[:,:,::-1], overlay[:,:,::-1]

num_images = 5
for i,cname in enumerate(class_names):
    subset = val_df[val_df['label']==i]['filename'].iloc[:num_images]

    for idx,fp in enumerate(subset):
        try:
            orig,heat,overlay = generate_gradcam(model,fp,i,last_conv)
            fig,ax = plt.subplots(1,3,figsize=(14,5))
            for j,img in enumerate([orig,heat,overlay]):
                ax[j].imshow(img); ax[j].axis("off")
            plt.savefig(os.path.join(output_dir,f"gradcam_{cname}_{idx}.png"))
            plt.close()
        except Exception as e:
            print("GradCAM Error:", e)

# =====================================================
# SAVE MODEL
# =====================================================
model.save(os.path.join(output_dir, f"{EXP_ID}_final_model.h5"))

print("\n============================")
print(f"EXPERIMENT {EXP_ID} COMPLETED")
print("============================\n")
