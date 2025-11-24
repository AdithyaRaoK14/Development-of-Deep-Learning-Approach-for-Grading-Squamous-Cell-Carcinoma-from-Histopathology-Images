"""
=======================================================================
PROJECT: Lung Cancer Multi-class Classification (SE-ResNet50 + Grad-CAM)
DESCRIPTION:
  - 3 Classes: well, mod, poor
  - well & mod capped at 500 images
  - poor augmented to 300 using color-based augmentation
  - SE-ResNet50 backbone (ResNet50 + Squeeze-and-Excitation)
  - Two-phase training: frozen base + fine-tune top 40 layers
  - Dropout 0.3 + L1 regularizer 1e-4
  - Stronger augmentations
  - Grad-CAM: 3 images per class (original + heatmap + overlay + embossed)
=======================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import os, random, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ============================================================
# PATHS AND CONFIG
# ============================================================
base_path = "/data/vaishnav25/Data"
output_dir = "/data/vaishnav25/ADITHYA/Results_SEResNet50"
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 1000
LR_PHASE1 = 3e-5
LR_PHASE2 = 1e-5
class_names = ["well", "mod", "poor"]

device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f" Using {device}")

# ============================================================
# DATA LOADING + COLOR AUGMENTATION FOR POOR CLASS
# ============================================================
class_paths = [os.path.join(base_path, c.capitalize()) for c in class_names]
all_image_paths, all_labels = [], []

for idx, class_dir in enumerate(class_paths):
    imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if idx < 2:
        imgs = imgs[:500]
    all_image_paths.extend(imgs)
    all_labels.extend([idx] * len(imgs))

datagen_aug = ImageDataGenerator(brightness_range=[0.8, 1.2], channel_shift_range=30)
poor_idx = [i for i, l in enumerate(all_labels) if l == 2]
if len(poor_idx) < 300:
    poor_imgs = [all_image_paths[i] for i in poor_idx]
    augmented_imgs = []
    while len(poor_imgs) + len(augmented_imgs) < 300:
        img_path = np.random.choice(poor_imgs)
        img = load_img(img_path, target_size=IMG_SIZE)
        x = img_to_array(img).reshape((1,) + img_to_array(img).shape)
        for batch in datagen_aug.flow(x, batch_size=1):
            augmented_imgs.append(array_to_img(batch[0]))
            break
    aug_dir = os.path.join(base_path, "Poor_Augmented")
    os.makedirs(aug_dir, exist_ok=True)
    for i, img in enumerate(augmented_imgs):
        aug_path = os.path.join(aug_dir, f"aug_{i}.png")
        img.save(aug_path)
        all_image_paths.append(aug_path)
        all_labels.append(2)

all_image_paths, all_labels = shuffle(np.array(all_image_paths), np.array(all_labels), random_state=42)
print(" Class distribution:")
for i, c in enumerate(class_names):
    print(f"{c}: {np.sum(all_labels==i)}")

# ============================================================
# TRAIN-VAL SPLIT
# ============================================================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

train_df = pd.DataFrame({'filename': train_paths, 'label': train_labels})
val_df = pd.DataFrame({'filename': val_paths, 'label': val_labels})

# ============================================================
# DATA GENERATORS
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    channel_shift_range=25,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='label',
    target_size=IMG_SIZE, class_mode='raw',
    batch_size=BATCH_SIZE, shuffle=True)

val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='label',
    target_size=IMG_SIZE, class_mode='raw',
    batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL DEFINITION
# ============================================================
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Multiply()([input_tensor, Reshape((1, 1, filters))(se)])
    return se

def build_seresnet50(input_shape=(224,224,3), num_classes=3):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False
    x = squeeze_excite_block(base.output)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=l1(1e-4))(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(LR_PHASE1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base

model, base_model = build_seresnet50()

# ============================================================
# PHASE 1: FROZEN BASE
# ============================================================
print("\n Phase 1: Training with frozen base...")
cb1 = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
]
h1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1, callbacks=cb1, verbose=1)

# ============================================================
# PHASE 2: FINE-TUNE TOP 40 LAYERS
# ============================================================
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(LR_PHASE2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n Phase 2: Fine-tuning top 40 layers...")
cb2 = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]
h2 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2, callbacks=cb2, verbose=1)

# ============================================================
# CURVES
# ============================================================
acc = h1.history['accuracy'] + h2.history['accuracy']
val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
loss = h1.history['loss'] + h2.history['loss']
val_loss = h1.history['val_loss'] + h2.history['val_loss']

plt.figure(figsize=(10,4))
plt.plot(acc,label='Train Acc'); plt.plot(val_acc,'--',label='Val Acc')
plt.legend(); plt.title("Accuracy Curve"); plt.savefig(os.path.join(output_dir,"accuracy_curve.png")); plt.close()

plt.figure(figsize=(10,4))
plt.plot(loss,label='Train Loss'); plt.plot(val_loss,'--',label='Val Loss')
plt.legend(); plt.title("Loss Curve"); plt.savefig(os.path.join(output_dir,"loss_curve.png")); plt.close()

# ============================================================
# CONFUSION MATRIX + ROC-AUC
# ============================================================
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs,axis=1)
y_true = val_labels

cm = confusion_matrix(y_true,y_pred)
ConfusionMatrixDisplay(cm,display_labels=class_names).plot(cmap='Blues',values_format='d')
plt.title("Confusion Matrix"); plt.savefig(os.path.join(output_dir,"confusion_matrix.png")); plt.close()

y_true_bin = label_binarize(y_true, classes=[0,1,2])
plt.figure(figsize=(8,6))
for i, c in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_pred_probs[:,i])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=2,label=f"{c} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("Multi-class ROC-AUC")
plt.savefig(os.path.join(output_dir,"roc_auc_curve.png")); plt.close()

print(" Confusion matrix and ROC-AUC plots saved!")

# ============================================================
# GRAD-CAM (3 IMAGES PER CLASS)
# ============================================================
def generate_gradcam(model,img_path,class_index=None,layer_name=None):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = np.expand_dims(img_to_array(img)/255., axis=0)
    preds = model.predict(x)
    if class_index is None: class_index = np.argmax(preds[0])
    if layer_name is None: layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs, weights.numpy())
    cam = np.maximum(cam, 0); cam /= cam.max()+1e-8
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    orig = cv2.imread(img_path); orig = cv2.resize(orig, IMG_SIZE)
    overlay = cv2.addWeighted(orig,0.6,heatmap,0.4,0)
    embossed = cv2.addWeighted(orig,0.5,heatmap,0.5,0)
    return orig[:,:,::-1], heatmap[:,:,::-1], overlay[:,:,::-1], embossed[:,:,::-1]

print(" Generating Grad-CAMs (3 per class)...")
for c_idx, cname in enumerate(class_names):
    samples = val_df[val_df['label']==c_idx].sample(3, random_state=42)
    for _, row in samples.iterrows():
        img_path = row['filename']
        orig, heat, over, emb = generate_gradcam(model, img_path)
        fig, ax = plt.subplots(1,4,figsize=(16,5))
        titles = ['Original','Heatmap','Overlay','Embossed']
        for i, im in enumerate([orig, heat, over, emb]):
            ax[i].imshow(im); ax[i].set_title(titles[i]); ax[i].axis('off')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"gradcam_{cname}_{os.path.basename(img_path)}.png")
        plt.savefig(save_path); plt.close()
        print(f" Saved {save_path}")

# ============================================================
# SAVE MODEL
# ============================================================
model.save(os.path.join(output_dir,"seresnet50_final.keras"))
print(f"\n Training + Grad-CAM complete! Model saved at: {output_dir}")
