import os
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset import load_dataset
from model import build_model

# Enable XLA and mixed precision for speedup

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy("float32")

# Load dataset
X_train, X_test, y_train, y_test = load_dataset()

# Optimize dataset pipeline for speed
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128).prefetch(tf.data.AUTOTUNE)

# Data Augmentation
aug = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Build model
model = build_model()

# Learning Rate Warmup
initial_learning_rate = 0.001
lr_schedule = CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=30000, alpha=0.0001)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5, amsgrad=True)



model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00005, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("models/best_captcha_solver.keras", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

os.makedirs("models", exist_ok=True)
model.save("models/captcha_solver_final.keras")
print("Model training complete and saved!")
