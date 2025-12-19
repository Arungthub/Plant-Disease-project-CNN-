import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5

BASE_DIR = r"C:\Users\arunk\Downloads\Plant Disease project\PlantVillage_Split"

train_dir = os.path.join(BASE_DIR, "train")
val_dir   = os.path.join(BASE_DIR, "validation")
test_dir  = os.path.join(BASE_DIR, "test")

# ✅ Get class names ONLY from train folder
class_names = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])

NUM_CLASSES = len(class_names)

print("Classes used:", class_names)
print("Number of classes:", NUM_CLASSES)

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=class_names
)

val_data = val_test_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=class_names
)

test_data = val_test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    classes=class_names
)

# CNN Model
model = Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# Evaluate
test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)

# Predictions
y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes
from sklearn.metrics import classification_report

labels = np.unique(y_true)
valid_class_names = [class_names[i] for i in labels]

print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=valid_class_names
))

print(confusion_matrix(y_true, y_pred))

# Save model and classes
model.save("plant_disease_model.h5")
np.save("class_names.npy", class_names)

# Accuracy plot
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.title("Accuracy")
plt.show()

# Loss plot
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.legend()
plt.title("Loss")
plt.show()
