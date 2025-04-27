import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image

# Paths for training, validation, and testing
dataset_path = './split_dataset/'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# Input configurations
input_size = 128
batch_size = 32
epochs = 20

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# Data generators for each dataset
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(input_size, input_size), batch_size=batch_size, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_path, target_size=(input_size, input_size), batch_size=batch_size, class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(input_size, input_size), batch_size=1, class_mode=None, shuffle=False
)

# Define model
efficient_net = EfficientNetB0(weights='imagenet', input_shape=(input_size, input_size, 3), include_top=False, pooling='max')
model = Sequential([
    efficient_net,
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_path = './model_checkpoint'
os.makedirs(checkpoint_path, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
    ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'best_model.keras'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
]

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)

# Save training results
print(history.history)

# Load the best model
best_model = load_model(os.path.join(checkpoint_path, 'best_model.keras'))

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Predict on test set
test_generator.reset()
predictions = best_model.predict(test_generator, verbose=1)

# Define paths
deepfake_folder = "./tmp_fake_faces/"
xls_path = './predictions.xlsx'

# Ensure deepfake image folder exists
if not os.path.exists(deepfake_folder):
    print(f"Deepfake image folder '{deepfake_folder}' does not exist!")
    exit()

# Assuming predictions are already generated
filenames = test_generator.filenames  # Get filenames from test_generator
predictions = best_model.predict(test_generator, verbose=1).flatten()  # Get predictions

# Construct deepfake image paths
deepfake_paths = [os.path.join(deepfake_folder, os.path.basename(f)) for f in filenames]

# Create DataFrame
results = pd.DataFrame({
    "Filename": filenames,
    "Prediction": predictions,
    "Deepfaked_Image_Path": deepfake_paths  # Adding path for verification
})

# Save DataFrame as Excel
results.to_excel(xls_path, index=False, engine='openpyxl')

# Load workbook and sheet
wb = openpyxl.load_workbook(xls_path)
ws = wb.active

# Insert images into the Excel file (column C)
for i, deepfake_path in enumerate(results["Deepfaked_Image_Path"], start=2):  # Start from row 2
    if os.path.exists(deepfake_path):  # Ensure the image exists
        # Load and resize image
        img = Image.open(deepfake_path)
        img = img.resize((100, 100))  # Resize to fit in Excel
        
        # Save with a unique filename for each row
        temp_image_path = f"/fake/temp_image_{i}.png"
        img.save(temp_image_path)  

        # Insert into Excel
        img_excel = OpenpyxlImage(temp_image_path)
        ws.add_image(img_excel, f"C{i}")  # Column C for images

# Save the modified Excel file
wb.save(xls_path)

print(f"Predictions with embedded images saved to '{xls_path}'")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get true labels
true_labels = test_generator.classes  

# Get predicted labels (threshold = 0.5 for binary classification)
predicted_labels = (predictions.flatten() > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(true_labels, predicted_labels))
