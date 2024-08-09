import os
import numpy as np
import pytesseract
from pytesseract import Output
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to load images from the 'images' folder
def load_data(directory):
    images_dir = os.path.join(directory, 'images')
    images = []
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(images_dir, filename)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)

# Function to extract text from an image using OCR
def ocr_extract_text(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    custom_config = r'--oem 3 --psm 6'
    ocr_result = pytesseract.image_to_string(image, config=custom_config)
    return ocr_result

# Function to process labels using OCR and filter out images without text
def process_labels_with_ocr(images):
    filtered_images = []
    processed_labels = []
    
    for img in images:
        text = ocr_extract_text(img).strip()
        if text:  # Only add non-empty text and corresponding images
            processed_labels.append(text)
            filtered_images.append(img)
    
    return np.array(filtered_images), np.array(processed_labels)

# Load the data from the 'images' folder within 'data'
images = load_data("D:\\Niên luận chuyên ngành\\data")

# Process the labels using OCR and filter images
filtered_images, labels_processed = process_labels_with_ocr(images)

# Check if there are any extracted labels
if labels_processed.size == 0:
    raise ValueError("No labels were extracted from the images.")

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_processed)
labels_categorical = to_categorical(labels_encoded)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(filtered_images, labels_categorical, test_size=0.2, random_state=42)

# Define the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(y_train.shape[1], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('student_card_model2.h5')
