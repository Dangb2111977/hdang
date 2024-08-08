import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract
import mysql.connector
from mysql.connector import Error

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame.")
        return None

    return frame

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def extract_text_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_image, config='--psm 6')
    return text

def decode_text(text):
    lines = text.split('\n')
    school_name = lines[0].strip() if len(lines) > 0 else "Unknown School"
    student_name = lines[1].strip() if len(lines) > 1 else "Unknown Name"
    student_code = lines[2].strip() if len(lines) > 2 else "Unknown Code"
    major = lines[3].strip() if len(lines) > 3 else "Unknown Major"
    course = int(lines[4].strip()) if len(lines) > 4 and lines[4].strip().isdigit() else -1

    return school_name, student_name, student_code, major, course

def store_information(school_name, student_name, student_code, major, course):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='face_recognition',
            user='root',
            password='Tieuquangdethuong2811@'
        )

        if connection.is_connected():
            cursor = connection.cursor()

            insert_query = '''INSERT INTO checkin_checkout (school_name, student_name, student_code, major, course)
                              VALUES (%s, %s, %s, %s, %s)'''
            record = (school_name, student_name, student_code, major, int(course))
            print(f"Inserting into database: School Name: {school_name}, Student Name: {student_name}, Student Code: {student_code}, Major: {major}, Course: {course}")
            cursor.execute(insert_query, record)
            connection.commit()

    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

try:
    # Load the saved model
    model = load_model('student_card_model.h5')
    print("Model loaded successfully.")

    # Capture an image
    image = capture_image()
    if image is not None:
        cv2.imshow('Captured Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        print("Image preprocessed successfully.")

        # Make predictions
        predictions = model.predict(preprocessed_image)
        print("Predictions:", predictions)
        print("Predicted label shape:", predictions.shape)

        # Extract text from image using OCR
        extracted_text = extract_text_from_image(image)
        print("Extracted Text:", extracted_text)

        # Decode the extracted text
        school_name, student_name, student_code, major, course = decode_text(extracted_text)

        # Store the extracted information in the MySQL database
        store_information(school_name, student_name, student_code, major, course)
        print("Information stored in the database successfully.")
    else:
        print("No image captured.")
except Exception as e:
    print(f"An error occurred: {e}")
