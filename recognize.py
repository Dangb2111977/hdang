import cv2
import numpy as np
import tensorflow as tf
import pytesseract
import mysql.connector
from mysql.connector import Error
from datetime import datetime

# Cấu hình đường dẫn tới tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn mô hình
model_path = 'studentcardmodeltwo.h5'

# Tải mô hình đã lưu
try:
    model = tf.keras.models.load_model(model_path)
    print("Mô hình đã được tải thành công.")
except Exception as e:
    print(f"Không thể tải mô hình: {e}")
    raise

def preprocess_image(image):
    """Tiền xử lý hình ảnh để phù hợp với đầu vào của mô hình."""
    image = cv2.resize(image, (224, 224))  # Resize hình ảnh
    image = image / 255.0  # Chuẩn hóa giá trị pixel
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    return image

def predict(image):
    """Dự đoán lớp của hình ảnh sử dụng mô hình."""
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions

def extract_info_from_lines(text_lines):
    """Trích xuất thông tin từ các dòng văn bản."""
    school_name = extract_school_name(text_lines)
    student_name = extract_student_name(text_lines)
    student_id = extract_student_id(text_lines)
    major = extract_major(text_lines)
    course_year = extract_course_year(text_lines)
    return student_id, student_name, school_name, major, course_year

def extract_school_name(text_lines):
    return text_lines[1].split(":")[-1].strip() if len(text_lines) > 1 else 'Unknown'

def extract_student_name(text_lines):
    return text_lines[3].split(":")[-1].strip() if len(text_lines) > 3 else 'Unknown'

def extract_student_id(text_lines):
    return text_lines[5].split(":")[-1].strip() if len(text_lines) > 5 else 'Unknown'

def extract_major(text_lines):
    return text_lines[6].split(":")[-1].strip() if len(text_lines) > 6 else 'Unknown'

def extract_course_year(text_lines):
    return text_lines[8].split(":")[-1].strip() if len(text_lines) > 8 else 'Unknown'

try:
    # Kết nối cơ sở dữ liệu
    conn = mysql.connector.connect(
        user='root',
        password='Tieuquangdethuong2811@',
        host='localhost',
        database='face_recognition'
    )

    if conn.is_connected():
        print("Kết nối thành công đến cơ sở dữ liệu")
        
        cursor = conn.cursor()
        
        # Tạo bảng nếu chưa tồn tại
        create_table_query = """
        CREATE TABLE IF NOT EXISTS checkin_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            school_name VARCHAR(100) NOT NULL,
            student_name VARCHAR(100) NOT NULL,
            student_id VARCHAR(50) NOT NULL,
            major VARCHAR(100) NOT NULL,
            course_year VARCHAR(10) NOT NULL,
            checkin_time DATETIME NOT NULL
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        print("Bảng đã được tạo hoặc đã tồn tại")
    
    else:
        print("Kết nối không thành công")

    # Mở camera
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Không thể mở video stream")
        raise Exception("Cannot open connection to camera")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Không thể đọc video")
            break

        # Tiền xử lý và dự đoán
        predictions = predict(frame)
        print("Dự đoán:", predictions)  # In ra dự đoán

        # Chuyển đổi hình ảnh sang thang độ xám và nhận diện văn bản
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        print("Extracted Text:", text)  # In ra văn bản trích xuất

        # Phân tách văn bản thành các dòng
        text_lines = text.split('\n')

        # Kiểm tra và phân tích thông tin thẻ sinh viên
        if len(text_lines) >= 8:  # Đảm bảo có đủ dòng
            student_id, student_name, school_name, major, course_year = extract_info_from_lines(text_lines)

            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")

            try:
                # Ghi lại thời gian vào cơ sở dữ liệu
                cursor.execute(
                    "INSERT INTO checkin_log (student_id, student_name, school_name, major, course_year, checkin_time) VALUES (%s, %s, %s, %s, %s, %s)", 
                    (student_id, student_name, school_name, major, course_year, current_time)
                )
                conn.commit()
                print("Data inserted successfully")
            except Error as err:
                print(f"Database error: {err}")

        else:
            print("Text does not contain enough lines")  # Văn bản không đủ các dòng cần thiết

        # Hiển thị video
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Error as err:
    print(f"Đã xảy ra lỗi khi kết nối hoặc thực thi mã: {err}")

finally:
    # Giải phóng camera và đóng kết nối cơ sở dữ liệu
    if 'video_capture' in locals():
        video_capture.release()
        print("Camera đã được giải phóng")
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print("Kết nối đã được đóng")
    cv2.destroyAllWindows()
