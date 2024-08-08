import pytesseract
from PIL import Image

# Cấu hình đường dẫn tới tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn tới hình ảnh bạn muốn nhận diện
image_path = 'C:\\Users\\Lehuy\\Downloads\\NLCN images\\451114393_1209628690190101_6357601146431663318_n.jpg'

# Mở hình ảnh bằng PIL
image = Image.open(image_path)

# Sử dụng pytesseract để nhận diện văn bản từ hình ảnh
text = pytesseract.image_to_string(image)

# In ra văn bản nhận diện được
print(text)
