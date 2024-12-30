import time
import cv2
import easyocr
import os
import serial
import requests
import pyrebase
from PIL import Image
from ultralytics import YOLO

# Cấu hình Firebase
firebase_config = {
    "apiKey": "AIzaSyAijDn8PFMEoZR-1rE0knfsdGYWkqV_sNs",
    "authDomain": "fir-cloud-9b508.firebaseapp.com",
    "databaseURL": "https://fir-cloud-9b508-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "fir-cloud-9b508",
    "storageBucket": "fir-cloud-9b508.firebasestorage.app",
    "messagingSenderId": "92325678928",
    "appId": "1:92325678928:web:6875825d7fdfebf7702655",
    "measurementId": "G-2H2G8BJMG5"
}

# Khởi tạo Firebase
firebase = pyrebase.initialize_app(firebase_config)

# Truy cập Realtime Database
db = firebase.database()

# Trạng thái button
buttonState = ""

# Hàm xử lý stream khi có thay đổi
def stream_handler(message):
    global buttonState
    print(f"Event: {message['event']}")  # Loại sự kiện (put, patch, delete)
    print(f"Path: {message['path']}")    # Đường dẫn đến dữ liệu thay đổi
    print(f"Data: {message['data']}")    # Giá trị thay đổi hoặc None nếu bị xóa

    # In ra giá trị mới nếu có sự thay đổi
    if message['data'] is not None:
        buttonState = message['data']
        print(f"Giá trị mới tại {message['path']}: {message['data']}")

def segment_image(image):
    # Convert the license plate image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # Apply GaussianBlur to reduce noise
    # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    #
    # Apply threshold to separate numbers and background
    # ret, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display processed image (optional)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Use EasyOCR to recognize text from processed image
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)

    # Create a dictionary to convert lowercase characters to desired characters
    conversion_dict = {
        'l': '1', 'o': '0', 'i': '1', 'g': '9',
        'a': '9', 'b': '6', 's': '5', 'q': '9'
    }

    # Convert text according to the dictionary
    def convert_text(text):
        filtered_text = ''.join([char if char.isalnum() else ' ' for char in text])
        return ''.join([conversion_dict.get(char, char) for char in filtered_text])

    # Convert and format results
    formatted_results = []
    for result in results:
        text = result[1]  # Extract recognized text
        converted_text = convert_text(text)  # Convert text
        formatted_results.append(converted_text)

    # Join results into a formatted string
    output = ' '.join(formatted_results)
    print(output)
    return output

# Set up a pretrained YOLO model
model = YOLO("runs/detect/train22/weights/best.pt")

# Load and process an image
results = model("test.jpg")
original_image = cv2.imread("test.jpg")

# Show detection results
for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save("kq.jpg")

# Loop through detection results to crop detected regions
for r in results:
    for box in r.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]

        # Convert to integer if necessary
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop the detected region from the original image
        cropped_image = original_image[y1:y2, x1:x2]

        # Save or display the cropped image
        cv2.imwrite("license_plate_cropped.jpg", cropped_image)

        # Open and display the cropped image
        im = Image.open("license_plate_cropped.jpg")
        im.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def send_api_request(API_URL):
    """
    Gửi yêu cầu GET đến API và in dữ liệu phản hồi.
    """
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            print("Dữ liệu nhận được từ API:")
            print(response.json())  # In dữ liệu nhận được
        else:
            print(f"Lỗi khi gửi yêu cầu GET. Mã trạng thái: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gửi yêu cầu: {e}")

def take_photo_and_send(url):
    # Mở Webcam
    cap = cv2.VideoCapture(0)

    output_dir = '.'

    if not cap.isOpened():
        print("Không thể mở camera")
        exit()

    ret, frame = cap.read()

    if ret:
        image_path = os.path.join(output_dir, 'test.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Đã lưu ảnh tại {image_path}")
    else:
        print("Không chụp được ảnh")

    # File to upload
    file_path = image_path

    # Prepare the files parameter
    files = {
        "image": open(file_path, "rb")
    }

    # Optional: Additional data to send
    data = {
        "user_id": "12345"
    }

    # Make the request
    response = requests.post(url, files=files, data=data)

    # Check the response
    if response.status_code == 200:
        print("Image uploaded successfully:", response.json())
    else:
        print(f"Failed to upload image. Status code: {response.status_code}")


# Lắng nghe sự thay đổi tại nhánh `status/checking`
my_stream = db.child("status/door1/isOpen").stream(stream_handler)

# Read the cropped license plate image
image_path = "license_plate_cropped.jpg"
image = cv2.imread(image_path)

# Segment and recognize text
segment_image(image)

# URL của API
url = "http://127.0.0.1:5000/vehicle/handle"  # Thay đổi thành URL thực tế của bạn

# Đường dẫn file cần upload
file_path = "./test.jpg"  # Thay đổi thành đường dẫn file thực tế

# Nội dung message
license_plate = "52 P2 86 151"
status = "enter"

# Mở file và gửi request
with open(file_path, "rb") as file:
    # Tạo payload với file và messages
    files = {"image": file}
    data = {"licensePlate": license_plate, "status": status}

    try:
        # Gửi request POST
        response = requests.post(url, files=files, data=data)

        # Xử lý kết quả trả về
        if response.status_code == 200:
            print("Upload thành công:", response.json())
        else:
            print("Upload thất bại:", response.json())
    except Exception as e:
        print("Lỗi khi gọi API:", str(e))