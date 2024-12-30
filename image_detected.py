import time
from re import match

import cv2
import easyocr
import os
import serial
import multiprocessing
import shutil
import pyrebase
import requests
import firebase_admin
from PIL import Image
from sympy import false
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

# Tạo danh sách biển số xe
list_license_plate = []

# Tạo danh sách biển số đã scan
list_license = []

# Mở Webcam
cap = cv2.VideoCapture(1)

# Trạng thái
checking = ""
isOpenDoor1 = false
isOpenDoor2 = false

# Hàm xử lý stream khi có thay đổi
def stream_handler_checking(message):
    global checking
    print(f"Event: {message['event']}")  # Loại sự kiện (put, patch, delete)
    print(f"Path: {message['path']}")  # Đường dẫn đến dữ liệu thay đổi
    print(f"Data: {message['data']}")  # Giá trị thay đổi hoặc None nếu bị xóa

    # In ra giá trị mới nếu có sự thay đổi
    if message['data'] is not None:
        checking = message['data']
        arduino.write(f"{checking} p".encode())
        print(f"Giá trị mới tại {message['path']}: {message['data']}")

# Hàm xử lý stream khi có thay đổi
def stream_handler_door1(message):
    global isOpenDoor1
    print(f"Event: {message['event']}")  # Loại sự kiện (put, patch, delete)
    print(f"Path: {message['path']}")  # Đường dẫn đến dữ liệu thay đổi
    print(f"Data: {message['data']}")  # Giá trị thay đổi hoặc None nếu bị xóa

    # In ra giá trị mới nếu có sự thay đổi
    if message['data'] is not None:
        isOpenDoor1 = message['data']
        print(f"door1_{isOpenDoor1} p")
        arduino.write(f"door1_{isOpenDoor1} p".encode())
        print(f"Giá trị mới tại {message['path']}: {message['data']}")

# Hàm xử lý stream khi có thay đổi
def stream_handler_door2(message):
    global isOpenDoor2
    print(f"Event: {message['event']}")  # Loại sự kiện (put, patch, delete)
    print(f"Path: {message['path']}")  # Đường dẫn đến dữ liệu thay đổi
    print(f"Data: {message['data']}")  # Giá trị thay đổi hoặc None nếu bị xóa

    # In ra giá trị mới nếu có sự thay đổi
    if message['data'] is not None:
        isOpenDoor2 = message['data']
        print(f"door2_{isOpenDoor2} p")
        arduino.write(f"door2_{isOpenDoor2} p".encode())
        print(f"Giá trị mới tại {message['path']}: {message['data']}")


# Khai báo hàm quét hình ảnh theo yêu cầu check in hoặc check out
def scan_image(crop):
    test_image = cv2.imread("test.jpg")

    if crop == "left":

        # Lấy chiều cao và chiều rộng của hình ảnh
        height, width, _ = test_image.shape

        # Chia ảnh thành 2 phần theo chiều dọc (trái và phải)
        original_image = test_image[:, :width // 2]  # Phần bên trái
        # right_image = test_image[:, width // 2:]  # Phần bên phải

    else:
        # Lấy chiều cao và chiều rộng của hình ảnh
        height, width, _ = test_image.shape

        # Chia ảnh thành 2 phần theo chiều dọc (trái và phải)
        # left_image = image[:, :width // 2]  # Phần bên trái
        original_image = test_image[:, width // 2:]  # Phần bên phải

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("runs/detect/train22/weights/best.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model(original_image)

    # Show the results
    for r in results:
        print(r.boxes)
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

        im.show()

        im.save("kq.jpg")

    # Loop through the detection results
    for r in results:
        for box in r.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # xyxy format

            # Convert to integer (if necessary)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop the detected region from the original image
            cropped_image = original_image[y1:y2, x1:x2]

            # Save or display the cropped image
            cv2.imwrite("license_plate_cropped.jpg", cropped_image)

            # Mở hình ảnh
            im = Image.open("license_plate_cropped.jpg")


# Khai báo hàm đọc ký tự từ hình ảnh
def segment_image(image):
    global checking
    # Sử dụng easyocr để nhận diện ký tự từ ảnh đã xử lý
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)

    # Tạo một từ điển chuyển đổi các ký tự thường thành ký tự mong muốn
    conversion_dict = {
        'l': '1',
        'o': '0',
        'i': '1',
        'g': '9',
        'a': '9',
        'b': '6',
        's': '5',
        'q': '9',
    }

    def convert_text(text):
        filtered_text = ''.join([char if char.isalnum() else ' ' for char in text])
        filtered_text = filtered_text.replace(' ', '')
        return ''.join([conversion_dict.get(char, char) for char in filtered_text])

    formatted_results = []
    for result in results:
        text = result[1]
        converted_text = convert_text(text)
        formatted_results.append(converted_text)

    output = ' '.join(formatted_results) + " s"

    if output in ["  s", " s"]:
        stop_signal = "RETURN s"
        arduino.write(stop_signal.encode())
        print("Nhận diện lỗi!!!")
        return

    print(output)

    # Gửi dữ liệu đến Arduino
    data_to_send = output
    arduino.write(data_to_send.encode())

    # Đọc phản hồi từ Arduino
    response = arduino.readline().decode('utf-8').strip()
    print(f"Arduino says: {response}")

    check_license_plate = arduino.readline().decode('utf-8').strip()

    if check_license_plate == "REMOVE_CAR":
        ID = arduino.readline().decode('utf-8').strip()
        check_all = ID + " " + response

        print("Xóa xe")
        print(check_all)

        if check_all in list_license_plate:
            sendData(f"license/{check_all}.jpg", response, ID, "exit")
            list_license_plate.remove(check_all)
            list_license.remove(response)
            os.remove(f"license/{check_all}.jpg")
            print("Xóa thành công")
            arduino.write("ACCEPT s".encode())

        else:
            print("Biển số xe không khớp")
            matching_entry = None
            for entry in list_license_plate:
                current_rfid, license_plate = entry.split(" ", 1)
                if current_rfid == ID:
                    matching_entry = entry
                    image = Image.open(f"license/{matching_entry}.jpg")
                    sendData("kq.jpg", response, ID, "conflict")
                    image.show()
                    break
            arduino.write("DENIED s".encode())

    if check_license_plate == "ADD_CAR":
        ID = arduino.readline().decode('utf-8').strip()
        check_all = ID + " " + response

        print("Thêm xe")
        print(check_all)

        if response in list_license:
            if check_all in list_license_plate:
                print("Xe đã tồn tại")
                arduino.write("UNAVAILABLE s".encode())
            else:
                print("Biển số này đã được đăng ký RFID")
                arduino.write("UNAVAILABLE s".encode())
        else:
            list_license.append(response)
            list_license_plate.append(check_all)
            check_all_image = cv2.imread("kq.jpg")
            file_path = os.path.join("license", f"{check_all}.jpg")
            cv2.imwrite(file_path, check_all_image)
            sendData(f"license/{check_all}.jpg", response, ID, "enter")
            print("Thêm thành công")
            arduino.write("AVAILABLE s".encode())


# =======================================================================================================================

# Thiết lập kết nối serial với Arduino
arduino = serial.Serial('COM4', 9600)

# Đường dẫn đến thư mục cần xóa tệp
directory = "license"

# Xóa toàn bộ tệp trong thư mục
shutil.rmtree(directory)
os.mkdir(directory)

print("Camera đã khởi chạy!!!")

# Lắng nghe sự thay đổi tại nhánh `status/checking`
my_stream_checking = db.child("status/checking").stream(stream_handler_checking)

# Lắng nghe sự thay đổi tại nhánh `status/checking`
my_stream_door1 = db.child("status/door1/isOpen").stream(stream_handler_door1)

# Lắng nghe sự thay đổi tại nhánh `status/checking`
my_stream_door2 = db.child("status/door2/isOpen").stream(stream_handler_door2)

# URL của API
url = "http://127.0.0.1:5000/vehicle/handle"  # Thay đổi thành URL thực tế của bạn

def sendData(file_path, license_plate, rfid, status):
    # Mở file và gửi request
    with open(file_path, "rb") as file:
        # Tạo payload với file và messages
        files = {"image": file}
        data = {"licensePlate": license_plate, "rfid": rfid, "status": status}

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

while True:

    begin = arduino.readline().decode('utf-8').strip()
    if begin == "RUN":
        while (checking == ""):
            if (checking != ""):
                print(checking)

        print(checking)
        arduino.write(f"{checking} p".encode())

        signal = ""
        while signal == "":
            signal = arduino.readline().decode('utf-8').strip()
        print(signal)

        if signal == "CHANGE":
            if (checking == "true"):
                checking = "false"
                db.child("status/checking").set(checking)
            else:
                checking = "true"
                db.child("status/checking").set(checking)


        if signal == "LOOP":
            print("")

        if signal == "RUN_CHECK":
            print(signal)
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

            check_signal = arduino.readline().decode('utf-8').strip()

            if check_signal == "ADD_CAR":
                scan_image("left")
            elif check_signal == "REMOVE_CAR":
                scan_image("right")
            else:
                arduino.write("RETURN s".encode())

            image_path = "license_plate_cropped.jpg"
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                segment_image(image)
                os.remove("license_plate_cropped.jpg")
                os.remove("kq.jpg")
                os.remove("test.jpg")
            else:
                print("Không nhận diện được biển số xe nào trong khung hình.")
                os.remove("kq.jpg")
                os.remove("test.jpg")
                arduino.write("RETURN s".encode())


        if signal == "RUN_SKIP":
            print(signal)
            RFID = arduino.readline().decode('utf-8').strip()
            list_check = []
            for check_all in list_license_plate:
                if RFID in check_all:
                    list_check.append(check_all)
                    print("Xóa xe")
                    print(check_all)
                    list_license_plate.remove(check_all)
                    os.remove(f"license/{check_all}.jpg")
                    check_all_license = check_all[9:]
                    for check_all in list_license:
                        if check_all_license in check_all:
                            list_check.append(check_all)
                            list_license.remove(check_all)

            if not list_check:
                arduino.write("DENIED s".encode())
            else:
                arduino.write("ACCEPT s".encode())