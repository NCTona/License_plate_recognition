import time
import cv2
import easyocr
import os
import serial
import shutil
import pyrebase
import requests
from PIL import Image, ImageTk
from sympy import false
from ultralytics import YOLO
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue

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
cap = cv2.VideoCapture(0)

# Trạng thái
checking = ""
isOpenDoor1 = false
isOpenDoor2 = false

# Hàng đợi để giao tiếp giữa luồng serial và GUI
serial_queue = queue.Queue()

# Hàm ghi log vào GUI
def log_message(message, root, log_text):
    log_text.configure(state='normal')
    log_text.insert(tk.END, f"{message}\n")
    log_text.see(tk.END)
    log_text.configure(state='disabled')
    root.update()

# Hàm cập nhật hình ảnh kq.jpg
def update_image(root, image_label):
    try:
        img = Image.open("kq.jpg")
        img = img.resize((325, 550), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_label.configure(image=photo)
        image_label.image = photo  # Giữ tham chiếu
        root.update()
    except:
        pass  # Không có hình ảnh thì bỏ qua

# Hàm xử lý stream khi có thay đổi
def stream_handler_checking(message, root, log_text, arduino):
    global checking
    log_message(f"Event: {message['event']}", root, log_text)
    log_message(f"Path: {message['path']}", root, log_text)
    log_message(f"Data: {message['data']}", root, log_text)
    if message['data'] is not None:
        checking = message['data']
        arduino.write(f"{checking} \n".encode())
        log_message(f"Giá trị mới tại {message['path']}: {message['data']}", root, log_text)

# Hàm xử lý stream khi có thay đổi
def stream_handler_door1(message, root, log_text, arduino):
    global isOpenDoor1
    log_message(f"Event: {message['event']}", root, log_text)
    log_message(f"Path: {message['path']}", root, log_text)
    log_message(f"Data: {message['data']}", root, log_text)
    if message['data'] is not None:
        isOpenDoor1 = message['data']
        log_message(f"door1_{isOpenDoor1}", root, log_text)
        arduino.write(f"door1_{isOpenDoor1} \n".encode())
        log_message(f"Giá trị mới tại {message['path']}: {message['data']}", root, log_text)

# Hàm xử lý stream khi có thay đổi
def stream_handler_door2(message, root, log_text, arduino):
    global isOpenDoor2
    log_message(f"Event: {message['event']}", root, log_text)
    log_message(f"Path: {message['path']}", root, log_text)
    log_message(f"Data: {message['data']}", root, log_text)
    if message['data'] is not None:
        isOpenDoor2 = message['data']
        log_message(f"door2_{isOpenDoor2}", root, log_text)
        arduino.write(f"door2_{isOpenDoor2} \n".encode())
        log_message(f"Giá trị mới tại {message['path']}: {message['data']}", root, log_text)

# Khai báo hàm quét hình ảnh theo yêu cầu check in hoặc check out
def scan_image(crop, root, image_label):
    test_image = cv2.imread("test.jpg")
    if crop == "left":
        height, width, _ = test_image.shape
        original_image = test_image[:, :width // 2]
    else:
        height, width, _ = test_image.shape
        original_image = test_image[:, width // 2:]
    model = YOLO("runs/detect/train22/weights/best.pt")
    results = model(original_image)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save("kq.jpg")
        update_image(root, image_label)  # Cập nhật hình ảnh trên GUI
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_image = original_image[y1:y2, x1:x2]
            cv2.imwrite("license_plate_cropped.jpg", cropped_image)

# Khai báo hàm đọc ký tự từ hình ảnh
def segment_image(image, root, log_text, arduino):
    global checking
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)
    conversion_dict = {
        'l': '1', 'o': '0', 'i': '1', 'g': '9', 'a': '9', 'b': '6', 's': '5', 'q': '9',
    }
    def convert_text(text):
        filtered_text = ''.join([char if char.isalnum() else ' ' for char in text])
        filtered_text = filtered_text.replace(' ', '')
        return ''.join([conversion_dict.get(char, char) for char in filtered_text])
    formatted_results = []
    for result in results:
        text = result[1]
        converted_text = convert_text(text)
        formatted_results.append(converted_text.upper())
    output = ' '.join(formatted_results) + " \n"
    if output in ["  \n", " \n"]:
        stop_signal = "RETURN \n"
        arduino.write(stop_signal.encode())
        log_message("Nhận diện lỗi!!!", root, log_text)
        return
    log_message(output.strip(), root, log_text)
    data_to_send = output
    arduino.write(data_to_send.encode())
    response = arduino.readline().decode('utf-8').strip()
    log_message(f"Arduino says: {response}", root, log_text)
    check_license_plate = arduino.readline().decode('utf-8').strip()
    if check_license_plate == "REMOVE_CAR":
        ID = arduino.readline().decode('utf-8').strip()
        check_all = ID + " " + response
        log_message("Xóa xe", root, log_text)
        log_message(check_all, root, log_text)
        if check_all in list_license_plate:
            sendData("kq.jpg", response, ID, "exit", root, log_text)
            list_license_plate.remove(check_all)
            list_license.remove(response)
            os.remove(f"license/{check_all}.jpg")
            log_message("Xóa thành công", root, log_text)
            arduino.write("ACCEPT \n".encode())
        else:
            log_message("Biển số xe không khớp", root, log_text)
            matching_entry = None
            for entry in list_license_plate:
                current_rfid, license_plate = entry.split(" ", 1)
                if current_rfid == ID:
                    matching_entry = entry
                    sendData("kq.jpg", response, ID, "conflict", root, log_text)
                    break
            arduino.write("DENIED \n".encode())
    if check_license_plate == "ADD_CAR":
        ID = arduino.readline().decode('utf-8').strip()
        check_all = ID + " " + response
        log_message("Thêm xe", root, log_text)
        log_message(check_all, root, log_text)
        if response in list_license:
            if check_all in list_license_plate:
                log_message("Xe đã tồn tại", root, log_text)
                arduino.write("UNAVAILABLE \n".encode())
            else:
                log_message("Biển số này đã được đăng ký RFID", root, log_text)
                arduino.write("UNAVAILABLE \n".encode())
        else:
            list_license.append(response)
            list_license_plate.append(check_all)
            check_all_image = cv2.imread("kq.jpg")
            file_path = os.path.join("license", f"{check_all}.jpg")
            cv2.imwrite(file_path, check_all_image)
            sendData(f"license/{check_all}.jpg", response, ID, "enter", root, log_text)
            log_message("Thêm thành công", root, log_text)
            arduino.write("AVAILABLE \n".encode())

# Hàm gửi dữ liệu đến API
def sendData(file_path, license_plate, rfid, status, root, log_text):
    with open(file_path, "rb") as file:
        files = {"image": (file_path, file, "image/jpeg")}
        data = {
            "licensePlate": license_plate,
            "rfid": rfid,
            "status": status
        }
        try:
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                log_message("Upload thành công: " + str(response.json()), root, log_text)
            else:
                log_message(f"Upload thất bại: {response.status_code} {response.text}", root, log_text)
        except Exception as e:
            log_message(f"Lỗi khi gọi API: {str(e)}", root, log_text)

# Hàm đọc serial liên tục (luồng phụ)
def serial_reader(root, log_text, image_label, arduino):
    global checking
    while True:
        try:
            begin = arduino.readline().decode('utf-8').strip()
            serial_queue.put(('log', begin))
            if begin == "RUN":
                while checking == "":
                    serial_queue.put(('log', "Wait checking!!!"))
                    time.sleep(1)
                serial_queue.put(('log', checking))
                arduino.write(f"{checking} \n".encode())
                signal = ""
                while signal == "":
                    signal = arduino.readline().decode('utf-8').strip()
                serial_queue.put(('log', signal))
                if signal == "CHANGE":
                    if checking == "true":
                        checking = "false"
                        db.child("status/checking").set(checking)
                    else:
                        checking = "true"
                        db.child("status/checking").set(checking)
                elif signal == "LOOP":
                    serial_queue.put(('log', ""))
                elif signal == "RUN_CHECK":
                    serial_queue.put(('log', signal))
                    output_dir = '.'
                    if not cap.isOpened():
                        serial_queue.put(('log', "Không thể mở camera"))
                        return
                    ret, frame = cap.read()
                    if ret:
                        image_path = os.path.join(output_dir, 'test.jpg')
                        cv2.imwrite(image_path, frame)
                        serial_queue.put(('log', f"Đã lưu ảnh tại {image_path}"))
                    else:
                        serial_queue.put(('log', "Không chụp được ảnh"))
                    check_signal = arduino.readline().decode('utf-8').strip()
                    if check_signal == "ADD_CAR":
                        scan_image("left", root, image_label)
                    elif check_signal == "REMOVE_CAR":
                        scan_image("right", root, image_label)
                    else:
                        arduino.write("RETURN \n".encode())
                    image_path = "license_plate_cropped.jpg"
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        segment_image(image, root, log_text, arduino)
                        os.remove("license_plate_cropped.jpg")
                        os.remove("kq.jpg")
                        os.remove("test.jpg")
                    else:
                        serial_queue.put(('log', "Không nhận diện được biển số xe nào trong khung hình."))
                        try:
                            os.remove("kq.jpg")
                            os.remove("test.jpg")
                            arduino.write("RETURN \n".encode())
                        except:
                            serial_queue.put(('log', "Hết chỗ để xe."))
                elif signal == "RUN_SKIP":
                    serial_queue.put(('log', signal))
                    RFID = arduino.readline().decode('utf-8').strip()
                    list_check = []
                    for check_all in list_license_plate:
                        if RFID in check_all:
                            list_check.append(check_all)
                            serial_queue.put(('log', "Xóa xe"))
                            serial_queue.put(('log', check_all))
                            list_license_plate.remove(check_all)
                            os.remove(f"license/{check_all}.jpg")
                            check_all_license = check_all[9:]
                            for check_all in list_license:
                                if check_all_license in check_all:
                                    list_check.append(check_all)
                                    list_license.remove(check_all)
                    if not list_check:
                        arduino.write("DENIED \n".encode())
                    else:
                        arduino.write("ACCEPT \n".encode())
        except Exception as e:
            serial_queue.put(('log', f"Lỗi trong serial_reader: {e}"))
            time.sleep(1)

# Hàm xử lý hàng đợi
def process_queue(root, log_text, image_label):
    try:
        while True:
            action, data = serial_queue.get_nowait()
            if action == 'log':
                log_message(data, root, log_text)
    except queue.Empty:
        pass
    root.after(100, process_queue, root, log_text, image_label)

# Thiết lập GUI
root = tk.Tk()
root.title("License Plate Recognition System")
root.geometry("900x600")

# Frame cho log
log_frame = tk.Frame(root)
log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)
log_label = tk.Label(log_frame, text="Log Output")
log_label.pack()
log_text = scrolledtext.ScrolledText(log_frame, height=30, width=50, state='disabled')
log_text.pack(fill=tk.BOTH, expand=True)

# Frame cho hình ảnh
image_frame = tk.Frame(root)
image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=10)
image_label = tk.Label(image_frame, text="Result Image (kq.jpg)")
image_label.pack()
image_display = tk.Label(image_frame)
image_display.pack()

# Thiết lập kết nối serial với Arduino
arduino = serial.Serial('COM4', 9600)

# Đường dẫn đến thư mục cần xóa tệp
directory = "license"
shutil.rmtree(directory, ignore_errors=True)
os.mkdir(directory)

log_message("Camera đã khởi chạy!!!", root, log_text)

# URL của API
url = "https://jay-humorous-koi.ngrok-free.app/vehicle/handle"

# Lắng nghe sự thay đổi Firebase
my_stream_checking = db.child("status/checking").stream(lambda msg: stream_handler_checking(msg, root, log_text, arduino))
my_stream_door1 = db.child("status/door1/isOpen").stream(lambda msg: stream_handler_door1(msg, root, log_text, arduino))
my_stream_door2 = db.child("status/door2/isOpen").stream(lambda msg: stream_handler_door2(msg, root, log_text, arduino))

# Khởi động luồng serial
serial_thread = threading.Thread(target=serial_reader, args=(root, log_text, image_display, arduino), daemon=True)
serial_thread.start()

# Bắt đầu xử lý hàng đợi
process_queue(root, log_text, image_display)

# Chạy GUI
root.mainloop()

# Đóng tài nguyên khi thoát
cap.release()
arduino.close()
my_stream_checking.close()
my_stream_door1.close()
my_stream_door2.close()