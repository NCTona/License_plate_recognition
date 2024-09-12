import cv2
import easyocr
import os
import serial
from PIL import Image
from ultralytics import YOLO

# # Đảm bảo thư mục lưu ảnh tồn tại
# output_dir = 'E:/Py/pythonProject'
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Không thể mở camera")
#     exit()
#
# # Chụp ảnh
# ret, frame = cap.read()
#
# if ret:
#     # Đặt tên file và lưu ảnh
#     image_path = os.path.join(output_dir, 'test.jpg')
#     cv2.imwrite(image_path, frame)
#     print(f"Đã lưu ảnh tại {image_path}")
# else:
#     print("Không chụp được ảnh")
#


# Thiết lập kết nối serial với Arduino
arduino = serial.Serial('COM8', 9600)  # COM3 là cổng của Arduino, có thể khác trên máy của bạn

while True:
    # Đọc dòng từ serial
    signal = arduino.readline().decode('utf-8').strip()

    if signal == "RUN_PYTHON":
        print(signal)
        # Load a pretrained YOLO model (recommended for training)
        model = YOLO("E:/Py/pythonProject/runs/detect/train11/weights/best.pt")

        # Train the model using the 'coco8.yaml' dataset for 3 epochs
        results = model("E:/Py/pythonProject/test.jpg")

        original_image = cv2.imread("E:/Py/pythonProject/test.jpg")

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
                cv2.imwrite("E:/Py/pythonProject/license_plate_cropped.jpg", cropped_image)

                # Mở hình ảnh
                im = Image.open("E:/Py/pythonProject/license_plate_cropped.jpg")

                # Hiển thị hình ảnh
                im.show()
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        def segment_image(image):
            # Chuyển ảnh biển số về màu xám
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            # Áp dụng GaussianBlur để giảm nhiễu
            # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            #
            # Áp dụng threshold để phân tách số và nền
            # ret, threshold_image = cv2.threshold(gray_image,0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Hiển thị ảnh đã xử lý (tuỳ chọn)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Sử dụng easyocr để nhận diện ký tự từ ảnh đã xử lý
            reader = easyocr.Reader(['en'], gpu=True)
            results = reader.readtext(image)

            # Tạo một từ điển chuyển đổi các ký tự thường thành ký tự mong muốn
            conversion_dict = {
                'l': '1',  # Chữ thường "l" có thể bị nhầm với số "1"
                'o': '0',  # Chữ thường "o" có thể bị nhầm với số "0"
                'i': '1',  # Chữ thường "i" có thể bị nhầm với số "1"
                'g': '9',  # Chữ thường "g" có thể bị nhầm với số "9"
                'a': '9',  # Chữ thường "a" có thể bị nhầm với số "9"
                'b': '6',  # Chữ thường "b" có thể bị nhầm với số "6"
                's': '5',  # Chữ thường "s" có thể bị nhầm với số "5"
                'q': '9',  # Chữ thường "q" có thể bị nhầm với số "9"
            }

            # Chuyển đổi văn bản theo từ điển
            def convert_text(text):
                # Thay thế các ký tự không phải chữ hoa hoặc số bằng khoảng trắng
                filtered_text = ''.join([char if char.isalnum() else ' ' for char in text])
                # Chuyển đổi các ký tự theo từ điển
                return ''.join([conversion_dict.get(char, char) for char in filtered_text])

            # Chuyển đổi và định dạng kết quả
            formatted_results = []
            for result in results:
                text = result[1]  # Lấy văn bản nhận diện được
                converted_text = convert_text(text)  # Chuyển đổi văn bản
                formatted_results.append(converted_text)

            # Ghép kết quả thành chuỗi theo định dạng mong muốn
            output = ' '.join(formatted_results)
            print(output)


        # Đọc hình ảnh chứa biển số xe
        image_path = "E:/Py/pythonProject/license_plate_cropped.jpg"
        image = cv2.imread(image_path)

        # Phân đoạn và nhận diện văn bản
        segment_image(image)

