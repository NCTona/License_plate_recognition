from ultralytics import YOLO
import easyocr
import cv2

# Load a pretrained YOLO model (recommended for training)
model = YOLO("E:/Py/pythonProject/runs/detect/train11/weights/best.pt")

ocr_reader = easyocr.Reader(['en'], gpu=True)

# Mở webcam
cap = cv2.VideoCapture(0)  # '0' là chỉ số của webcam, có thể thay đổi nếu bạn dùng nhiều camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán vị trí biển số xe bằng YOLO
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Cắt vùng biển số từ khung hình
            plate_image = frame[int(y1):int(y2), int(x1):int(x2)]

            # Nhận diện ký tự trên biển số bằng EasyOCR
            ocr_results = ocr_reader.readtext(plate_image)
            for (bbox, text, prob) in ocr_results:
                # Hiển thị kết quả nhận diện
                print(f"Detected License Plate: {text} with confidence {prob:.2f}")


                # Hàm lọc kết quả, thay thế các ký tự không phải chữ in hoa hoặc số bằng khoảng trắng
                def filter_text(text):
                    return ''.join([char if char.isalnum() else ' ' for char in text.upper()])


                # Khoảng cách giữa các dòng
                line_height = 20

                # Vẽ bounding box và text lên khung hình
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Tính toán vị trí bắt đầu cho dòng đầu tiên
                start_y = int(y1) - 10 - (len(ocr_results) - 1) * line_height

                # Hiển thị từng dòng text theo thứ tự từ trên xuống dưới
                for i, (bbox, text, prob) in enumerate(ocr_results):
                    filtered_text = filter_text(text)  # Lọc kết quả OCR
                    y_position = start_y + i * line_height
                    cv2.putText(frame, filtered_text, (int(x1), y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)

    # Hiển thị khung hình
    cv2.imshow('License Plate Detection', frame)

    # Dừng lại khi nhấn phím 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
