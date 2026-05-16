# License Plate Recognition System 🚗🔍

Hệ thống nhận diện biển số xe tự động ứng dụng Trí tuệ nhân tạo (AI) và tích hợp IoT.

## 🌟 Giới thiệu
Dự án sử dụng YOLOv8 để phát hiện vùng chứa biển số xe và EasyOCR để trích xuất ký tự. Hệ thống được tích hợp với Firebase để đồng bộ trạng thái theo thời gian thực và giao tiếp với module phần cứng Arduino qua cổng Serial để tự động điều khiển đóng/mở barie kết hợp xác thực qua thẻ RFID.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python, C++ (Arduino)
- **AI/Computer Vision:** OpenCV, YOLOv8 (Ultralytics), EasyOCR, Pillow, Numpy
- **IoT/Hardware:** Arduino (giao tiếp qua thư viện `pyserial`), sử dụng **FreeRTOS** để quản lý đa tiến trình (multi-tasking) như xử lý tín hiệu, nút bấm, và kiểm tra thẻ RFID đồng thời.
- **Cloud/Database:** Firebase Realtime Database (`pyrebase`, `firebase_admin`)
- **API:** Tích hợp `requests` để gửi dữ liệu hình ảnh và ID nhận diện tới Server/Backend xử lý.

## 📁 Cấu trúc dự án
- `image_detected.py` / `image_detected_app.py`: Logic thực thi chính. Chịu trách nhiệm mở webcam, lắng nghe thay đổi từ Firebase, gọi model nhận diện, tương tác Arduino và gửi dữ liệu đi.
- `training_model.py`: Script dùng để huấn luyện model YOLOv8 (chạy với `yolov8s.pt` và config `mydata.yaml`).
- `license_video_detected.py`: Script hỗ trợ nhận diện trực tiếp qua video.
- `license_plate_recognation/license_plate_recognation.ino`: File mã nguồn chính của Arduino (định dạng `.ino`) dùng để biên dịch và nạp trực tiếp vào vi điều khiển.
- `ControlArduino.txt`: Bản sao của mã nguồn Arduino dưới định dạng văn bản thuần túy, giúp người dùng dễ dàng xem và đọc trực tiếp mã mà không cần cài phần mềm chuyên dụng (Arduino IDE).
- `requirement.txt`: Danh sách các thư viện cần thiết.

## ⚙️ Hướng dẫn cài đặt

1. **Cài đặt thư viện:**
   Mở terminal và chạy lệnh sau để cài đặt các package phụ thuộc:
   ```bash
   pip install -r requirement.txt
   ```

2. **Cấu hình Firebase & API:**
   - Cập nhật thông tin `firebase_config` trong source code với Firebase Project của bạn.
   - Cập nhật biến `url` (API endpoint) phù hợp để nhận webhook/dữ liệu trả về.

3. **Cấu hình phần cứng (Arduino):**
   - Đảm bảo Arduino đã được cắm vào máy. Kiểm tra và sửa lại port `COM` trong code (mặc định đang là `COM4`).

4. **Khởi chạy ứng dụng:**
   ```bash
   python image_detected.py
   ```
