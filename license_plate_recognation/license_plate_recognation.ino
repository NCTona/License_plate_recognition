//Khai báo thư viện

#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
#include <LiquidCrystal.h>

//Khai báo các chân kết nối của Arduino với các linh kiện

#define SS_PIN 10
#define RST_PIN 9
#define SENSOR_IN A2
#define SENSOR_OUT A3
#define BUTTON A4
#define SERVO_PIN 2
#define BUZZER_PIN A0
#define SERVO_PIN2 A1
#define LCD_RS 3
#define LCD_ENABLE 4
#define LCD_D4 5
#define LCD_D5 6
#define LCD_D6 7
#define LCD_D7 8

//Khởi tạo các đối tượng (instances) cho các module

MFRC522 mfrc522(SS_PIN, RST_PIN);  // Tạo đối tượng cho đọc thẻ RFID
Servo servo;
Servo servo2;                                                           // Tạo đối tượng cho Servo
LiquidCrystal lcd(LCD_RS, LCD_ENABLE, LCD_D4, LCD_D5, LCD_D6, LCD_D7);  // Tạo đối tượng cho màn hình LCD

int buttonState = 0;
int availableSlots = 5;  // Số lượng chỗ đỗ trống ban đầu
String parkedRFIDs[5];   // Mảng để lưu trữ thông tin về các thẻ RFID đã đỗ

// Thêm biến để theo dõi thời gian còi kêu

unsigned long buzzerStartTime = 0;
const unsigned long buzzerDuration = 5000;  // Thời gian kêu còi: 5 giây

void setup() {
  Serial.begin(9600);
  SPI.begin();  // Khởi tạo bus SPI
  delay(100);
  mfrc522.PCD_Init();  // Khởi tạo đọc thẻ RFID

  servo2.attach(SERVO_PIN2);  // Khởi tạo servo2
  servo2.write(90);           // Đặt servo2 ban đầu ở vị trí 0 độ

  servo.attach(SERVO_PIN);
  servo.write(90);  // Đóng Servo ban đầu

  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(SENSOR_IN, INPUT_PULLUP);
  pinMode(SENSOR_OUT, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);


  lcd.begin(16, 2);  // Khởi tạo màn hình LCD 16x2
  lcd.print("Smart Parking");
  lcd.setCursor(0, 1);
  lcd.print("Available: " + String(availableSlots));
}

void loop() {

  buttonState = digitalRead(BUTTON);
  int sensorOut = digitalRead(SENSOR_OUT);
  int sensorIn = digitalRead(SENSOR_IN);

  if (buttonState == 1) {
    buzzBuzzer(100);
  }

  while (buttonState == 1) {

    lcd.clear();
    lcd.print("CheckOutManually");
    lcd.setCursor(0, 1);
    lcd.print("Available: " + String(availableSlots));

    if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
      String rfid = getRFID();  // Đọc mã RFID từ thẻ

      if (rfid != "") {
        buzzBuzzer(100);
        lcd.clear();
        lcd.print("RFID: " + rfid);

        Serial.println("RUN_SKIP");

        Serial.println(rfid);


        bool isParkedCar = false;
        for (int i = 0; i < 3; i++) {
          // Xử lý thẻ đã đọc
          if (rfid == parkedRFIDs[i]) {
            isParkedCar = true;
            break;
          }
        }

        if (isParkedCar) {
          String data = "";

          while (data == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            data = Serial.readStringUntil('s');
          }

          if (data == "DENIED ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);

            delay(1000);
            lcd.clear();
            lcd.print("Check out manually");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          } else if (data == "ACCEPT ") {
            delay(1000);
            servo2.write(0);
            removeParkedRFID(rfid);
            lcd.setCursor(1, 1);
            lcd.print("See You Again!");
            buzzBuzzer(1000);
            delay(1000);
            lcd.clear();
            lcd.print("Check out manually");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(++availableSlots));

            sensorOut = digitalRead(SENSOR_OUT);
            while (sensorOut == 1) {
              sensorOut = digitalRead(SENSOR_OUT);
              delay(50);
            }

            int sensorOutCheck = 0;
            while (sensorOut + sensorOutCheck == 0 || sensorOut + sensorOutCheck == 1) {
              sensorOut = digitalRead(SENSOR_OUT);
              delay(500);
              sensorOutCheck = digitalRead(SENSOR_OUT);
              delay(50);
            }

            delay(2000);
            servo2.write(90);  // Close the servo
          }
        } else {
          String data = "";

          while (data == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            data = Serial.readStringUntil('s');
          }

          if (data == "DENIED ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            delay(2000);
            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);

            lcd.clear();
            lcd.print("Check out manually");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          }
        }
      }
    }
    buttonState = digitalRead(BUTTON);
  }
  servo2.write(90);

  if (buttonState == 0) {
    buzzBuzzer(100);
  }

  while (buttonState == 0) {

    lcd.clear();
    lcd.print("Smart Parking");
    lcd.setCursor(0, 1);
    lcd.print("Available: " + String(availableSlots));

    // Kiểm tra nếu có thẻ RFID mới được đưa vào
    if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
      String rfid = getRFID();  // Đọc mã RFID từ thẻ

      if (rfid != "") {
        buzzBuzzer(100);
        lcd.clear();
        lcd.print("RFID: " + rfid);

        Serial.println("RUN_ADD");

        bool isParkedCar = false;
        for (int i = 0; i < 3; i++) {
          // Xử lý thẻ đã đọc
          if (rfid == parkedRFIDs[i]) {
            isParkedCar = true;
            break;
          }
        }

        if (isParkedCar) {
          Serial.println("REMOVE_CAR");
          String data = "";

          while (data == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            data = Serial.readStringUntil('s');
          }

          if (data == "RETURN ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);

            delay(1000);
            lcd.clear();
            lcd.print("Smart Parking");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          }

          Serial.println(data);

          Serial.println("REMOVE_CAR");
          Serial.println(rfid);

          String check = "";

          while (check == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            check = Serial.readStringUntil('s');
          }

          if (check == "DENIED ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            for (int i = 0; i <= 2; i++) {
              buzzBuzzer(100);
              delay(50);
              buzzBuzzer(100);
              delay(50);
              buzzBuzzer(100);
              delay(200);
            }

            delay(1000);
            lcd.clear();
            lcd.print("Smart Parking");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          }

          servo2.write(0);  // Open the servo
          lcd.setCursor(1, 1);
          lcd.print("See You Again!");
          buzzBuzzer(1000);
          delay(1000);
          lcd.clear();
          lcd.print("Available: " + String(++availableSlots));

          sensorOut = digitalRead(SENSOR_OUT);
          while (sensorOut == 1) {
            sensorOut = digitalRead(SENSOR_OUT);
            delay(50);
          }

          int sensorOutCheck = 0;
          while (sensorOut + sensorOutCheck == 0 || sensorOut + sensorOutCheck == 1) {
            sensorOut = digitalRead(SENSOR_OUT);
            delay(500);
            sensorOutCheck = digitalRead(SENSOR_OUT);
            delay(50);
          }

          delay(2000);
          servo2.write(90);  // Close the servo

          removeParkedRFID(rfid);
        } else if (availableSlots > 0) {
          Serial.println("ADD_CAR");
          String data = "";

          while (data == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            data = Serial.readStringUntil('s');
          }

          if (data == "RETURN ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);

            delay(1000);
            lcd.clear();
            lcd.print("Smart Parking");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          }

          Serial.println(data);
          Serial.println("ADD_CAR");
          Serial.println(rfid);

          String signal = "";

          while (signal == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            signal = Serial.readStringUntil('s');
          }

          if (signal == "UNAVAILABLE ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            for (int i = 0; i <= 2; i++) {
              buzzBuzzer(100);
              delay(50);
              buzzBuzzer(100);
              delay(50);
              buzzBuzzer(100);
              delay(200);
            }

            delay(1000);
            lcd.clear();
            lcd.print("Smart Parking");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          }

          servo.write(0);  // Open the servo
          lcd.setCursor(4, 1);
          lcd.print("Welcome!");
          buzzBuzzer(1000);
          delay(1000);
          lcd.clear();
          lcd.print("Available: " + String(--availableSlots));

          sensorIn = digitalRead(SENSOR_IN);
          while (sensorIn == 1) {
            sensorIn = digitalRead(SENSOR_IN);
            delay(50);
          }

          int sensorInCheck = 0;
          while (sensorIn + sensorInCheck == 0 || sensorIn + sensorInCheck == 1) {
            sensorIn = digitalRead(SENSOR_IN);
            delay(500);
            sensorInCheck = digitalRead(SENSOR_IN);
            delay(50);
          }

          delay(2000);
          servo.write(90);

          addParkedRFID(rfid);
        } else {
          Serial.println("FULL_CAR");
          String data = "";

          while (data == "") {
            // Hiển thị dữ liệu nhận được lên Serial Monitor
            data = Serial.readStringUntil('s');
          }

          if (data == "RETURN ") {
            lcd.clear();
            lcd.print("UNAVAILABLE");

            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);
            delay(100);
            buzzBuzzer(500);

            delay(1000);
            lcd.clear();
            lcd.print("Smart Parking");
            lcd.setCursor(0, 1);
            lcd.print("Available: " + String(availableSlots));

            return;
          }
        }

        lcd.clear();
        lcd.print("Smart Parking");
        lcd.setCursor(0, 1);
        lcd.print("Available: " + String(availableSlots));
      }
    }
    // Kiểm tra và điều khiển còi kêu

    if (isBuzzerOn()) {
      if (millis() - buzzerStartTime >= buzzerDuration) {
        stopBuzzer();
      }
    }
    mfrc522.PICC_HaltA();  // Dừng truyền thẻ RFID
    buttonState = digitalRead(BUTTON);
  }
}

String getRFID()  // Hàm dùng để đọc mã RFID từ thẻ
{
  String rfid = "";  // Xử lý và định dạng mã RFID
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    rfid.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? "0" : ""));
    rfid.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  mfrc522.PICC_HaltA();
  return rfid;
}

void addParkedRFID(String rfid)  // Hàm dùng để thêm thông tin thẻ đã đỗ vào mảng
{
  for (int i = 0; i < 3; i++)  // Thêm thông tin thẻ vào mảng parkedRFIDs
  {
    if (parkedRFIDs[i] == "") {
      parkedRFIDs[i] = rfid;
      break;
    }
  }
}

void removeParkedRFID(String rfid)  // Hàm dùng để xóa thông tin thẻ đã rời khỏi mảng
{
  for (int i = 0; i < 3; i++)  // Xóa thông tin thẻ ra khỏi mảng parkedRFIDs
  {
    if (parkedRFIDs[i] == rfid) {
      parkedRFIDs[i] = "";
      break;
    }
  }
}

void buzzBuzzer(unsigned int duration)  // Hàm dùng để kêu còi với thời gian cho trước
{
  digitalWrite(BUZZER_PIN, HIGH);  // Kêu còi với thời gian đã định sẵn
  delay(duration);
  digitalWrite(BUZZER_PIN, LOW);
}

void startBuzzer()  // Hàm dùng để phát tín hiệu cho còi
{
  digitalWrite(BUZZER_PIN, HIGH);  // Còi bắt đầu kêu để báo hiệu
  buzzerStartTime = millis();      // Lưu thời gian còi bắt đầu kêu
}

void stopBuzzer()  // Hàm dùng để tắt còi
{
  digitalWrite(BUZZER_PIN, LOW);  // Dừng phát tín hiệu cho còi
  buzzerStartTime = 0;            // Đặt thời gian bắt đầu kêu còi về 0
}

bool isBuzzerOn()  // Hàm kiểm tra trạng thái của còi có đang kêu hay không
{
  return buzzerStartTime > 0;
}