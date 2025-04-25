#include <Arduino_FreeRTOS.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
#include <LiquidCrystal.h>
#include <semphr.h>

// Pin definitions
#define SS_PIN 53
#define RST_PIN 5
#define SENSOR_IN 4
#define SENSOR_OUT 3
#define BUTTON 2
#define SERVO_PIN 49
#define BUZZER_PIN 6
#define SERVO_PIN2 48
#define LCD_RS 12
#define LCD_EN 11
#define LCD_D4 10
#define LCD_D5 9
#define LCD_D6 8
#define LCD_D7 7

// Task handles
TaskHandle_t MainTaskHandle;
TaskHandle_t ButtonTaskHandle;
TaskHandle_t CheckTaskHandle;
TaskHandle_t RunCheckHandle;
TaskHandle_t RunSkipHandle;
TaskHandle_t SignalTaskHandle;

// Queue handle for checking
QueueHandle_t SignalQueue;

// Semaphore handle
SemaphoreHandle_t HandleTaskSemaphore;
SemaphoreHandle_t RunSkipTaskSemaphore;
SemaphoreHandle_t RunCheckTaskSemaphore;


// Hardware instance
MFRC522 mfrc522(SS_PIN, RST_PIN);
Servo servo;
Servo servo2;
LiquidCrystal lcd(LCD_RS, LCD_EN, LCD_D4, LCD_D5, LCD_D6, LCD_D7);

// Global variables
byte sensorOut = digitalRead(SENSOR_OUT);  // int -> byte
byte sensorIn = digitalRead(SENSOR_IN);    // int -> byte
volatile byte changeSignal = 0;            // int -> byte

byte buttonState = digitalRead(BUTTON);  // int -> byte (0-255 range is sufficient)

String checking = "";

byte availableSlots = 20;  // int -> byte (max 255 slots is enough)
String parkedRFIDs[19];

unsigned int buzzerStartTime = 0;  // long -> int (max 65,535ms is sufficient)
const unsigned int buzzerDuration = 5000;

const byte unavailableSentSignal = 0;
const byte warningSentSignal = 1;
const byte welcomeSentSignal = 2;
const byte goodbyeSentSignal = 3;

void smartParkingLCD() {
  lcd.print(F("Smart Parking"));
  lcd.setCursor(0, 1);
  lcd.print(F("Available: "));
  lcd.print(String(availableSlots));
}

void checkingManuallyLCD() {
  lcd.print(F("CheckOutManually"));
  lcd.setCursor(0, 1);
  lcd.print(F("Available: "));
  lcd.print(String(availableSlots));
}

void unavailableSignal() {
  lcd.clear();
  lcd.print(F("UNAVAILABLE"));
  buzzBuzzer(500);
  vTaskDelay(100 / portTICK_PERIOD_MS);
  buzzBuzzer(500);
  vTaskDelay(100 / portTICK_PERIOD_MS);
  buzzBuzzer(500);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
  lcd.clear();
}

void warningSignal() {
  lcd.clear();
  lcd.print(F("WARNING"));
  for (byte i = 0; i <= 2; i++) {  // int -> byte
    buzzBuzzer(100);
    vTaskDelay(50 / portTICK_PERIOD_MS);
    buzzBuzzer(100);
    vTaskDelay(50 / portTICK_PERIOD_MS);
    buzzBuzzer(100);
    vTaskDelay(200 / portTICK_PERIOD_MS);
  }
  vTaskDelay(1500 / portTICK_PERIOD_MS);
  lcd.clear();
}

void welcomeSignal() {
  servo.write(0);
  lcd.setCursor(4, 1);
  lcd.print(F("Welcome!"));
  buzzBuzzer(1000);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
  lcd.clear();
  lcd.print(F("Available: "));
  lcd.print(String(--availableSlots));

  sensorIn = digitalRead(SENSOR_IN);
  while (sensorIn == 1) {
    sensorIn = digitalRead(SENSOR_IN);
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }

  byte sensorInCheck = 0;  // int -> byte
  while (sensorIn + sensorInCheck == 0 || sensorIn + sensorInCheck == 1) {
    sensorIn = digitalRead(SENSOR_IN);
    vTaskDelay(500 / portTICK_PERIOD_MS);
    sensorInCheck = digitalRead(SENSOR_IN);
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }

  vTaskDelay(1500 / portTICK_PERIOD_MS);
  servo.write(90);
}

void goodbyeSignal() {
  lcd.setCursor(1, 1);
  lcd.print(F("See You Again!"));
  buzzBuzzer(1000);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
  lcd.clear();
  lcd.print(F("Available: "));
  lcd.print(String(++availableSlots));

  sensorOut = digitalRead(SENSOR_OUT);
  while (sensorOut == 1) {
    sensorOut = digitalRead(SENSOR_OUT);
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }

  byte sensorOutCheck = 0;  // int -> byte
  while (sensorOut + sensorOutCheck == 0 || sensorOut + sensorOutCheck == 1) {
    sensorOut = digitalRead(SENSOR_OUT);
    vTaskDelay(500 / portTICK_PERIOD_MS);
    sensorOutCheck = digitalRead(SENSOR_OUT);
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }

  vTaskDelay(1000 / portTICK_PERIOD_MS);
  servo2.write(90);
}

void checkDoor(String* checkingPtr) {
  String data = "";
  data = Serial.readStringUntil('\n');
  if (data != "") {
    if (data == "true " || data == "false ") {
      *checkingPtr = "";
      Serial.println(F("LOOP"));
    } else if (data == "door1_True ") {
      servo.write(0);
    } else if (data == "door2_True ") {
      servo2.write(0);
    } else if (data == "door1_False ") {
      servo.write(90);
    } else if (data == "door2_False ") {
      servo2.write(90);
    }
  }
}

void checkStatus(String* checkingPtr, byte* changeSignalPtr) {  // int* -> byte*
  if (*changeSignalPtr == 1) {
    *checkingPtr = "";
    *changeSignalPtr = 0;
    Serial.println(F("CHANGE"));
  }
  vTaskDelay(1 / portTICK_PERIOD_MS);
}

void SignalTask(void* pvParameters) {
  (void)pvParameters;

  while (1) {
    byte receivedSignal;
    if (xQueueReceive(SignalQueue, &receivedSignal, portMAX_DELAY) == pdPASS) {
      switch (receivedSignal) {
        case 0:
          vTaskSuspend(RunCheckHandle);
          vTaskSuspend(RunSkipHandle);
          unavailableSignal();
          vTaskResume(RunSkipHandle);
          vTaskResume(RunCheckHandle);
          break;

        case 1:
          vTaskSuspend(RunCheckHandle);
          vTaskSuspend(RunSkipHandle);
          warningSignal();
          vTaskResume(RunSkipHandle);
          vTaskResume(RunCheckHandle);
          break;

        case 2:
          vTaskSuspend(RunCheckHandle);
          vTaskSuspend(RunSkipHandle);
          welcomeSignal();
          vTaskResume(RunSkipHandle);
          vTaskResume(RunCheckHandle);
          break;

        case 3:
          vTaskSuspend(RunCheckHandle);
          vTaskSuspend(RunSkipHandle);
          goodbyeSignal();
          vTaskResume(RunSkipHandle);
          vTaskResume(RunCheckHandle);
          break;
      }
    }
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }
}

void RunSkip(void* pvParameters) {
  (void)pvParameters;

  while (1) {

    if (xSemaphoreTake(RunSkipTaskSemaphore, portMAX_DELAY) == pdTRUE) {

      while (checking == "false ") {
        lcd.clear();
        checkingManuallyLCD();

        if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
          if (xSemaphoreTake(HandleTaskSemaphore, portMAX_DELAY) == pdTRUE) {
            checking = "";
            String rfid = getRFID();

            if (rfid != "") {
              buzzBuzzer(100);
              lcd.clear();
              lcd.print(F("RFID: "));
              lcd.print(rfid);

              Serial.println(F("RUN_SKIP"));
              Serial.println(rfid);

              bool isParkedCar = false;
              for (byte i = 0; i < 20; i++) {  // int -> byte, adjusted to array size
                if (rfid == parkedRFIDs[i]) {
                  isParkedCar = true;
                  break;
                }
              }

              if (isParkedCar) {
                String data = "";
                while (data == "") {
                  data = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (data == "DENIED ") {
                  vTaskDelay(1000 / portTICK_PERIOD_MS);

                  xQueueSend(SignalQueue, &unavailableSentSignal, portMAX_DELAY);
                  vTaskDelay(50 / portTICK_PERIOD_MS);

                  checkingManuallyLCD();
                  xSemaphoreGive(HandleTaskSemaphore);

                  continue;
                } else if (data == "ACCEPT ") {
                  vTaskDelay(1000 / portTICK_PERIOD_MS);
                  servo2.write(0);

                  xQueueSend(SignalQueue, &goodbyeSentSignal, portMAX_DELAY);
                  vTaskDelay(50 / portTICK_PERIOD_MS);

                  removeParkedRFID(rfid);
                  xSemaphoreGive(HandleTaskSemaphore);
                }
              } else {
                String data = "";
                while (data == "") {
                  data = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (data == "DENIED ") {
                  vTaskDelay(1000 / portTICK_PERIOD_MS);

                  xQueueSend(SignalQueue, &unavailableSentSignal, portMAX_DELAY);
                  vTaskDelay(50 / portTICK_PERIOD_MS);


                  checkingManuallyLCD();
                  xSemaphoreGive(HandleTaskSemaphore);

                  continue;
                }
              }
            }
            xSemaphoreGive(HandleTaskSemaphore);
          }
        }
      }
      vTaskResume(MainTaskHandle);
    }
  }
}

void RunCheck(void* pvParameters) {
  (void)pvParameters;

  while (1) {

    if (xSemaphoreTake(RunCheckTaskSemaphore, portMAX_DELAY) == pdTRUE) {

      while (checking == "true ") {
        lcd.clear();
        smartParkingLCD();

        if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
          if (xSemaphoreTake(HandleTaskSemaphore, portMAX_DELAY) == pdTRUE) {
            checking = "";
            String rfid = getRFID();

            if (rfid != "") {
              buzzBuzzer(100);
              lcd.clear();
              lcd.print(F("RFID: "));
              lcd.print(rfid);

              Serial.println(F("RUN_CHECK"));

              bool isParkedCar = false;
              for (byte i = 0; i < 20; i++) {  // int -> byte, adjusted to array size
                if (rfid == parkedRFIDs[i]) {
                  isParkedCar = true;
                  break;
                }
              }

              if (isParkedCar) {
                Serial.println(F("REMOVE_CAR"));
                String data = "";
                while (data == "") {
                  data = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (data == "RETURN ") {

                  xQueueSend(SignalQueue, &unavailableSentSignal, portMAX_DELAY);

                  smartParkingLCD();
                  xSemaphoreGive(HandleTaskSemaphore);

                  continue;
                }

                Serial.println(data);
                Serial.println(F("REMOVE_CAR"));
                Serial.println(rfid);

                String check = "";
                while (check == "") {
                  check = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (check == "DENIED ") {

                  xQueueSend(SignalQueue, &warningSentSignal, portMAX_DELAY);

                  smartParkingLCD();
                  xSemaphoreGive(HandleTaskSemaphore);

                  continue;
                }

                servo2.write(0);

                xQueueSend(SignalQueue, &goodbyeSentSignal, portMAX_DELAY);
                vTaskDelay(50 / portTICK_PERIOD_MS);

                removeParkedRFID(rfid);
                xSemaphoreGive(HandleTaskSemaphore);

              } else if (availableSlots > 0) {
                Serial.println(F("ADD_CAR"));
                String data = "";
                while (data == "") {
                  data = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (data == "RETURN ") {

                  xQueueSend(SignalQueue, &unavailableSentSignal, portMAX_DELAY);

                  smartParkingLCD();
                  xSemaphoreGive(HandleTaskSemaphore);
                  continue;
                }

                Serial.println(data);
                Serial.println(F("ADD_CAR"));
                Serial.println(rfid);

                String signal = "";
                while (signal == "") {
                  signal = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (signal == "UNAVAILABLE ") {

                  xQueueSend(SignalQueue, &warningSentSignal, portMAX_DELAY);

                  smartParkingLCD();
                  xSemaphoreGive(HandleTaskSemaphore);

                  continue;
                }

                xQueueSend(SignalQueue, &welcomeSentSignal, portMAX_DELAY);
                vTaskDelay(50 / portTICK_PERIOD_MS);

                addParkedRFID(rfid);
                xSemaphoreGive(HandleTaskSemaphore);

              } else {
                Serial.println(F("FULL_CAR"));
                String data = "";
                while (data == "") {
                  data = Serial.readStringUntil('\n');
                  vTaskDelay(1 / portTICK_PERIOD_MS);
                }

                if (data == "RETURN ") {

                  xQueueSend(SignalQueue, &unavailableSentSignal, portMAX_DELAY);

                  smartParkingLCD();
                  xSemaphoreGive(HandleTaskSemaphore);

                  continue;
                }
              }

              lcd.clear();
              smartParkingLCD();
            }
            xSemaphoreGive(HandleTaskSemaphore);
          }
        }

        if (isBuzzerOn()) {
          if (millis() - buzzerStartTime >= buzzerDuration) {
            stopBuzzer();
          }
        }
        mfrc522.PICC_HaltA();
      }
      vTaskResume(MainTaskHandle);
    }
  }
}

void MainTask(void* pvParameters) {
  (void)pvParameters;

  buttonState = digitalRead(BUTTON);

  while (1) {

    Serial.println(F("RUN"));

    while (checking == "") {
      checking = Serial.readStringUntil('\n');
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    if (checking == "false ") {
      buzzBuzzer(100);
      xSemaphoreGive(RunSkipTaskSemaphore);
      vTaskSuspend(NULL);
    } else if (checking == "true ") {
      buzzBuzzer(100);
      xSemaphoreGive(RunCheckTaskSemaphore);
      vTaskSuspend(NULL);
    }
  }
}

void CheckTask(void* pvParameters) {
  (void)pvParameters;

  while (1) {
    if (xSemaphoreTake(HandleTaskSemaphore, 0) == pdTRUE) {
      checkStatus(&checking, &changeSignal);
      checkDoor(&checking);
      xSemaphoreGive(HandleTaskSemaphore);
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

void ButtonTask(void* pvParameters) {
  (void)pvParameters;

  while (1) {
    if (xSemaphoreTake(HandleTaskSemaphore, 0) == pdTRUE) {
      byte currentButtonState = digitalRead(BUTTON);  // int -> byte
      if (buttonState != currentButtonState) {
        changeSignal = 1;
        buttonState = currentButtonState;
      }
      xSemaphoreGive(HandleTaskSemaphore);
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

void setup() {
  Serial.begin(9600);
  SPI.begin();
  mfrc522.PCD_Init();

  servo2.attach(SERVO_PIN2);
  servo2.write(90);
  servo.attach(SERVO_PIN);
  servo.write(90);

  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(SENSOR_IN, INPUT_PULLUP);
  pinMode(SENSOR_OUT, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);

  lcd.begin(16, 2);

  HandleTaskSemaphore = xSemaphoreCreateMutex();
  RunSkipTaskSemaphore = xSemaphoreCreateBinary();
  RunCheckTaskSemaphore = xSemaphoreCreateBinary();

  SignalQueue = xQueueCreate(5, sizeof(byte));

  xTaskCreate(MainTask, "MainTask", 256, NULL, 3, &MainTaskHandle);
  xTaskCreate(SignalTask, "SignalTask", 256, NULL, 2, &SignalTaskHandle);
  xTaskCreate(RunCheck, "RunCheck", 512, NULL, 1, &RunCheckHandle);
  xTaskCreate(RunSkip, "RunSkip", 512, NULL, 1, &RunSkipHandle);
  xTaskCreate(CheckTask, "CheckTask", 128, NULL, 1, &CheckTaskHandle);
  xTaskCreate(ButtonTask, "ButtonTask", 128, NULL, 1, &ButtonTaskHandle);

  vTaskStartScheduler();
}

void loop() {
  // Empty
}

// Helper functions
String getRFID() {
  String rfid = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    rfid.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? "0" : ""));
    rfid.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  mfrc522.PICC_HaltA();
  return rfid;
}

void addParkedRFID(String rfid) {
  for (byte i = 0; i < 20; i++) {  // int -> byte, adjusted to array size
    if (parkedRFIDs[i] == "") {
      parkedRFIDs[i] = rfid;
      break;
    }
  }
}

void removeParkedRFID(String rfid) {
  for (byte i = 0; i < 20; i++) {  // int -> byte, adjusted to array size
    if (parkedRFIDs[i] == rfid) {
      parkedRFIDs[i] = "";
      break;
    }
  }
}

void buzzBuzzer(unsigned int duration) {
  digitalWrite(BUZZER_PIN, HIGH);
  vTaskDelay(duration / portTICK_PERIOD_MS);
  digitalWrite(BUZZER_PIN, LOW);
}

void startBuzzer() {
  digitalWrite(BUZZER_PIN, HIGH);
  buzzerStartTime = millis();
}

void stopBuzzer() {
  digitalWrite(BUZZER_PIN, LOW);
  buzzerStartTime = 0;
}

bool isBuzzerOn() {
  return buzzerStartTime > 0;
}