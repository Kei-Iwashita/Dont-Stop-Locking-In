#include <Servo.h>

#define SERVO_PIN  9
#define BUZZER_PIN 7
#define LED_PIN    13

const int SEG_PINS[8] = {A1, 2, 8, 11, 12, A2, 6, 10};  // A, B, C, D, E, F, G, Decimal Point
const int DIG_PINS[4] = {A0, 4, 3, 5};                   // Digit 1 (left) to Digit 4 (right)

const byte DIGITS[10] = {
    0b00111111,
    0b00000110,
    0b01011011,
    0b01001111,
    0b01100110,
    0b01101101,
    0b01111101,
    0b00000111,
    0b01111111,
    0b01101111
};

const byte SEG_BLANK = 0b00000000;
const byte SEG_DASH  = 0b01000000;
const byte SEG_COLON = 0b10000000;

Servo         myServo;
int           remainingSecs = -1;
unsigned long lastToggle    = 0;
unsigned long lastDigitTime = 0;
bool          beepState     = false;
int           currentDigit  = 0;
String        inputBuffer   = "";

byte digitBuffer[4] = {SEG_DASH, SEG_DASH, SEG_DASH, SEG_DASH};

void setSegments(byte mask) {
    for (int i = 0; i < 8; i++) {
        digitalWrite(SEG_PINS[i], (mask >> i) & 1);
    }
}

void refreshDisplay() {
    for (int i = 0; i < 4; i++) digitalWrite(DIG_PINS[i], HIGH);
    setSegments(digitBuffer[currentDigit]);
    if (currentDigit == 1) digitalWrite(SEG_PINS[7], HIGH);
    digitalWrite(DIG_PINS[currentDigit], LOW);
    currentDigit = (currentDigit + 1) % 4;
}

void runDisplayMs(unsigned long ms) {
    unsigned long start = millis();
    while (millis() - start < ms) {
        if (millis() - lastDigitTime >= 3) {
            refreshDisplay();
            lastDigitTime = millis();
        }
    }
}

void showNumber(int n) {
    digitBuffer[0] = DIGITS[(n / 1000) % 10];
    digitBuffer[1] = DIGITS[(n / 100)  % 10];
    digitBuffer[2] = DIGITS[(n / 10)   % 10];
    digitBuffer[3] = DIGITS[ n         % 10];
}

void showTime(int secs) {
    if (secs < 0) {
        for (int i = 0; i < 4; i++) digitBuffer[i] = SEG_DASH;
        return;
    }
    int mins = secs / 60;
    int s    = secs % 60;
    digitBuffer[0] = DIGITS[mins / 10];
    digitBuffer[1] = DIGITS[mins % 10];
    digitBuffer[2] = DIGITS[s    / 10];
    digitBuffer[3] = DIGITS[s    % 10];
}

int getInterval(int secs) {
    if (secs <= 0) return 50;
    if (secs >= 5) return 500;
    return 50 + (secs * (500 - 50) / 5);
}

void processCommand(String cmd) {
    cmd.trim();
    if (cmd == "1") {
        myServo.write(90);
    } else if (cmd == "0") {
        myServo.write(0);
    } else if (cmd == "X") {
        remainingSecs = -1;
        beepState     = false;
        digitalWrite(BUZZER_PIN, LOW);
        digitalWrite(LED_PIN,    LOW);
        showTime(-1);
    } else if (cmd.length() >= 2 && cmd[0] == 'T') {
        remainingSecs = cmd.substring(1).toInt();
        showTime(remainingSecs);
    }
}





void runTests() {
    Serial.println("TEST 1: LED");
    for (int i = 0; i < 5; i++) {
        digitalWrite(LED_PIN, HIGH); delay(300);
        digitalWrite(LED_PIN, LOW);  delay(300);
    }
    delay(500);

    Serial.println("TEST 2: Buzzer");
    for (int i = 0; i < 3; i++) {
        digitalWrite(BUZZER_PIN, HIGH); delay(200);
        digitalWrite(BUZZER_PIN, LOW);  delay(200);
    }
    delay(500);

    Serial.println("TEST 3: Servo");
    myServo.write(0);  delay(800);
    myServo.write(90); delay(800);
    myServo.write(0);  delay(800);
    delay(500);

    Serial.println("TEST 4: Display - all segments on");
    for (int i = 0; i < 4; i++) digitBuffer[i] = 0b11111111;
    runDisplayMs(1500);

    Serial.println("TEST 5: Display - 0000 to 9999");
    for (int d = 0; d <= 9999; d += 1111) {
        showNumber(d);
        runDisplayMs(600);
    }

    Serial.println("TEST 6: Display - countdown 9 to 0");
    for (int i = 9; i >= 0; i--) {
        showNumber(i);
        runDisplayMs(400);
    }

    for (int i = 0; i < 4; i++) digitBuffer[i] = SEG_BLANK;
    runDisplayMs(300);

    Serial.println("Well what happened?");
}

void setup() {
    Serial.begin(9600);

    myServo.attach(SERVO_PIN);
    myServo.write(0);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LED_PIN,    OUTPUT);
    for (int i = 0; i < 8; i++) pinMode(SEG_PINS[i], OUTPUT);
    for (int i = 0; i < 4; i++) {
        pinMode(DIG_PINS[i], OUTPUT);
        digitalWrite(DIG_PINS[i], HIGH);
    }

    showTime(-1);

    //runTests();  // hardware and wiring
}

void loop() {
    unsigned long now = millis();

    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n') {
            processCommand(inputBuffer);
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }

    if (now - lastDigitTime >= 3) {
        refreshDisplay();
        lastDigitTime = now;
    }

    if (remainingSecs >= 0 && remainingSecs <= 5) {
        int interval = getInterval(remainingSecs);
        if (now - lastToggle >= (unsigned long)interval) {
            beepState = !beepState;
            digitalWrite(BUZZER_PIN, beepState ? HIGH : LOW);
            digitalWrite(LED_PIN,    beepState ? HIGH : LOW);
            lastToggle = now;
        }
    } else {
        if (beepState) {
            beepState = false;
            digitalWrite(BUZZER_PIN, LOW);
            digitalWrite(LED_PIN,    LOW);
        }
    }
}