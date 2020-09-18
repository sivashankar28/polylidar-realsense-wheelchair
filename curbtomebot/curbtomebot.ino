void setup() {
  Serial.begin(9600);
  Serial1.begin(9600);
}
void loop() {
//  Serial.println("HERE");
  if (Serial.available() > 0) {
    String data1 = Serial.readStringUntil('\n');
    Serial.print("You sent me: ");
    Serial1.print(data1);
    Serial.println(data1);
  }
}
