String data1;
void setup() {
  Serial.begin(9600);
  Serial1.begin(9600);
}
void loop() {
  data1="";
//  Serial.println("HERE");
  if (Serial.available() > 0) {
    data1 = Serial.readStringUntil('\n');
    Serial.print("You sent me: ");
    Serial1.print(data1+'\n');
    Serial.println(data1);
//    delay(1000);
  }
//  delay(100);
//    Serial.print("You sent me outside: ");
//    Serial.println(data1);
//   Serial1.print(data1+'\n');
}
