import serial
import time

ser= serial.Serial('/dev/ttyACM0', baudrate= 9600)
# time.sleep(3)

# def getValues() :
#      ser.write(b'g')
#      arduinoData= ser.readline().decode('ascii')
#      return arduinoData


while (1):
     # userInput = input ('Get Data Point?')
     # if userInput == 'y':
     ser.write(b'Hello')
     #    print(getValues())