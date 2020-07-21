import time
import serial

def foo():
    print("sent")
    ardu= serial.Serial('/dev/ttyACM0',9600, timeout=.1)
    time.sleep(1)
    ardu.write('s'.encode())
    time.sleep(1)
    #ardu.close()


foo()

