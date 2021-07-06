import socket
import struct
s=socket.socket()
host="169.254.41.103"       #This is your Server IP!
port=2345
s.connect((host,port))
data = struct.pack('!d', 3.1425)
s.send(data)
rece=s.recv(1024)
print("Received",rece)
s.close()