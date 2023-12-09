import logging
 
import serial
import time




ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(3)
ser.reset_input_buffer()
logging.info("Serial communication started")
i = 170
a = 170
try:
    while True:
        string = "M:"+str(i)+":"+str(a)+":"+str(i)+":"+str(a)+"\n" 
        ser.write(string.encode('utf-8'))		#Motor:FRight:FLeft:BLeft:BRight
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        time.sleep(1)
        #i+=50
        #a = 510 - i
        if i>510:
          i = 0
except KeyboardInterrupt:
    ser.close()
    logging.info("Serial communication stopped")


 if __name__ == '__main__':
    