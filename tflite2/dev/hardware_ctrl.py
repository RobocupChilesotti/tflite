import time
import logging
import serial


# Multithread what's needed


def initialize_ser_com():
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(3)
    ser.reset_input_buffer()
    logging.info("Serial communication started")


# NO-MULTITHREADING!
def turn_x_deg(turn_deg, direction, speed):
    # Robot turns 'turn_deg' degrees on the center axis in the 'direction' direction
    # 'direction' can either be r (right) or l (left)
    done = False

    # Return flag 'done' when action completed

    return done


def set_turn(direction, speed):
    # Starts to turn on center axis in the given direction at the given speed
    # 'direction' can either be r (right) or l (left)

    return False


def stop():
    # Stop motors

    return False
