####
#
# Stepper Motor Program
#
####                                                                                                                                                                                                                                                                                                                                                                                       t pins

__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research Group'

import logging.config
import time
import RPi.GPIO as GPIO


logfile = 'januswm'
logger = logging.getLogger(logfile)

# TODO Check value with switches on controller
# Stepper Controller switches: S1 = OFF, S2 = OFF, S3 = OFF


def rotate_motor(
        nbr_rot: float,
):
    """
    Moves motor

    :param dirpin: int
    :param pulpin: int
    :param enapin: int
    :param nbr_rot: float
    """
    # GPIO Setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Direction and pulse pins
    enapin = 13
    dirpin = 19
    pulpin = 26

    # Set GPIO pin modes
    GPIO.setup(dirpin, GPIO.OUT)
    GPIO.setup(pulpin, GPIO.OUT)
    GPIO.setup(enapin, GPIO.OUT)

    # Less than 0, direction:
    # counter-clockwise
    #
    # Greater than 0, direction:
    # clockwise
    steps_rev = 800
    pulse_freq = 6400.0
    pulse_delay = 1.0 / pulse_freq

    cal_steps = 0

    GPIO.output(enapin, True)
    if nbr_rot < 0:
        GPIO.output(dirpin, False)
    elif nbr_rot > 0:
        GPIO.output(dirpin, True)

    nbr_steps = int(nbr_rot * float(steps_rev))
    print(nbr_steps)

    # Move axis
    for d in range(0, nbr_steps):
        GPIO.output(pulpin, True)
        time.sleep(pulse_delay)
        GPIO.output(pulpin, False)
        time.sleep(pulse_delay)

    GPIO.output(enapin, False)

    GPIO.cleanup()
