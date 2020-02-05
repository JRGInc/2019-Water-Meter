#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':
    import time
    from common import stepper

    # 500 rotations requires about 3 minutes
    timea = time.time()
    stepper.rotate_motor(500)
    print(time.time() - timea)
