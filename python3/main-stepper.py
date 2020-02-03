#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':

	import time
	from tb6600 import stepper
	
	
	timea = time.time()
	stepper.rotate_motor(18000)
	print(time.time() - timea)
