# 2019 Water Meter Code Base

This code library executes several functions:

1. Train Inception v4 with TensorFlow on Ubuntu desktop computer with nVidia GPU
2. Test Inception v4 with TensorFlow on Ubuntu desktop computer with nVidia GPU
3. Image capture, prediction, and transmission on Raspberry PI with PiCamera v2
4. Control stepper motor controller via Raspberry PI with TB6600 to rotate water meter shaft
5. Image capture experimentation on Raspberry PI

Create and download code to the following directory:

```
/opt/Janus/WM/
```

After downloading the code, create the following additional directory structure prior to running any function:

```
/cache
/data
	/history
	/images
		/00--movies
		/01--original
		/02--scaled
		/03--screws
		/04--grotated
		/05--frotated
		/06--rectangled
		/07--windowed
		/08--inverted
		/09--contoured
		/10--digits
		/11--prediction
		/12--overlaid
		/13--train_sifted
			# Directories /d0 to /d5 
		/14--train_selected
			# Directories /d0 to /d5 each containing directories /0 to /29
		/15--train
		/16--valid
		/17--test_sifted
			# Directories /d0 to /d5 
		/18--test_selected
			# Directories /d0 to /d5 each containing directories /0 to /29
		/19--test
		/20--test_errors
			# Directories /0 to /29 each containing directories /0 to /29
	/latest
	/results
	/transmit
/logs
	/train
	/validation
	/test
/model
/weights
	/final
	/periodic
```

## Capture Images for TensorFlow Model Training

This requires two Raspberry Pis: one with TB6600 stepper motor driver and NEMA 17 stepper motor attached to meter shaft, the other with PiCamera v2 to image meter.  It is important to have smooth and consistent shaft rotation. Then RPI3B+ on the stepper motor rotates the motor much more smoothly than the RPI0W.  Each full rotation of the least significant digit requires approximately 200 main shaft rotations on the meter.

The method used to gather images for training is to capture video while the meter shaft is continuously rotating.  After a length of video is captured, transfer file to a desktop computer to extract still images.

1. Start continuous video capture on the Raspberry Pi from terminal:

```
pi@raspberrypi:~$ sudo /opt/Janus/WM/python3/main-video.py
```

2. On the stepper controller Raspberry Pi
   - Determine number of target rotations (20,000) is a good starting point (expect this to take hours)
   - Graphicaly open ```/opt/Janus/WM/python3/main-stepper.py``` for editing
   - Find and edit this python line with number of rotations:

```
stepper.rotate_motor(18000)
```

2. Cont'd
   - Close file
   - Open terminal and execute BASH code

```
pi@raspberrypi:~$ sudo /opt/Janus/WM/python3/main-stepper.py
```
 
3.  After stepper motor completes rotations, stop the video cpature by selecting terminal window and pressing ```CTRL-C```.
4.  Videos are segmented into files of 1-minute length, and located in the ```/opt/Janus/WM/data/images/00--movies/``` directory.
5.  Download code to desktop computer and build directory structure outlined above.
6.  Transfer these files to the identical directory on the desktop computer.  These files are large and numerous, best to use a USB stick, rather than SFTP.
7.  On desktop computer:
	  - Set the execution routine to extract JPG images from the video files and disable all other routines
	  - Graphicaly open ```/opt/Janus/WM/python3/config/core.py``` for editing
	  - Find (near bottom) and edit this python dictionary to appear as follows
```
self.batch_proc_en_dict = {
	'convert_h264': True,
	'build_train_digits': False,
	'sift_train_digits': False,
	'shift_train_digits': False,
	'split_dataset': False,
	'train_model': False,
	'build_test_images': False,
	'sift_test_digits': False,
	'save_test_digits': False,
	'test_model': False,
}
```

7. Cont'd:
   - Close file
   - Open terminal and execute BASH code:

```
username@hostname:~$ sudo /opt/Janus/WM/python3/main-train.py
```
7. Cont'd:
   - Resulting JPG images will be located ```/opt/Janus/WM/data/images/00--original/``` directory.
	
