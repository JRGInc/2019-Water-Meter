# 2019 Water Meter Code Base

This code library executes several functions:

1. Train Inception v4 image classification model with TensorFlow on Ubuntu desktop computer with nVidia GPU
2. Test Inception v4 image classification with TensorFlow on Ubuntu desktop computer with nVidia GPU
3. Image capture, classification, and transmission on Raspberry PI with PiCamera v2
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
			# (representing 6 digit positions in the digit window)
		/14--train_selected
			# Directories /d0 to /d5 each containing directories /0 to /29 
			# (representing 30 classes of each digit position)
		/15--train
		/16--valid
		/17--test_sifted
			# Directories /d0 to /d5 (representing 6 digit positions in the digit window)
			# (representing 6 digit positions in the digit window)
		/18--test_selected
			# Directories /d0 to /d5 each containing directories /0 to /29
			# (representing 30 classes of each digit position)
		/19--test
		/20--test_errors
			# Directories /0 to /29 each containing directories /0 to /29
			# (representing 30 classes each with 30 classes where images were misclassified)
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

## Capture Videos and Convert to Images for TensorFlow Image Classification Model Training

This requires two Raspberry Pis: one with TB6600 stepper motor driver and NEMA 17 stepper motor attached to meter shaft, the other with PiCamera v2 to image meter.  It is important to have smooth and consistent shaft rotation. Then RPI3B+ on the stepper motor rotates the motor much more smoothly than the RPI0W.  Each full rotation of the least significant digit requires approximately 200 main shaft rotations on the meter.

The method used to gather images for training is to capture video while the meter shaft is continuously rotating.  After a length of video is captured, transfer file to a desktop computer to extract still images.

1. Start continuous video capture on the Raspberry Pi from terminal:

```
pi@raspberrypi:~$ sudo /opt/Janus/WM/python3/main-video.py
```

2. On the stepper controller Raspberry Pi
   - Determine number of target rotations (20,000) is a good starting point (expect this to take hours)
   - Graphically open ```/opt/Janus/WM/python3/main-stepper.py``` for editing
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
	  - Graphically open ```/opt/Janus/WM/python3/config/core.py``` for editing
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
   - Resulting JPG images will be located ```/opt/Janus/WM/data/images/00--original/``` directory with file name convention ```orig_YYYY-MM-DD_HHMM_nnnnnnn_sgNNNN.jpg```, where `nnnnnnn` represents an incremental capture sequence, and ```sgNNNN``` represents the video segment number.
	
## Preprocess Converted Images for Training

A number of steps must be completed before images can be used to train the TensorFlow Inception v4 model:

1. Create cropped images of each digit in digit window.
2. Copy digits into directory according to position in digit window.
3. Select digits for use in model training (**See Note Below**).
4. Further crop digits on left and right edges to shift the center of the digit in the resulting image.
5. Random copy of training images to validatation images.  Both sets are used during model training.

For each of these steps (except step 3), the python dictionary in the file ```/opt/Janus/WM/python3/config/core.py``` must be edited, changing the appropriate line to ```True``` and leaving all others as ```False```:

```
self.batch_proc_en_dict = {
	'convert_h264': False,
	'build_train_digits': False,  # Step 1, change to True
	'sift_train_digits': False,   # Step 2, change to True
	'shift_train_digits': False,  # Step 4, change to True
	'split_dataset': False,       # Step 5, change to True
	'train_model': False,
	'build_test_images': False,
	'sift_test_digits': False,
	'save_test_digits': False,
	'test_model': False,
}
```
After each step in sequence, for steps 1, 2, 4, and 5, open terminal and execute BASH code: 

```
username@hostname:~$ sudo /opt/Janus/WM/python3/main-train.py
```

The output directories for these steps are located here:

```
# Step 1
/opt/Janus/WM/data/images/10--digits
# Step 2
/opt/Janus/WM/data/images/13--train_sifted/dN
# Step 4
opt/Janus/WM/data/images/15--train
# Step 5
opt/Janus/WM/data/images/16--valid
```

Intermediate steps executed by Python will produce outputs in ```/opt/Janus/data/images/02--scaled``` to ```/opt/Janus/data/images/09--contoured```, which can be analyzed for troubleshooting scaling, rotation, and cropping functions.

### Important Note for Step 3:

For step 3, the images in ```/opt/Janus/WM/data/images/13--train_sifted/dN``` must be visually inspected and copied to directory ```/opt/Janus/WM/data/images/14--train-selected/dN/C```.  Care must be taken to select and copy images into the proper class ```C``` directory.  Misclassifying an image at this stage will negatively impact accuracy of predictions at a later stage--with affects that are not easy to investigate.  The images can be selected and copied in bulk or individually.

### Image Preprocessing Settings

A number of settings can be changed that are located in ```/opt/Janus/WM/config/tensor.ini```.  These are loaded once at the beginning of execution for each step.  Once these settings are set, they should not be changed for the duration on the entire project: training, testing, and predictions.  **Changes made here can cause code execution or image preprocessing failures.**  The following provides a summary of these settings:

```
[TensorFlow]
full_width = 84           # Full width of each individual digit (width of digit window on training images divided by six)
tf_width = 65		  # Width of digit fed into TensorFlow prior to TensorFlow operations
shadow = 6                # Visual shadow at top of digit window caused by lighting
height = 93		  # Height of digit fed into TensorFlow prior to TensorFlow operations
shift_en = True		  # Enables left to right shifting of tf_width crop window inside full_width window
			  # Digit cylinders have left to right wobble during rotation
shift = 18 		  # How many times to shift the image
batch_size = 6            # TensorFlow batch size set to number of digits in digit window
img_tgt_width = 299       # Width required by Inception v4 model
img_tgt_height = 299	  # Height required by Inception v4 model
			  # TensorFlow upscales smaller images to meet requirements
nbr_classes = 30          # Number of classes into which images are sorted
nbr_channels = 1          # 1 = single-color gray-scale JPG, 3 = RGB JPG
patience = 3              # Number of epochs to wait before ending training early if criteria is met
epochs = 8		  # Max number of epochs to train model
format = h5		  # Model file format
```

## Train TensorFlow Inception v4 Model for Image Classification

A dedicated Ubuntu computer with nVidia GPU should be set aside for this step.  With 145,000 training images, training time has been observed to take 18 hours.  More images and larger image sizes will significantly increase training time.  

1. Code must be downloaded to the proper location and additional directories setup as noted above.
2. If using a dedicated computer, images located in the ```/opt/Janus/WM/data/images/15--train``` and ```/opt/Janus/WM/data/images/16--valid``` directories must be copied to the identical location on the dedicated computer.
3. The file ```/opt/Janus/WM/python3/config/core.py``` must be edited as noted here:

```
self.batch_proc_en_dict = {
	'convert_h264': False,
	'build_train_digits': False,
	'sift_train_digits': False,
	'shift_train_digits': False,
	'split_dataset': False,
	'train_model': True,
	'build_test_images': False,
	'sift_test_digits': False,
	'save_test_digits': False,
	'test_model': False,
}
```

4. Open terminal and execute BASH code: 

```
username@hostname:~$ sudo /opt/Janus/WM/python3/main-train.py
```

After each epoch of training, the validation image set is compared to present state of the model.  Since the proper classification of images in both sets are known, TensorFlow provides a dynamic status of the model after each epoch: loss, accuracy, validation loss, and validation accuracy.  In the early epochs loss should trend heavily downward and accuracy heavily upward.  As training progresses these values will plateau.  Weights are produced for each epoch and saved in ```/opt/Janus/WM/weights/periodic``` and a functional model is saved in ```/opt/Janus/WM/model```

At the end of training, final weights and model are saved in ```/opt/Janus/WM/weights/final``` and ```/opt/Janus/WM/model```, respectively.  An accuracy and loss vs epoch chart is produced and saved in ```/opt/Janus/WM/weights```.

### Important Note for Training

During the first epoch of training a data file of both the train and validation images are built in ```/opt/Janus/WM/cache/train``` and ```/opt/Janus/WM/cache/valid```, respectively.  If images are added, changed, or deleted in any fashion, these directories must be emptied prior to re-execution of training.

## Images for Model Testing

Prior collecting images for testing, the videos in the ```/opt/Janus/WM/data/images/00--movies``` and ```/opt/Janus/WM/data/images/01--originals``` directories which previously contained the unprocessed training images should be archived to prevent loss.

At this point a decision must be made to collect a new set of images, either in video format or JPG format.  The videos should be processed exactly as those used for training images (see **Capture Videos and Convert to Images for TensorFlow Image Classification Model Training** above).  An image set, if available, which will provide more useful results are images captured in real operations.  

Once JPG images are extracted or chosen, they should be placed in ```/opt/Janus/WM/data/images/01--original``` directory.  File name convention for the images should follow ```orig_YYYY-MM-DD_HHMM_nnnnnnn_sgNNNN.jpg```, where `nnnnnnn` represents an incremental capture sequence, and ```sgNNNN``` represents the video segment number. Use ```sg0001``` for individually captured still images.  A useful linux program to use for bulk renaming operations is ```gprename```.

## Preprocess Converted Images for Testing

In a manner similar to model training, a number of steps must be completed before images can be used to test the TensorFlow Inception v4 model:

1. Create cropped images of each digit in digit window.
2. Copy digits into directory according to position in digit window.
3. Select digits for use in model testing (**See Note Below**).
4. Copy selected digits to director for testing

For each step (except step 3), the python dictionary in the file ```/opt/Janus/WM/python3/config/core.py``` must be edited, changing the appropriate line to ```True``` and leaving all others as ```False```:

```
self.batch_proc_en_dict = {
	'convert_h264': False,
	'build_train_digits': False, 
	'sift_train_digits': False,  
	'shift_train_digits': False,  
	'split_dataset': False,       
	'train_model': False,
	'build_test_images': False,	# Step 1, change to True
	'sift_test_digits': False,	# Step 2, change to True
	'save_test_digits': False,	# Step 4, change to True
	'test_model': False,
}
```
After each step in sequence, for steps 1, 2, and 4, open terminal and execute BASH code: 

```
username@hostname:~$ sudo /opt/Janus/WM/python3/main-train.py
```

The output directories for these steps are located here:

```
# Step 1
/opt/Janus/WM/data/images/10--digits
# Step 2
/opt/Janus/WM/data/images/17--test_sifted/dN
# Step 4
opt/Janus/WM/data/images/19--test
```

Intermediate steps executed by Python will produce outputs in ```/opt/Janus/data/images/02--scaled``` to ```/opt/Janus/data/images/09--contoured```, which can be analyzed for troubleshooting scaling, rotation, and cropping functions.

### Important Note for Step 3:

For step 3, the images in ```/opt/Janus/WM/data/images/17--test_sifted/dN``` must be visually inspected and copied to directory ```/opt/Janus/WM/data/images/18--test-selected/dN/C```.  Care must be taken to select and copy images into the proper class ```C``` directory.  Misclassifying an image at this stage will result in erroneous accuracy calculations at a later stage--with affects that are not easy to investigate.  The images can be selected and copied in bulk or individually.

## Test TensorFlow Inception v4 Model for Image Classification

A dedicated Ubuntu computer with nVidia GPU should be set aside for this step.  With 2,000 training images, testing time has been observed to take 2 hours.  More images will significantly increase testing time.  

1. Code must be downloaded to the proper location and additional directories setup as noted above.
2. If using a dedicated computer, images located in the ```/opt/Janus/WM/data/images/19--test```directory must be copied to the identical location on the dedicated computer.  Additionally all models in ```/opt/Janus/WM/model``` and weights in ```/opt/Janus/WM/weights``` directories must be copied to identical locations.
3. The file ```/opt/Janus/WM/python3/config/core.py``` must be edited as noted here:

```
self.batch_proc_en_dict = {
	'convert_h264': False,
	'build_train_digits': False,
	'sift_train_digits': False,
	'shift_train_digits': False,
	'split_dataset': False,
	'train_model': False,
	'build_test_images': False,
	'sift_test_digits': False,
	'save_test_digits': False,
	'test_model': True,
}
```

4. Open terminal and execute BASH code: 

```
username@hostname:~$ sudo /opt/Janus/WM/python3/main-train.py
```

This produces a ```CSV``` file in ```/opt/Janus/WM/data/results``` directory with file name convention of ```Predictions_vN_AAA_BBB_CCC_YYYY-MM-DD_HHMM.cs```, where ```AAA``` is the train version number, ```BBB``` is the test version number, and ```CCC``` is the test set number.  These numbers can be set in ```/opt/Janus/WM/python3/config/core.py``` near the top of the file, according to a designated scheme.  The ```N``` number designates the class number being tested.  It is set in ```/opt/Janus/WM/python3/main-test.py``` with the ```value``` python variable near the end of the file.




