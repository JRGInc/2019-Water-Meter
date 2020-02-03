from __future__ import absolute_import, division, print_function
import csv
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, NUM_CLASSES, save_model_dir, model_index, save_every_n_epoch, train_dir, \
    valid_dir, test_dir, cache_dir
import os
from machine import inception_v4
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import time


if __name__ == '__main__':
    # GPU settings
    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_dir = Path(train_dir)
    valid_dir = Path(valid_dir)
    test_dir = Path(test_dir)
    # cache_dir = Path(cache_dir)
    CLASS_NAMES = np.array([int(item.name) for item in train_dir.glob('*')])
    train_count = len(list(train_dir.glob('*/*.jpg')))
    valid_count = len(list(valid_dir.glob('*/*.jpg')))
    test_count = len(list(test_dir.glob('*/*.jpg')))
    print('TRAIN IMAGE COUNT: {0}: '.format(train_count))
    print('VALID IMAGE COUNT: {0}: '.format(valid_count))
    print('TEST IMAGE COUNT: {0}: '.format(test_count))

    TRAIN_STEPS = np.ceil(train_count / BATCH_SIZE)
    VALID_STEPS = np.ceil(valid_count / BATCH_SIZE)
    TEST_STEPS = np.ceil(test_count / BATCH_SIZE)

    train_list_ds = tf.data.Dataset.list_files(str(train_dir / '*/*'))
    valid_list_ds = tf.data.Dataset.list_files(str(valid_dir / '*/*'))

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return tf.strings.to_number(parts[-2], tf.dtypes.int32)  # == CLASS_NAMES

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=1)
        # resize the image to the desired size.
        img = tf.image.resize_with_pad(img, IMAGE_WIDTH, IMAGE_HEIGHT)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        return tf.image.convert_image_dtype(img, tf.float32)


    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    valid_labeled_ds = valid_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


    def prepare_for_training(ds, cache=None, shuffle_buffer_size=200):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE, drop_remainder=True)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    train_ds = prepare_for_training(ds=train_labeled_ds, cache=cache_dir)
    valid_ds = prepare_for_training(ds=valid_labeled_ds, cache=cache_dir)

    # image_batch, label_batch = next(iter(test_ds))
    #
    # print(image_batch.shape)
    # print(label_batch.shape)

    # show_batch(image_batch.numpy(), label_batch.numpy())

    # create model
    log_dir = 'logs/'

    # callbacks = [
    #     # Interrupt training if `val_loss` stops improving for over 2 epochs
    #     tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', min_delta=0, verbose=1, mode='auto'),
    #     # Write TensorBoard logs to `./logs` directory, enabling causes error that causes first epoch to fail
    #     # Do not run tensorboard callback for production model
    #     # keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', histogram_freq=1)
    # ]
    #
    # timea = time.time()
    #
    # model = inception_v4.create_inception_v4()
    # # Specify the training configuration (optimizer, loss, metrics)
    # model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
    #               # Loss function to minimize
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #               # List of metrics to monitor
    #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # # model.summary()
    # H = model.fit(
    #     x=train_ds,
    #     epochs=EPOCHS,
    #     steps_per_epoch=TRAIN_STEPS,
    #     validation_data=valid_ds,
    #     validation_steps=VALID_STEPS,
    #     callbacks=callbacks,
    #     verbose=1
    # )
    #
    # print('TRAINING TIME: {0}'.format(round((time.time() - timea), 3)))
    # nbr_epochs = len(H.history["val_loss"])
    # print('Training ended at {0} epochs.'.format(nbr_epochs))
    # N = np.arange(0, nbr_epochs)
    # title = "Training Loss and Accuracy"
    #
    # # plot the training loss and accuracy
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(N, H.history["loss"], label="train_loss")
    # plt.plot(N, H.history["val_loss"], label="val_loss")
    # plt.plot(N, H.history["sparse_categorical_accuracy"], label="train_acc")
    # plt.plot(N, H.history["val_sparse_categorical_accuracy"], label="val_acc")
    # plt.title(title)
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend()
    # plt.savefig("loss and accuracy.png")
    #
    # # save the whole model

    model_url = os.path.join(save_model_dir)
    # # model.save(model_url, save_format='tf')
    # model.save_weights(model_url)

    # model_test = tf.keras.models.load_model(model_url)
    model_test = get_model()
    model_test.load_weights(model_url)
    # model_test.summary()

    print('TEST STEPS: {0}'.format(TEST_STEPS))
    correct = 0
    index = 0

    with open('MNIST prediction results.csv', mode='w') as predict_file:
        predict_file_writer = csv.writer(
            predict_file,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        predict_file_writer.writerow([
            'File',
            'Label',
            'Prediction',
            'Probability'
        ])

    label_val = None
    prediction = None
    probability = None
    test_list_ds = None
    test_labeled_ds = None
    test_ds = None

    num_files = 0
    for root, dirs, files in os.walk(top=test_dir):
        for file in files:
            num_files += 1

    print('NUMBER TEST FILES: {0}'.format(num_files))
    print('TEST IMAGE COUNT: {0}: '.format(test_count))

    for root, dirs, files in os.walk(top=test_dir):
        for file in files:
            file_url = os.path.join(root, file)
            print(file_url)
            test_list_ds = tf.data.Dataset.list_files(str(file_url))
            test_labeled_ds = test_list_ds.map(process_path)
            test_ds = test_labeled_ds.batch(BATCH_SIZE)
            for record in test_ds:
                for label in record[1]:
                    # print(label.numpy())
                    label_val = int(label.numpy())

            predictions = model_test.predict(test_ds, steps=1)
            for element in predictions:
                # print(index)
                prediction = element.argmax(axis=0)
                # print(prediction)
                probability = element[prediction]
                # print(probability)

            if label_val == prediction:
                correct += 1

            with open('MNIST prediction results.csv', mode='a') as predict_file:
                predict_file_writer = csv.writer(
                    predict_file,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL
                )
                predict_file_writer.writerow([
                    file_url,
                    label_val,
                    prediction,
                    probability
                ])

            index += 1
            test_list_ds = None
            test_labeled_ds = None
            test_ds = None
            predictions = None
            predict_file_writer = None

            print('Accuracy: {0}%'.format(round(float(correct / test_count), 4) * 100))
            print('FILE NUMBER: {0}'.format(index))

            if index == num_files:
                print('REACHED END OF TEST DIRECTORY (INNER LOOP)')
                sys.exit()
