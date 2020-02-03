__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import os
import shutil
import tensorflow as tf
from common.file_ops import copy_file


logfile = 'januswm'
logger = logging.getLogger(logfile)


def sift_digits(
    orig_path: str,
    sift_path: str
) -> bool:
    """
    Sifts digit images and separates them into folders according to digit position

    :param orig_path: str
    :param sift_path: str

    :return sift_err: bool
    """
    sift_err = False

    try:
        for img_orig_name in os.listdir(orig_path):
            img_orig_url = os.path.join(
                orig_path,
                img_orig_name
            )
            img_orig_core_name = str(img_orig_name.split(sep='.')[0])

            digit = 'd' + img_orig_core_name.split(sep='_')[4]

            sift_dest_path = os.path.join(
                sift_path,
                digit
            )
            img_orig_name = img_orig_core_name + '.jpg'
            img_dest_url = os.path.join(
                sift_dest_path,
                img_orig_name
            )

            copy_err = copy_file(
                data_orig_url=img_orig_url,
                data_dest_url=img_dest_url
            )

            if copy_err:
                sift_err = True
                log = 'Failed to sift digits in path {0}.'.format(orig_path)
                logger.error(msg=log)
                print(log)
                break

    except Exception as exc:
        sift_err = True
        log = 'Failed to sift digits in path {0}.'.format(orig_path)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return sift_err


def rename_digits(
    orig_path: str,
    rename_path: str
) -> bool:
    """
    Renames digit images and separates them into folders according to digit position

    :param orig_path: str
    :param rename_path: str

    :return sift_err: bool
    """
    sift_err = False

    for root, dirs, images in os.walk(top=orig_path):
        for image in images:
            img_orig_url = os.path.join(root, image)
            img_parts = image.split(sep='_')
            if len(img_parts[2]) == 7:
                img_name = img_parts[0] + '_' +\
                    img_parts[1] + '_' +\
                    img_parts[3] + '_' +\
                    img_parts[4] + '_' +\
                    img_parts[2] + '_' +\
                    img_parts[5]
            else:
                img_name = image

            path_parts = root.split(sep='/')
            digit_path = path_parts[8] + '/'
            value_path = path_parts[9] + '/'
            img_rename_url = os.path.join(
                rename_path,
                digit_path,
                value_path,
                img_name
            )

            shutil.copy2(
                src=img_orig_url,
                dst=img_rename_url
            )

    return sift_err


def tf_converter(
    saved_mdl_url: str
):
    model = tf.keras.models.load_model(saved_mdl_url)
    tf.saved_model.save(model, '/opt/Janus/WM/model/pb')

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='/opt/Janus/WM/model/pb')
    tflite_model = converter.convert()
    open('/opt/Janus/WM/model/tflite/saved_model.tflite', "wb").write(tflite_model)


if __name__ == '__main__':
    # sift_digits(
    #     orig_path='/opt/Janus/WM/data/images/17--test_sifted/orig/',
    #     sift_path='/opt/Janus/WM/data/images/17--test_sifted/'
    # )
    # rename_digits(
    #     orig_path='/opt/Janus/WM/data/images/18--test_selected/orig/',
    #     rename_path='/opt/Janus/WM/data/images/18--test_selected/'
    # )
    tf_converter(saved_mdl_url='/opt/Janus/WM/model/keras/tf_mdl_00005-0.00001.h5')
