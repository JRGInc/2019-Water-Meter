#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import time
from common import img_ops, picamera
from machine import tensor

logfile = 'januswm-capture'
logger = logging.getLogger(logfile)


def capture(
    core_cfg: any,
    capture_cfg: any,
    tensor_cfg: any
) -> bool:
    """
    Captures image, processes captured image, makes TensorFlow
    prediction, then transmits image

    :param core_cfg: any
    :param capture_cfg: any
    :param tensor_cfg: any

    :return: err_vals_dict['img_olay']: bool
    """
    timea = time.time()

    # Error values dictionary
    err_vals_dict = {
        'img_orig': True,
        'img_scale': True,
        'img_screw': True,
        'img_rotd': True,
        'img_digw': True,
        'img_digs': True,
        'img_olay': True,
        'pred_vals': True,
    }

    # Load configuration settings
    img_path_dict = core_cfg.get(attrib='img_path_dict')

    img_seq = capture_cfg.get(attrib='img_seq')
    img_orig_dtg = capture_cfg.get(attrib='img_orig_dtg')
    img_url_dict = capture_cfg.get(attrib='img_url_dict')
    led_cfg_dict = capture_cfg.get(attrib='led_cfg_dict')
    led_set_dict = capture_cfg.get(attrib='led_set_dict')
    cam_cfg_dict = capture_cfg.get(attrib='cam_cfg_dict')
    pred_cfg_dict = capture_cfg.get(attrib='pred_cfg_dict')
    err_xmit_url = capture_cfg.get(attrib='err_xmit_url')
    img_retain_dict = capture_cfg.get(attrib='img_retain_dict')

    tf_dict = tensor_cfg.get(attrib='tf_dict')

    # LED flash, Camera warm-up & original image capture
    # img_test_orig_url = os.path.join(
    #     '/opt/Janus/WM/data/images/00--test',
    #     'test_orig_0.jpg'
    # )
    # img_test_rotd_url = os.path.join(
    #     '/opt/Janus/WM/data/images/00--test',
    #     'test_rotd_4.jpg'
    # )
    # err_vals_dict['img_rotd'] = img_ops.rotate(
    #     err_xmit_url=err_xmit_url,
    #     img_orig_url=img_test_orig_url,
    #     img_rotd_url=img_test_rotd_url,
    #     img_rotd_ang=8.00,
    #     img_rotd_fmt=img_fmt_dict['rotd']
    # )
    # err_vals_dict['img_orig'] = file_ops.copy_file(
    #     data_orig_url=img_test_rotd_url,
    #     data_dest_url=img_url_dict['orig']
    # )

    err_vals_dict['img_orig'] = picamera.snap_shot(
        err_xmit_url=err_xmit_url,
        led_cfg_dict=led_cfg_dict,
        led_set_dict=led_set_dict,
        cam_cfg_dict=cam_cfg_dict,
        img_orig_url=img_url_dict['orig']
    )
    # img_url_dict['orig'] = '/opt/Janus/WM/data/images/01--original/orig_2020-01-14_1218_3000016.jpg'
    # print(img_url_dict)
    print('Capture error: {0}'.format(err_vals_dict['img_orig']))

    # Proceed if flash and image capture was successful
    img_scale = None
    if not err_vals_dict['img_orig']:
        # Scale image to correct size
        img_scale, err_vals_dict['img_scale'] = img_ops.scale(
            err_xmit_url=err_xmit_url,
            img_orig_url=img_url_dict['orig'],
            img_scale_url=img_url_dict['scale'],
        )
        print('Scale error: {0}'.format(err_vals_dict['img_scale']))

    img_screw_list = [None, None, None, None, None]
    if not err_vals_dict['img_scale']:

        # Find screws and return sorted list
        img_screw_list, err_vals_dict['img_screw'] = img_ops.find_screws(
            img_scale=img_scale,
            err_xmit_url=err_xmit_url,
            img_orig_url=img_url_dict['orig'],
            img_screw_url=img_url_dict['screw'],
            img_screw_ret=img_retain_dict['screw']
        )
        print('Find screw error: {0}'.format(err_vals_dict['img_screw']))

    # Rotate image if no find screw error
    img_rotd = None
    if not err_vals_dict['img_screw']:
        img_rotd, err_vals_dict['img_rotd'] = img_ops.rotate(
            err_xmit_url=err_xmit_url,
            img_orig_url=img_url_dict['scale'],
            img_grotd_url=img_url_dict['grotd'],
            img_frotd_url=img_url_dict['frotd'],
            img_screw_list=img_screw_list,
        )
        print('Rotation error: {0}'.format(err_vals_dict['img_rotd']))

    # Crop to individual digits if no gray-scale error
    img_digw = None
    if not err_vals_dict['img_rotd']:

        # Crop close to digit window, leave some space for differences in zoom
        img_digw, err_vals_dict['img_digw'] = img_ops.crop_rect(
            img_rotd=img_rotd,
            err_xmit_url=err_xmit_url,
            img_rect_url=img_url_dict['rect'],
            img_digw_url=img_url_dict['digw'],
            tf_dict=tf_dict
        )
        print('Digit window error: {0}'.format(err_vals_dict['img_digw']))

    if not err_vals_dict['img_digw']:
        err_vals_dict['img_digs'] = img_ops.crop_digits(
            img_digw=img_digw,
            img_digw_url=img_url_dict['digw'],
            img_path_dict=img_path_dict,
            tf_dict=tf_dict,
            err_xmit_url='',
            mode_str='pred',
        )
        print('Crop digits error: {0}'.format(err_vals_dict['img_digs']))

    img_olay_text = 'Date & Time: ' + \
        img_orig_dtg.split('_')[0] + \
        ' ' + img_orig_dtg.split('_')[1]

    if pred_cfg_dict['pred_en']:

        # Execute TensorFlow prediction
        timeb = time.time()

        # Only import this library if predictions are enabled and
        # image is successfully converted to numpy array

        err_vals_dict['pred_vals'], pred_list, img_olay_text_values = tensor.predict(
            err_xmit_url=err_xmit_url,
            img_seq=img_seq,
            img_digw_url=img_url_dict['digw'],
            img_orig_dtg=img_orig_dtg,
            tf_dict=tf_dict
        )
        img_olay_text = img_olay_text + img_olay_text_values
        print('Prediction time elapsed: {0} sec'.format(time.time() - timeb))

    # Overlay image with date-time stamp and value if
    # no TensorFlow error.
    # if not err_vals_dict['pred_vals']:
    else:
        img_olay_text = img_olay_text + '           Prediction Not Enabled'

    if not err_vals_dict['img_digw']:
        err_vals_dict['img_olay'] = img_ops.overlay(
            err_xmit_url=err_xmit_url,
            img_digw_url=img_url_dict['digw'],
            img_olay_url=img_url_dict['olay'],
            img_olay_text=img_olay_text
        )
        print('Overlay error: {0}'.format(err_vals_dict['img_olay']))

    print('Total capture time elapsed: {0} sec'.format(time.time() - timea))
    return err_vals_dict['img_olay']
