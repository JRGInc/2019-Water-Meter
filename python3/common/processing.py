#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import os
import time
from common import img_ops
from config.core import CoreCfg
from config.tensor import TensorCfg


def process_images(
    mode_str: str
) -> bool:
    """
    Executes image processing to prepare images for tensor flow operations

    :param mode_str: str

    :return: err_vals_dict['img_digs']: bool
    """
    logfile = 'januswm-' + mode_str
    logger = logging.getLogger(logfile)

    timea = time.time()

    core_cfg = CoreCfg()
    tensor_cfg = TensorCfg(core_cfg=core_cfg)

    # Error values dictionary
    err_vals_dict = {
        'img_scale': True,
        'img_screw': True,
        'img_rotd': True,
        'img_digw': True,
        'img_digs': True
    }

    img_path_dict = core_cfg.get(attrib='img_path_dict')
    orig_path = img_path_dict['orig']
    tf_dict = tensor_cfg.get(attrib='tf_dict')

    for img_orig_name in sorted(os.listdir(orig_path)):
        img_scale_name = 'scale' + img_orig_name[4::]
        img_screw_name = 'screw' + img_orig_name[4::]
        img_grotd_name = 'grotd' + img_orig_name[4::]
        img_frotd_name = 'frotd' + img_orig_name[4::]
        img_rect_name = 'rect' + img_orig_name[4::]
        img_digw_name = 'digw' + img_orig_name[4::]

        img_orig_url = os.path.join(
           img_path_dict['orig'],
           img_orig_name
        )
        img_scale_url = os.path.join(
            img_path_dict['scale'],
            img_scale_name
        )
        img_screw_url = os.path.join(
            img_path_dict['screw'],
            img_screw_name
        )
        img_grotd_url = os.path.join(
            img_path_dict['grotd'],
            img_grotd_name
        )
        img_frotd_url = os.path.join(
            img_path_dict['frotd'],
            img_frotd_name
        )
        img_rect_url = os.path.join(
            img_path_dict['rect'],
            img_rect_name
        )
        img_digw_url = os.path.join(
            img_path_dict['digw'],
            img_digw_name
        )

        # Scale image and return image
        img_scale, err_vals_dict['img_scale'] = img_ops.scale(
            err_xmit_url='',
            img_orig_url=img_orig_url,
            img_scale_url=img_scale_url
        )
        log = 'Scale image error: {0}'.format(err_vals_dict['img_scale'])
        logger.info(log)
        print(log)

        # Find screws and return sorted list
        img_screw_list = []
        if not err_vals_dict['img_scale']:
            img_screw_list, err_vals_dict['img_screw'] = img_ops.find_screws(
                img_scale=img_scale,
                err_xmit_url='',
                img_orig_url=img_scale_url,
                img_screw_url=img_screw_url,
                img_screw_ret=True
            )
            log = 'Find screws error: {0}'.format(err_vals_dict['img_screw'])
            logger.info(log)
            print(log)

        # Find angle of rotation if no gray-scale error
        img_rotd = None
        if not err_vals_dict['img_screw']:

            img_rotd, err_vals_dict['img_rotd'] = img_ops.rotate(
                err_xmit_url='',
                img_orig_url=img_orig_url,
                img_grotd_url=img_grotd_url,
                img_frotd_url=img_frotd_url,
                img_screw_list=img_screw_list
            )
            log = 'Find angle error: {0}'.format(err_vals_dict['img_rotd'])
            logger.info(log)
            print(log)

        # Crop to individual digits if no gray-scale error
        img_digw = None
        if not err_vals_dict['img_rotd']:

            # Crop close to digit window, leave some space for differences in zoom
            img_digw, err_vals_dict['img_digw'] = img_ops.crop_rect(
                img_rotd=img_rotd,
                err_xmit_url='',
                img_rect_url=img_rect_url,
                img_digw_url=img_digw_url,
                tf_dict=tf_dict
            )
            log = 'Crop digit window error: {0}'.format(err_vals_dict['img_digw'])
            logger.info(log)
            print(log)

        if not err_vals_dict['img_digw']:
            err_vals_dict['img_digs'] = img_ops.crop_digits(
                img_digw=img_digw,
                img_digw_url=img_digw_url,
                img_path_dict=img_path_dict,
                tf_dict=tf_dict,
                err_xmit_url='',
                mode_str=mode_str,
            )
            log = 'Crop digits error: {0}'.format(err_vals_dict['img_digs'])
            logger.info(log)
            print(log)

    print('Total image processing time elapsed: {0} sec'.format(time.time() - timea))
    return err_vals_dict['img_digs']
