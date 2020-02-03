#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'


if __name__ == '__main__':
    import logging
    import logging.config
    import os
    import sys
    import time as ttime
    from common import capture, file_ops, img_ops
    from config.core import CoreCfg
    from config.capture import CaptureCfg
    from config.tensor import TensorCfg
    from config.log import LogCfg
    from datetime import *
    from tendo import singleton

    # Configure logging
    log_config_obj = LogCfg()
    logging.config.dictConfig(log_config_obj.config)
    logfile = 'januswm-capture'
    logger = logging.getLogger(name=logfile)
    logging.getLogger(name=logfile).setLevel(level=logging.INFO)

    # Single instance
    try:
        me = singleton.SingleInstance()

    except singleton.SingleInstanceException:
        log = 'Duplicate capture process, shutting down.'
        logger.warning(msg=log)
        print(log)
        sys.exit(-1)

    for i in range(1, 6):
        logger.info(msg='')

    log = 'JanusWM Capture logging started'
    logger.info(msg=log)

    core_cfg = CoreCfg()
    cfg_url_dict = core_cfg.get(attrib='cfg_url_dict')
    data_path_dict = core_cfg.get(attrib='data_path_dict')

    timea = ttime.time()

    minute = int(datetime.today().strftime('%M'))
    hour = int(datetime.today().strftime('%H'))
    execution_minute = (hour * 60) + minute

    capture_cfg = CaptureCfg(core_cfg=core_cfg)

    img_seq = capture_cfg.get(attrib='img_seq')
    img_orig_dtg = capture_cfg.get(attrib='img_orig_dtg')
    img_capt_dict = capture_cfg.get(attrib='img_capt_dict')
    img_fmt_dict = capture_cfg.get(attrib='img_fmt_dict')
    update_freq = capture_cfg.get(attrib='update_freq')
    img_xmit_dict = capture_cfg.get(attrib='img_xmit_dict')
    pred_cfg_dict = capture_cfg.get(attrib='pred_cfg_dict')
    gprs_cfg_dict = capture_cfg.get(attrib='gprs_cfg_dict')
    err_xmit_url = capture_cfg.get(attrib='err_xmit_url')

    tensor_cfg = TensorCfg(core_cfg=core_cfg)

    print(execution_minute)
    log = 'Capture execution minute: {0}'.format(execution_minute)
    logger.info(msg=log)

    img_olay_err = False
    if img_capt_dict['img_capt_freq'] > 0:
        if not (execution_minute % img_capt_dict['img_capt_freq']):
            img_olay_err = capture.capture(
                core_cfg=core_cfg,
                capture_cfg=capture_cfg,
                tensor_cfg=tensor_cfg
            )

    # Get list of overlaid images and find latest, pass to transmission
    # If list is empty pass empty string to transmission
    img_path_dict = core_cfg.get(attrib='img_path_dict')
    for key in img_xmit_dict:
        if img_xmit_dict[key] > 0:
            if not (execution_minute % img_xmit_dict[key]):

                img_key_list = os.listdir(img_path_dict[key])
                if not img_key_list:
                    img_key_last = 'None'
                else:
                    img_key_last = max(
                        [os.path.join(img_path_dict[key], i) for i in img_key_list],
                        key=os.path.getmtime
                    )

                if (key == 'orig') or (key == 'rotd'):
                    img_qual = 15
                    img_crop_en = True
                else:
                    img_qual = 70
                    img_crop_en = False

                if not (img_key_last == 'None'):
                    img_crop_dict = {
                        'ulx': 900,
                        'uly': 900,
                        'brx': 2100,
                        'bry': 1700,
                    }

                    if not (key == 'cont') and \
                            not (key == 'digs') and \
                            not (key == 'pred'):

                        log = 'Moving image {0} file to transmission folder'.format(img_key_last)
                        logger.info(msg=log)
                        print(log)

                        img_xmit_err = img_ops.copy_image(
                            err_xmit_url=err_xmit_url,
                            img_orig_url=img_key_last,
                            img_dest_path=data_path_dict['xmit'],
                            img_dest_name=os.path.basename(img_key_last),
                            img_crop_dict=img_crop_dict,
                            img_dest_qual=img_qual,
                            img_crop_en=img_crop_en
                        )

                    else:
                        img_digs_file = os.path.basename(img_key_last)
                        img_dig_file_base = img_digs_file.split(sep='.')[0]
                        img_dig_file_ext = img_digs_file.split(sep='.')[1]
                        img_dig_parts = img_digs_file.split(sep='_')

                        for digit in range(0, 6):
                            img_dig_file = str(img_dig_parts[0]) + '_' + \
                                str(img_dig_parts[1]) + '_' + \
                                str(img_dig_parts[2]) + '_' + \
                                str(img_dig_parts[3]) + '_' + \
                                str(digit) + '.' + \
                                str(img_dig_file_ext)

                            img_dig_url = os.path.join(
                                img_path_dict[key],
                                img_dig_file
                            )

                            log = 'Moving image {0} file to transmission folder'.format(img_dig_url)
                            logger.info(msg=log)
                            print(log)

                            img_xmit_err = img_ops.copy_image(
                                err_xmit_url=err_xmit_url,
                                img_orig_url=img_dig_url,
                                img_dest_path=data_path_dict['xmit'],
                                img_dest_name=os.path.basename(img_dig_url),
                                img_crop_dict=img_crop_dict,
                                img_dest_qual=img_qual,
                                img_crop_en=img_crop_en
                            )

            if img_xmit_dict[key] == 1:
                set_err, msg = capture_cfg.set(
                    section='Image_Transmission',
                    attrib=key,
                    value='0'
                )

    if pred_cfg_dict['last_freq'] > 0:
        if not (execution_minute % pred_cfg_dict['last_freq']):
            last_name_str = 'last_' + img_orig_dtg + '_' + img_seq + '.txt'
            last_url_str = os.path.join(
                data_path_dict['last'],
                last_name_str
            )
            xmit_url_str = os.path.join(
                data_path_dict['xmit'],
                last_name_str
            )

            log = 'Moving {0} file to transmission folder'.format(last_url_str)
            logger.info(msg=log)
            print(log)

            file_ops.copy_file(
                data_orig_url=last_url_str,
                data_dest_url=xmit_url_str
            )

            if pred_cfg_dict['last_freq'] == 1:
                set_err, msg = capture_cfg.set(
                    section='Prediction_Settings',
                    attrib='last_prediction_freq',
                    value='0'
                )

    if pred_cfg_dict['hist_freq'] > 0:
        if not (execution_minute % pred_cfg_dict['hist_freq']):
            if execution_minute == 0:
                img_date = img_orig_dtg.split('_')[0]
                img_year = int(img_date.split('-')[0])
                img_month = int(img_date.split('-')[1])
                img_day = int(img_date.split('-')[2])
                yesterday = datetime(img_year, img_month, img_day) - timedelta(days=1)
                hist_name_str = 'hist_' + yesterday.strftime('%Y-%m-%d') + '.txt'
            else:
                hist_name_str = 'hist_' + img_orig_dtg.split('_')[0] + '.txt'

            hist_url_str = os.path.join(
                data_path_dict['hist'],
                hist_name_str
            )
            xmit_url_str = os.path.join(
                data_path_dict['xmit'],
                hist_name_str
            )

            log = 'Moving {0} file to transmission folder'.format(hist_url_str)
            logger.info(msg=log)
            print(log)

            file_ops.copy_file(
                data_orig_url=hist_url_str,
                data_dest_url=xmit_url_str
            )

            if pred_cfg_dict['hist_freq'] == 1:
                set_err, msg = capture_cfg.set(
                    section='Prediction_Settings',
                    attrib='prediction_history_freq',
                    value='0'
                )

    if img_capt_dict['img_capt_freq'] > 0:
        if not (execution_minute % img_capt_dict['img_capt_freq']):

            # Remove all unwanted images only after transmissions occur
            img_retain_dict = capture_cfg.get(attrib='img_retain_dict')
            if not img_retain_dict['orig']:
                img_ops.remove_images(img_path=img_path_dict['orig'])
                log = 'Deleted original images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained original images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['scale']:
                img_ops.remove_images(img_path=img_path_dict['scale'])
                log = 'Deleted scaled images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained scaled images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['screw']:
                img_ops.remove_images(img_path=img_path_dict['screw'])
                log = 'Deleted screw images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained screw images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['grotd']:
                img_ops.remove_images(img_path=img_path_dict['grotd'])
                log = 'Deleted gross rotated images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained gross rotated images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['frotd']:
                img_ops.remove_images(img_path=img_path_dict['frotd'])
                log = 'Deleted fine rotated images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained fine rotated images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['rect']:
                img_ops.remove_images(img_path=img_path_dict['rect'])
                log = 'Deleted rectangled images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained rectangled images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['digw']:
                img_ops.remove_images(img_path=img_path_dict['digw'])
                log = 'Deleted digit window images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained digit window images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['inv']:
                img_ops.remove_images(img_path=img_path_dict['inv'])
                log = 'Deleted inverted window images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained inverted window images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['cont']:
                img_ops.remove_images(img_path=img_path_dict['cont'])
                log = 'Deleted contoured digit images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained contoured digit images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['digs']:
                img_ops.remove_images(img_path=img_path_dict['digs'])
                log = 'Deleted digit images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained digit images on disk.'
                logger.info(msg=log)
            print(log)

            if not img_retain_dict['olay']:
                img_ops.remove_images(img_path=img_path_dict['olay'])
                log = 'Deleted overlaid images from disk.'
                logger.warning(msg=log)
            else:
                log = 'Retained overlaid images on disk.'
                logger.info(msg=log)
            print(log)

    if img_capt_dict['img_capt_freq'] == 1:
        set_err, msg = capture_cfg.set(
            section='Capture_Settings',
            attrib='image_capture_freq',
            value='0'
        )

    # increment image sequence only after image is captured,
    # even if there was an error
    img_seq = str(int(img_seq) + 1)
    file_ops.f_request(
        file_cmd='file_replace',
        file_name=cfg_url_dict['seq'],
        num_bytes=7,
        data_file_in=[img_seq]
    )

    print('Total capture execution time elapsed: {0} sec'.format(ttime.time() - timea))
