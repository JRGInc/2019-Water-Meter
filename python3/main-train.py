#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':

    import logging.config
    import os
    from common import conversion
    from common import processing
    from machine.tensor import version, train
    from common.data_ops import SplitDataset, sift_digits
    from common.img_ops import shift_train_image
    from config.capture import CaptureCfg
    from config.core import CoreCfg
    from config.tensor import TensorCfg
    from config.log import LogCfg

    log_config_obj = LogCfg()
    logging.config.dictConfig(log_config_obj.config)
    logfile = 'januswm-train'
    logger = logging.getLogger(logfile)

    mode_str = 'train'
    core_cfg = CoreCfg()
    capture_cfg = CaptureCfg(core_cfg=core_cfg)
    tensor_cfg = TensorCfg(core_cfg=core_cfg)

    err_vals_dict = {
        'convert_h264': True,
        'build_train_digits': True,
        'sift_train_digits': True,
        'shift_train_digits': True,
        'split_dataset': True,
        'build_train_data': True,
        'build_train_labels': True,
        'train_model': True
    }

    batch_proc_en_dict = core_cfg.get(attrib='batch_proc_en_dict')
    data_path_dict = core_cfg.get(attrib='data_path_dict')
    img_path_dict = core_cfg.get(attrib='img_path_dict')
    train_url_dict = core_cfg.get(attrib='train_url_dict')
    mdl_url_str = core_cfg.get(attrib='mdl_url_str')
    valid_ratio = core_cfg.get(attrib='valid_ratio')
    tf_dict = tensor_cfg.get(attrib='tf_dict')

    if batch_proc_en_dict['convert_h264']:
        for mov_name in os.listdir(img_path_dict['mov']):
            vid_h264_url = os.path.join(
                img_path_dict['mov'],
                mov_name
            )
            vid_fps_int = 5

            img_core_name = str(mov_name.split('.')[0])
            img_orig_name = 'orig_' + img_core_name.split('_')[1] + '_' +\
                img_core_name.split('_')[2] + '_' + \
                img_core_name.split('_')[3] + '_sg%04d.jpg'
            img_orig_url = img_path_dict['orig'] + img_orig_name

            err_vals_dict['convert_h264'] = conversion.h264_to_jpg(
                vid_h264_url=vid_h264_url,
                vid_fps_int=vid_fps_int,
                img_orig_url=img_orig_url
            )
            log = 'Convert H264 error: {0}'.format(err_vals_dict['convert_h264'])
            logger.info(log)
            print(log)

    if batch_proc_en_dict['build_train_digits']:
        err_vals_dict['build_train_digits'] = processing.process_images(
            mode_str=mode_str
        )
        log = 'Build train digits error: {0}'.format(err_vals_dict['build_train_digits'])
        logger.info(log)
        print(log)

    if batch_proc_en_dict['sift_train_digits']:
        err_vals_dict['sift_train_digits'] = sift_digits(
            mode_str='train',
            orig_path=img_path_dict['digs'],
            sift_path=img_path_dict['trn_sift']
        )
        log = 'Sift digits error: {0}'.format(err_vals_dict['sift_train_digits'])
        logger.info(log)
        print(log)

    if batch_proc_en_dict['shift_train_digits']:
        err_vals_dict['shift_train_digits'] = shift_train_image(
            orig_path=img_path_dict['trn_sel'],
            shift_path=img_path_dict['trn'],
            tf_dict=tf_dict
        )
        log = 'Shift digits error: {0}'.format(err_vals_dict['shift_train_digits'])
        logger.info(log)
        print(log)

    if batch_proc_en_dict['split_dataset']:
        split_dataset = SplitDataset(
            train_imgs_dir=img_path_dict['trn'],
            valid_imgs_dir=img_path_dict['vld'],
            valid_ratio=valid_ratio,
            show_progress=True)
        split_dataset.start_splitting()

    if batch_proc_en_dict['train_model']:
        version()
        err_vals_dict['train_model'] = train(
            tf_dict=tf_dict
        )
        log = 'Train model error: {0}'.format(err_vals_dict['train_model'])
        logger.info(log)
        print(log)

    print('Machine Learning and Testing Script Concluded.')
