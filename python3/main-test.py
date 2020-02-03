#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':

    import logging.config
    import time
    from common import processing
    from machine.tensor import test
    from common.data_ops import sift_digits
    from common.img_ops import save_test_image
    from config.capture import CaptureCfg
    from config.core import CoreCfg
    from config.tensor import TensorCfg
    from config.log import LogCfg

    log_config_obj = LogCfg()
    logging.config.dictConfig(log_config_obj.config)
    logfile = 'januswm-test'
    logger = logging.getLogger(logfile)

    mode_str = 'test'
    core_cfg = CoreCfg()
    capture_cfg = CaptureCfg(core_cfg=core_cfg)
    tensor_cfg = TensorCfg(core_cfg=core_cfg)

    err_vals_dict = {
        'build_test_images': True,
        'sift_test_digits': True,
        'save_test_digits': True,
        'test_model': True
    }

    batch_proc_en_dict = core_cfg.get(attrib='batch_proc_en_dict')
    data_path_dict = core_cfg.get(attrib='data_path_dict')
    img_path_dict = core_cfg.get(attrib='img_path_dict')
    train_url_dict = core_cfg.get(attrib='train_url_dict')
    test_url_str = core_cfg.get(attrib='test_url_str')
    tf_dict = tensor_cfg.get(attrib='tf_dict')

    if batch_proc_en_dict['build_test_images']:
        err_vals_dict['build_test_images'] = processing.process_images(
            mode_str=mode_str
        )
        log = 'Build test images error: {0}'.format(err_vals_dict['build_test_images'])
        logger.info(log)
        print(log)

    if batch_proc_en_dict['sift_test_digits']:
        err_vals_dict['sift_test_digits'] = sift_digits(
            mode_str='test',
            orig_path=img_path_dict['digs'],
            sift_path=img_path_dict['test_sift']
        )
        log = 'Sift digits error: {0}'.format(err_vals_dict['sift_test_digits'])
        logger.info(log)
        print(log)

    if batch_proc_en_dict['save_test_digits']:
        err_vals_dict['save_test_digits'] = save_test_image(
            orig_path=img_path_dict['test_sel'],
            shift_path=img_path_dict['test']
        )
        log = 'Shift digits error: {0}'.format(err_vals_dict['save_test_digits'])
        logger.info(log)
        print(log)

    if batch_proc_en_dict['test_model']:
        value = 8
        print('Performing predictions for value: {0}'.format(value))
        err_vals_dict['test_model'] = test(
            tf_dict=tf_dict,
            value=str(value)
        )
        log = 'Test model error for value {0}: {1} \n'.format(value, err_vals_dict['test_model'])
        logger.info(log)
        print(log)
        time.sleep(30)

    print('Machine Learning and Testing Script Concluded.')
