__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import configparser
import logging
import os
from common import file_ops
from datetime import *

logfile = 'januswm-capture'
logger = logging.getLogger(logfile)


class CaptureCfg(object):
    """
    Class attributes of configuration settings for capture operations
    """
    def __init__(
        self,
        core_cfg: any
    ) -> None:
        """
        Sets object properties either directly or via ConfigParser from ini file

        :param core_cfg: any
        """
        # Define image sequence, dtg, and name for captured image
        cfg_url_dict = core_cfg.get(attrib='cfg_url_dict')
        self.img_seq = file_ops.f_request(
            file_cmd='data_read',
            file_name=cfg_url_dict['seq'],
            num_bytes=7
        )
        self.img_orig_dtg = datetime.today().strftime('%Y-%m-%d_%H%M')
        img_core_name = self.img_orig_dtg + '_' + self.img_seq

        # Define file names for images in /img/* directories
        img_name_dict = {
            'orig': 'orig_' + img_core_name + '.jpg',
            'scale': 'scale_' + img_core_name + '.jpg',
            'screw': 'screw_' + img_core_name + '.jpg',
            'grotd': 'grotd_' + img_core_name + '.jpg',
            'frotd': 'frotd_' + img_core_name + '.jpg',
            'rect': 'rect_' + img_core_name + '.jpg',
            'digw': 'digw_' + img_core_name + '.jpg',
            'inv': 'inv_' + img_core_name + '.jpg',
            'cont': 'cont_' + img_core_name + '.jpg',
            'digs': 'digs_' + img_core_name + '.jpg',
            'pred': 'pred_' + img_core_name + '.jpg',
            'olay': 'olay_' + img_core_name + '.jpg'
        }

        err_xmit_name = 'errors_' + img_core_name + '.txt'

        # Define ful urls for images in /img/* paths
        img_path_dict = core_cfg.get(attrib='img_path_dict')
        self.img_url_dict = {
            'orig': os.path.join(
                img_path_dict['orig'],
                img_name_dict['orig']
            ),
            'scale': os.path.join(
                img_path_dict['scale'],
                img_name_dict['scale']
            ),
            'screw': os.path.join(
                img_path_dict['screw'],
                img_name_dict['screw']
            ),
            'grotd': os.path.join(
                img_path_dict['grotd'],
                img_name_dict['grotd']
            ),
            'frotd': os.path.join(
                img_path_dict['frotd'],
                img_name_dict['frotd']
            ),
            'rect': os.path.join(
                img_path_dict['rect'],
                img_name_dict['rect']
            ),
            'digw': os.path.join(
                img_path_dict['digw'],
                img_name_dict['digw']
            ),
            'inv': os.path.join(
                img_path_dict['inv'],
                img_name_dict['inv']
            ),
            'cont': os.path.join(
                img_path_dict['cont'],
                img_name_dict['cont']
            ),
            'digs': os.path.join(
                img_path_dict['digs'],
                img_name_dict['digs']
            ),
            'pred': os.path.join(
                img_path_dict['pred'],
                img_name_dict['pred']
            ),
            'olay': os.path.join(
                img_path_dict['olay'],
                img_name_dict['olay']
            )
        }

        data_path_dict = core_cfg.get(attrib='data_path_dict')
        self.err_xmit_url = os.path.join(
            data_path_dict['xmit'],
            err_xmit_name
        )

        self.ini_file = cfg_url_dict['capt']
        self.config = configparser.ConfigParser()
        self.config.read_file(f=open(self.ini_file))

        self.img_capt_dict = {
            # All frequencies specified in this file must be a
            # multiple of this number
            'exec_interval': self.config.getint(
                'Capture_Settings',
                'execution_interval'
            ),
            # Image capture frequency in minutes, minimum = 6 min and maximum = 1440
            'img_capt_freq': self.config.getint(
                'Capture_Settings',
                'image_capture_freq'
            )
        }

        # Prediction configuration dictionary
        self.pred_cfg_dict = {
            # Enables tensorflow prediction, overlay of images,
            # and transmission of overlaid images
            'pred_en': self.config.getboolean(
                'Prediction_Settings',
                'prediction_enable'
            ),
            'last_file': 'prediction_latest_' + img_core_name + '.txt',
            'hist_file': 'prediction_history_' + img_core_name + '.txt',
            # Specify which files to transmit and how frequently
            # 0 = do not transmit
            # 1 = single transmission only, resets to 0 after transmission
            # 6-1440 = minute intervals to transmit
            'last_freq': self.config.getint(
                'Prediction_Settings',
                'last_prediction_freq'
            ),
            'hist_freq': self.config.getint(
                'Prediction_Settings',
                'prediction_history_freq'
            )
        }

        # Specify which images to transmit and how frequently
        # 0 = do not transmit
        # 1 = single transmission only, resets to 0 after transmission
        # 6-1440 = minute intervals to transmit
        self.img_xmit_dict = {
            'orig': self.config.getint(
                'Image_Transmission',
                'original_freq'
            ),
            'scale': self.config.getint(
                'Image_Transmission',
                'scale_freq'
            ),
            'screw': self.config.getint(
                'Image_Transmission',
                'screws_freq'
            ),
            'grotd': self.config.getint(
                'Image_Transmission',
                'grotated_freq'
            ),
            'frotd': self.config.getint(
                'Image_Transmission',
                'frotated_freq'
            ),
            'rect': self.config.getint(
                'Image_Transmission',
                'rectangled_freq'
            ),
            'digw': self.config.getint(
                'Image_Transmission',
                'windowed_freq'
            ),
            'inv': self.config.getint(
                'Image_Transmission',
                'inverted_freq'
            ),
            'cont': self.config.getint(
                'Image_Transmission',
                'contoured_freq'
            ),
            'digs': self.config.getint(
                'Image_Transmission',
                'digits_freq'
            ),
            'pred': self.config.getint(
                'Image_Transmission',
                'prediction_freq'
            ),
            'olay': self.config.getint(
                'Image_Transmission',
                'overlaid_freq'
            )
        }

        # Specify which set of images to retain locally
        # True = retain, False = delete
        self.img_retain_dict = {
            'orig': self.config.getboolean(
                'Image_Retention',
                'original'
            ),
            'scale': self.config.getboolean(
                'Image_Retention',
                'scale'
            ),
            'screw': self.config.getboolean(
                'Image_Retention',
                'screws'
            ),
            'grotd': self.config.getboolean(
                'Image_Retention',
                'grotated'
            ),
            'frotd': self.config.getboolean(
                'Image_Retention',
                'frotated'
            ),
            'rect': self.config.getboolean(
                'Image_Retention',
                'rectangled'
            ),
            'digw': self.config.getboolean(
                'Image_Retention',
                'windowed'
            ),
            'inv': self.config.getboolean(
                'Image_Retention',
                'inverted'
            ),
            'cont': self.config.getboolean(
                'Image_Retention',
                'contoured'
            ),
            'digs': self.config.getboolean(
                'Image_Retention',
                'digits'
            ),
            'pred': self.config.getboolean(
                'Image_Retention',
                'prediction'
            ),
            'olay': self.config.getboolean(
                'Image_Retention',
                'overlaid'
            )
        }

        # LED configuration dictionary
        self.led_cfg_dict = {
            'count': 2,            # Number of LED pixels.
            'pin': 18,            # GPIO pin connected to the pixels (18 uses PWM!).
            'freq_hz': 800000,    # LED signal frequency in hertz (usually 800khz)
            'dma': 10,            # DMA channel to use for generating signal (try 10)
            'brightness': 255,    # Set to 0 for darkest and 255 for brightest
            'invert': False,    # True to invert the signal (when using NPN transistor level shift)
            'channel': 0        # set to '1' for GPIOs 13, 19, 41, 45 or 53
        }

        # LED rgbw settings dictionary
        self.led_set_dict = {
            'r': 100,
            'g': 100,
            'b': 100,
            'w': 255
        }

        # Py Camera configuration dictionary
        # PiCam v2 modes:
        # 1  1920x1080
        # 2  3280x2464
        # 5  1640x922
        # 6  1280x720
        # VGA 640x480
        # PAL 768x576
        # SVGA 800x600
        # XGA  1024x768
        self.cam_cfg_dict = {
            'mode': 2,
            'width': 1536,
            'height': 1536,
            'quality': 20,
            'shutter': 100000,              # Microseconds, 0 = auto
            'sharpness': 100,               # -100 to 100, 0 = default
            'saturation': 0,                # -100 to 100, 0 = default
            'rotation': 180,                # 0 to 359 degrees
            'exposure': 'antishake'         # exposure setting
        }

    def get(
        self,
        attrib: str
    ) -> any:
        """
        Gets configuration attributes

        :param attrib: str

        :return: any
        """
        if attrib == 'img_seq':
            return self.img_seq
        elif attrib == 'img_orig_dtg':
            return self.img_orig_dtg
        elif attrib == 'img_url_dict':
            return self.img_url_dict
        elif attrib == 'err_xmit_url':
            return self.err_xmit_url
        elif attrib == 'img_capt_dict':
            return self.img_capt_dict
        elif attrib == 'pred_cfg_dict':
            return self.pred_cfg_dict
        elif attrib == 'img_xmit_dict':
            return self.img_xmit_dict
        elif attrib == 'img_retain_dict':
            return self.img_retain_dict
        elif attrib == 'led_cfg_dict':
            return self.led_cfg_dict
        elif attrib == 'led_set_dict':
            return self.led_set_dict
        elif attrib == 'cam_cfg_dict':
            return self.cam_cfg_dict

    def set(
        self,
        section: str,
        attrib: str,
        value: str
    ) -> (bool, str):
        """
        Sets configuration attributes by updating ini file

        :param section: str
        :param attrib: str
        :param value: str
        :return: set_err: bool
        """
        set_err = False
        valid_section = True
        valid_option = True
        valid_value = True
        log = ''

        if section == 'Capture_Settings':
            if attrib == 'execution_interval':
                try:
                    if int(value) < 0:
                        log = 'Attribute value {0} is less than 1: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif (int(value) > 1) and (int(value) < 5):
                        log = 'Attribute value {0} is greater than 1 and less than 5: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) > 1440:
                        log = 'Attribute value {0} is greater than 1440: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    else:
                        if not (int(value) == 5) and \
                                not (int(value) == 6) and \
                                not (int(value) == 10) and \
                                not (int(value) == 12) and \
                                not (int(value) == 15) and \
                                not (int(value) == 20) and \
                                not (int(value) == 30):
                            log = 'Attribute value {0} is not a valid value (5, 6, 10, 12, 15, 20, 30): {1}.'.\
                                format(attrib, value)
                            logger.error(msg=log)
                            log1 = 'Retaining previous value.'.format(attrib)
                            logger.warning(msg=log1)
                            valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            elif attrib == 'image_capture_freq':
                try:
                    if int(value) < 0:
                        log = 'Attribute value {0} is less than 0: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif (int(value) > 1) and (int(value) < self.img_capt_dict['exec_interval']):
                        log = 'Attribute value {0} is between 1 and {1}: {2}.'. \
                            format(attrib, self.img_capt_dict['exec_interval'], value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) % self.img_capt_dict['exec_interval']:
                        log = 'Attribute value {0} is not divisible by execution interval: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) > 1440:
                        log = 'Attribute value {0} is greater than 1440: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            else:
                valid_option = False

        elif section == 'Prediction_Settings':
            if attrib == 'prediction_enable':
                if not (value == 'True') and not (value == 'False'):
                    log = 'Attribute value {0} is not a boolean: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            if attrib == 'last_prediction_freq':
                try:
                    if int(value) < 0:
                        log = 'Attribute value {0} is less than 0: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif (int(value) > 1) and (int(value) < self.img_capt_dict['exec_interval']):
                        log = 'Attribute value {0} is between 1 and {1}: {2}.'. \
                            format(attrib, self.img_capt_dict['exec_interval'], value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) % self.img_capt_dict['exec_interval']:
                        log = 'Attribute value {0} is not divisible by execution interval: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) > 1440:
                        log = 'Attribute value {0} is greater than 1440: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            elif attrib == 'prediction_history_freq':
                try:
                    if int(value) < 0:
                        log = 'Attribute value {0} is less than 0: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif (int(value) > 1) and (int(value) < self.img_capt_dict['exec_interval']):
                        log = 'Attribute value {0} is between 1 and {1}: {2}.'.\
                            format(attrib, self.img_capt_dict['exec_interval'], value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) % self.img_capt_dict['exec_interval']:
                        log = 'Attribute value {0} is not divisible by execution interval: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) > 1440:
                        log = 'Attribute value {0} is greater than 1440: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            else:
                valid_option = False

        elif section == 'Image_Transmission':
            if attrib == 'orig':
                attrib = 'original_freq'

            elif attrib == 'scale':
                attrib = 'scale_freq'

            elif attrib == 'screw':
                attrib = 'screws_freq'

            elif attrib == 'grotd':
                attrib = 'grotated_freq'

            elif attrib == 'frotd':
                attrib = 'frotated_freq'

            elif attrib == 'rect':
                attrib = 'rectangled_freq'

            elif attrib == 'digw':
                attrib = 'windowed_freq'

            elif attrib == 'inv':
                attrib = 'inverted_freq'

            elif attrib == 'cont':
                attrib = 'contoured_freq'

            elif attrib == 'digs':
                attrib = 'digits_freq'

            elif attrib == 'pred':
                attrib = 'prediction_freq'

            elif attrib == 'olay':
                attrib = 'overlaid_freq'

            else:
                valid_option = False

            if valid_option:
                try:
                    if int(value) < 0:
                        log = 'Attribute value {0} is less than 0: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif (int(value) > 1) and (int(value) < self.img_capt_dict['exec_interval']):
                        log = 'Attribute value {0} is between 1 and {1}: {2}.'. \
                            format(attrib, self.img_capt_dict['exec_interval'], value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) % self.img_capt_dict['exec_interval']:
                        log = 'Attribute value {0} is not divisible by execution interval: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) > 1440:
                        log = 'Attribute value {0} is greater than 1440: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

        elif section == 'Image_Retention':
            if attrib == 'original':
                pass

            elif attrib == 'scale':
                pass

            elif attrib == 'screw':
                pass

            elif attrib == 'grotated':
                pass

            elif attrib == 'frotated':
                pass

            elif attrib == 'rectangled':
                pass

            elif attrib == 'windowed':
                pass

            elif attrib == 'inverted':
                pass

            elif attrib == 'contoured':
                pass

            elif attrib == 'digits':
                pass

            elif attrib == 'prediction':
                pass

            elif attrib == 'overlaid':
                pass

            else:
                valid_option = False

            if valid_option:
                if not (value == 'True') and not (value == 'False'):
                    log = 'Attribute value {0} is not a boolean: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

        else:
            valid_section = False
            valid_option = False

        if valid_section:
            if valid_option:
                if valid_value:
                    try:
                        self.config.set(
                            section=section,
                            option=attrib,
                            value=value
                        )
                        self.config.write(
                            fp=open(
                                file=self.ini_file,
                                mode='w',
                                encoding='utf-8'
                            ),
                            space_around_delimiters=True
                        )
                        log = 'ConfigParser successfully set and wrote options to file.'
                        logger.info(msg=log)
                    except Exception as exc:
                        log = 'ConfigParser failed to set and write options to file.'
                        logger.error(msg=log)
                        logger.error(msg=exc)
                        print(log)
                        print(exc)

            else:
                set_err = True
                log = 'ConfigParser failed to set and write options to file, invalid option.'
                logger.error(msg=log)

        else:
            set_err = True
            log = 'ConfigParser failed to set and write options to file, invalid section.'
            logger.error(msg=log)

        return set_err, log
