__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import configparser
import logging

logfile = 'januswm'
logger = logging.getLogger(logfile)


class TensorCfg(object):
    """
    Class attributes of configuration settings for TensorFlow operations
    """
    def __init__(
        self,
        core_cfg: any
    ) -> None:
        """
        Sets object properties either directly or via ConfigParser from ini file

        :param core_cfg: any
        """
        cfg_url_dict = core_cfg.get(attrib='cfg_url_dict')
        self.ini_file = cfg_url_dict['tf']
        self.config = configparser.ConfigParser()
        self.config.read_file(f=open(self.ini_file))

        # Dimensions based on already cropped image of all 6 digits
        # Digits are listed as least significant = digit 0
        # Only upper left x location given, the other points
        # are determined programmatically
        self.tf_dict = {
            'full_width': self.config.getint(
                'TensorFlow',
                'full_width'
            ),
            'tf_width': self.config.getint(
                'TensorFlow',
                'tf_width'
            ),
            'shadow': self.config.getint(
                'TensorFlow',
                'shadow'
            ),
            'height': self.config.getint(
                'TensorFlow',
                'height'
            ),
            'shift_en': self.config.getboolean(
                'TensorFlow',
                'shift_en'
            ),
            'shift': self.config.getint(
                'TensorFlow',
                'shift'
            ),
            'batch_size': self.config.getint(
                'TensorFlow',
                'batch_size'
            ),
            'img_tgt_width': self.config.getint(
                'TensorFlow',
                'img_tgt_width'
            ),
            'img_tgt_height': self.config.getint(
                'TensorFlow',
                'img_tgt_height'
            ),
            'nbr_classes': self.config.getint(
                'TensorFlow',
                'nbr_classes'
            ),
            'nbr_channels': self.config.getint(
                'TensorFlow',
                'nbr_channels'
            ),
            'patience': self.config.getint(
                'TensorFlow',
                'patience'
            ),
            'epochs': self.config.getint(
                'TensorFlow',
                'epochs'
            ),
            'format': self.config.get(
                'TensorFlow',
                'format'
            )
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
        if attrib == 'tf_dict':
            return self.tf_dict

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

        if section == 'TensorFlow':
            opt_type = 'int'
            if attrib == 'full_width':
                pass

            elif attrib == 'tf_width':
                pass

            elif attrib == 'shadow':
                pass

            elif attrib == 'height':
                pass

            elif attrib == 'shift_en':
                opt_type = 'bool'
                pass

            elif attrib == 'shift':
                pass

            elif attrib == 'batch_size':
                pass

            elif attrib == 'img_tgt_width':
                pass

            elif attrib == 'img_tgt_height':
                pass

            elif attrib == 'nbr_classes':
                pass

            elif attrib == 'nbr_channels':
                pass

            elif attrib == 'patience':
                pass

            elif attrib == 'epochs':
                pass

            elif attrib == 'format':
                opt_type = 'str'

            else:
                valid_option = False

            if valid_option and (opt_type == 'int'):
                try:
                    if int(value) < 1:
                        log = 'Attribute value {0} is less than 1: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                    elif int(value) > 100:
                        log = 'Attribute value {0} is greater than 100: {1}.'.format(attrib, value)
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

            if valid_option and (opt_type == 'bool'):
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
