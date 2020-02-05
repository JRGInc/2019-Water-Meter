__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import os

logfile = 'januswm'
logger = logging.getLogger(logfile)


class CoreCfg(object):
    """
    Class attributes of configuration settings for capture operations
    """

    def __init__(
            self
    ) -> None:
        """
        Sets object properties directly
        """
        self.base_dir = os.path.dirname('/opt/Janus/WM/')

        self.train_version = '019'
        self.test_version = '001'
        self.test_set = '005'

        # Define core paths
        core_dirs_dict = {
            'arch': 'archives/',
            'cache': 'cache/',
            'cfg': 'config/',
            'data': 'data/',
            'logs': 'logs/',
            'mdl': 'model/',
            'py3': 'python3/',
            'wgts': 'weights/'
        }

        # Define core paths
        self.core_path_dict = {
            'arch': os.path.join(
                self.base_dir,
                core_dirs_dict['arch']
            ),
            'train_cache': os.path.join(
                self.base_dir,
                core_dirs_dict['cache'],
                'train/'
            ),
            'valid_cache': os.path.join(
                self.base_dir,
                core_dirs_dict['cache'],
                'valid/'
            ),
            'cfg': os.path.join(
                self.base_dir,
                core_dirs_dict['cfg']
            ),
            'data': os.path.join(
                self.base_dir,
                core_dirs_dict['data']
            ),
            'logs': os.path.join(
                self.base_dir,
                core_dirs_dict['logs']
            ),
            'mdl': os.path.join(
                self.base_dir,
                core_dirs_dict['mdl']
            ),
            'py3': os.path.join(
                self.base_dir,
                core_dirs_dict['py3']
            ),
            'wgts': os.path.join(
                self.base_dir,
                core_dirs_dict['wgts']
            )
        }

        # Define file names in /config path
        cfg_name_dict = {
            'capt': 'capture.ini',
            'err': 'errors.txt',
            'hist': 'prediction_history.txt',
            'last': 'prediction_latest.txt',
            'seq': 'sequence.txt',
            'tf': 'tensor.ini',
            'xmit': 'transmit.ini',
        }

        # Define full urls for files in /config path
        self.cfg_url_dict = {
            'capt': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['capt']
            ),
            'err': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['err']
            ),
            'hist': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['hist']
            ),
            'last': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['last']
            ),
            'seq': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['seq']
            ),
            'tf': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['tf']
            ),
            'xmit': os.path.join(
                self.core_path_dict['cfg'],
                cfg_name_dict['xmit']
            )
        }

        # Define data paths
        data_dirs_dict = {
            'hist': 'history/',
            'img': 'images/',
            'last': 'latest/',
            'rslt': 'results/',
            'xmit': 'transmit/'
        }

        # Define full urls for files in /data path
        self.data_path_dict = {
            'hist': os.path.join(
                self.core_path_dict['data'],
                data_dirs_dict['hist']
            ),
            'img': os.path.join(
                self.core_path_dict['data'],
                data_dirs_dict['img']
            ),
            'last': os.path.join(
                self.core_path_dict['data'],
                data_dirs_dict['last']
            ),
            'rslt': os.path.join(
                self.core_path_dict['data'],
                data_dirs_dict['rslt']
            ),
            'xmit': os.path.join(
                self.core_path_dict['data'],
                data_dirs_dict['xmit']
            )
        }

        # Define image paths
        img_dirs_dict = {
            'mov': '00--movies/',
            'orig': '01--original/',
            'scale': '02--scaled/',
            'screw': '03--screws/',
            'grotd': '04--grotated/',
            'frotd': '05--frotated/',
            'rect': '06--rectangled/',
            'digw': '07--windowed/',
            'inv': '08--inverted',
            'cont': '09--contoured/',
            'digs': '10--digits/',
            'pred': '11--prediction/',
            'olay': '12--overlaid/',
            'trn_sift': '13--train_sifted/',
            'trn_sel': '14--train_selected/',
            'trn': '15--train/',
            'vld': '16--valid/',
            'test_sift': '17--test_sifted/',
            'test_sel': '18--test_selected/',
            'test': '19--test/',
            'test_err': '20--test_errors/'
        }

        # Define full urls for files in /images path
        self.img_path_dict = {
            'mov': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['mov']
            ),
            'orig': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['orig']
            ),
            'scale': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['scale']
            ),
            'screw': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['screw']
            ),
            'grotd': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['grotd']
            ),
            'frotd': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['frotd']
            ),
            'rect': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['rect']
            ),
            'digw': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['digw']
            ),
            'inv': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['inv']
            ),
            'cont': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['cont']
            ),
            'digs': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['digs']
            ),
            'pred': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['pred']
            ),
            'olay': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['olay']
            ),
            'trn_sift': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['trn_sift']
            ),
            'trn_sel': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['trn_sel']
            ),
            'trn': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['trn']
            ),
            'vld': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['vld']
            ),
            'test_sift': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['test_sift']
            ),
            'test_sel': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['test_sel']
            ),
            'test': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['test']
            ),
            'test_err': os.path.join(
                self.data_path_dict['img'],
                img_dirs_dict['test_err']
            )
        }

        # How many train images to use for validation
        self.valid_ratio = 0.25

        # Specify which batch process must be executed
        # True = enable, False = disabled
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

    def get(
        self,
        attrib: str
    ) -> any:
        """
        Gets configuration attributes

        :param attrib: str

        :return: any
        """
        if attrib == 'cptr_base_dir':
            return self.base_dir
        elif attrib == 'train_version':
            return self.train_version
        elif attrib == 'test_version':
            return self.test_version
        elif attrib == 'test_set':
            return self.test_set
        elif attrib == 'cfg_url_dict':
            return self.cfg_url_dict
        elif attrib == 'core_path_dict':
            return self.core_path_dict
        elif attrib == 'data_path_dict':
            return self.data_path_dict
        elif attrib == 'img_path_dict':
            return self.img_path_dict
        elif attrib == 'batch_proc_en_dict':
            return self.batch_proc_en_dict
        elif attrib == 'valid_ratio':
            return self.valid_ratio
