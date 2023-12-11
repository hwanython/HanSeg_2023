import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
from hashlib import shake_256
import time
from munch import Munch, munchify, unmunchify
from utils.AugmentFactory import *
from utils.TaskFactory import *
import pandas as pd

def _nrrd_reader(path):
    raw_data, _ = nrrd.read(path)
    data = torch.from_numpy(raw_data).float()
    affine = torch.eye(4, requires_grad=False)  # Identity matrix(단위 행렬)
    return data, affine

def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5) # output len: 2*5=10
    return h.upper()

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="configs/preprocessing.yaml",      
                            help="the preprocessing config file to be used to run the experiment")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    args = arg_parser.parse_args()


  

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit
    
    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    preproc = yaml.load(open(args.config, "r"), yaml.FullLoader)
    preproc = munchify(preproc)


    source_dir = preproc.source_dir
    save_dir = preproc.save_dir
    # set the title name with timehash
    title = f'preprocessing_{timehash()}'
    save_dir= os.path.join(save_dir, title)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f'source_dir: {source_dir}')
    logging.info(f'save_dir: {save_dir}')

    preproc_processing = AugFactory(preproc.preprocessing).get_transform()
    copy_preprocessing_path = os.path.join(save_dir, 'preprocessing.yaml')
    if args.config is not None:
        shutil.copy(args.config, copy_preprocessing_path)

    transform=preproc_processing
    
    logging.info(f'experiment title: {preproc.experiment.name}')

    # main
    patient_data_list = os.listdir(source_dir)
    subjects = []
    for patient in patient_data_list:
        # generate labels
        ct_data_path = os.path.join(source_dir, patient, patient + '_IMG_CT.nrrd')
        # mr_data_path = os.path.join(root, patient, patient + '_IMG_MR_T1.nrrd')
        label_path = os.path.join(source_dir, patient, patient + f'_{preproc.experiment.name}.seg.nrrd')
        if not os.path.isfile(ct_data_path):
            raise ValueError(f'Missing CT data file for patient {patient} ({ct_data_path})')
        # if not os.path.isfile(mr_data_path):
            # raise ValueError(f'Missing MR_TI data file for patient {patient} ({mr_data_path})')
        if not os.path.isfile(label_path):
            raise ValueError(f'Missing LABEL file for patient {patient} ({label_path})')

        subject_dict = {

            'patient': patient,
            'ct': tio.ScalarImage(ct_data_path, reader=_nrrd_reader),
            # 'mr': tio.ScalarImage(mr_data_path, reader=self._nrrd_reader),
            'label': tio.LabelMap(label_path, reader=_nrrd_reader),
        }
        
        # preprocessing
        subject = tio.Subject(**subject_dict)
        transform_subject = transform(subject)
        # save
        os.makedirs(os.path.join(save_dir, patient), exist_ok=True)
        ct_data_path = os.path.join(save_dir, patient, patient + '_IMG_CT.nrrd')
        label_path = os.path.join(save_dir, patient, patient + f'_{preproc.experiment.name}.seg.nrrd')
        nrrd.write(ct_data_path, transform_subject['ct'][tio.DATA].squeeze(0).numpy())
        nrrd.write(label_path, transform_subject['label'][tio.DATA].squeeze(0).numpy())
        print(f"Saved {patient} patients")
        subjects.append(tio.Subject(**subject_dict))

    print(f"Completed {len(subjects)} patients")
        
    