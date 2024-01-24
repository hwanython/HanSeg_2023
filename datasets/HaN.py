import os
import json
import logging
import logging.config
from pathlib import Path
import nrrd
import numpy as np
import torch
import torchio as tio
from datasets.label_dict import LABEL_dict, Anchor_dict   # from datasets/label_dict.py
from torch.utils.data import DataLoader
from datasets.SamplerFactory import SamplerFactory

class HaN(tio.SubjectsDataset):
    """
    MICCAI dataset
    """
    def __init__(self, config, splits, transform=None, sampler=None, **kwargs):
        self.config = config
        self.splits = splits
        self.root = Path(self.config.data_loader.dataset)
        if not isinstance(splits, list):
            splits = [splits]
        self.seed = self.config.seed
        self.sampler = sampler
        subjects_list = self._get_subjects_list(self.root, splits)
        super().__init__(subjects_list, transform, **kwargs)

    def _numpy_reader(self, path):
        data = torch.from_numpy(np.load(path)).float()
        affine = torch.eye(4, requires_grad=False)
        return data, affine
    
    def _split_data(self, data_list):
        # train and val data split
        np.random.seed(self.seed)
        split_ratio = 0.8
        train_size = int(split_ratio * len(data_list))
        val_size = int((len(data_list) - train_size))
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size+val_size]
        return train_data, val_data
    
    def _generate_jsondata(self, train_data: list, val_data: list, test_data=None):
        if test_data:
            test_data = test_data
        else:
            test_data = val_data
            
        json_data = {
                'train': train_data,
                'val': val_data,
                 "test": test_data
            }
        return json_data

    def _get_subjects_list(self, root, splits):
        # TODO : check the path
        patient_data_list = os.listdir(root)
        patient_data_list = [entry for entry in patient_data_list if os.path.isdir(os.path.join(root, entry))]

        # TODO: change the method, reading whole list and split train/val/test and kfold       
        if self.config.data_loader.kfold == 1:
            train_data, val_data = self._split_data(patient_data_list)
            json_splits = self._generate_jsondata(train_data, val_data)

        # consists of the data sets
        subjects = []
        for split in splits:
            for patient in json_splits[split]:
                # generate labels
                ct_data_path = os.path.join(root, patient, patient + '_IMG_CT.nrrd')
                # mr_data_path = os.path.join(root, patient, patient + '_IMG_MR_T1.nrrd')
                label_path = os.path.join(root, patient, patient + f'_{self.config.experiment.name}.seg.nrrd')
                if not os.path.isfile(ct_data_path):
                    raise ValueError(f'Missing CT data file for patient {patient} ({ct_data_path})')
                # if not os.path.isfile(mr_data_path):
                    # raise ValueError(f'Missing MR_TI data file for patient {patient} ({mr_data_path})')
                if not os.path.isfile(label_path):
                    raise ValueError(f'Missing LABEL file for patient {patient} ({label_path})')

                subject_dict = {
                    'partition': split,
                    'patient': patient,
                    'ct': tio.ScalarImage(ct_data_path),
                    # 'mr': tio.ScalarImage(mr_data_path, reader=self._nrrd_reader),
                    'label': tio.LabelMap(label_path,),
                }

                subjects.append(tio.Subject(**subject_dict))
            print(f"Loaded {len(subjects)} patients for split {split}")
        return subjects

    def get_loader(self, config):
        # patch-based training
        if config.patch_loader:
            sampler = SamplerFactory(config).get()
            queue = tio.Queue(
                subjects_dataset=self,
                max_length=300,
                samples_per_volume=10,
                sampler=sampler,
                num_workers=config.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True, 
                start_background=False,
            )
            loader =  DataLoader(queue, batch_size=config.batch_size, num_workers=0, pin_memory=True)
        else: # subject-based training
            dataset = tio.SubjectsDataset(self._subjects, transform=self._transform)
            loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
        return loader
