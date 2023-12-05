import os
import json
import logging
import logging.config
from pathlib import Path
import nrrd
import numpy as np
import torch
import torchio as tio

from torch.utils.data import DataLoader
from datasets.SamplerFactory import SamplerFactory

class HaN(tio.SubjectsDataset):
    """
    MICCAI dataset
    """

    def __init__(self, root, filename, splits, transform=None, sampler=None, **kwargs):
        root = Path(root)
        if not isinstance(splits, list):
            splits = [splits]
        self.sampler = sampler
        subjects_list = self._get_subjects_list(root, filename, splits)
        super().__init__(subjects_list, transform, **kwargs)

    def _numpy_reader(self, path):
        data = torch.from_numpy(np.load(path)).float()
        affine = torch.eye(4, requires_grad=False)
        return data, affine

    def _nrrd_reader(self, path):
        raw_data, _ = nrrd.read(path)
        data = torch.from_numpy(raw_data).float()
        affine = torch.eye(4, requires_grad=False)  # Identity matrix(단위 행렬)
        return data, affine

    def _get_subjects_list(self, root, filename, splits):
        # TODO : check the path
        dense_dir = root / 'labels'
        data_dir = root / 'dicom'
        splits_path = root / filename
        # load splits json file
        # care the json file ','. Do not write the ',' on behind of last item!!

        # TODO: change the method, reading whole list and split train/val/test and kfold
        with open(splits_path) as splits_file:
            json_splits = json.load(splits_file)

        # consists of the data sets
        subjects = []
        for split in splits:
            for patient in json_splits[split]:
                data_path = os.path.join(data_dir, patient + '.nrrd')
                # TODO : check the path
                dense_path = os.path.join(dense_dir, patient + '_dense.seg.nrrd')
                if not os.path.isfile(data_path):
                    raise ValueError(f'Missing data file for patient {patient} ({data_path})')
                if not os.path.isfile(dense_path):
                    raise ValueError(f'Missing dense file for patient {patient} ({dense_path})')

                subject_dict = {
                    'partition': split,
                    'patient': patient,
                    'data': tio.ScalarImage(data_path, reader=self._nrrd_reader),
                    'label': tio.LabelMap(dense_path, reader=self._nrrd_reader),
                }

                subjects.append(tio.Subject(**subject_dict))
            print(f"Loaded {len(subjects)} patients for split {split}")
        return subjects

    def get_loader(self, config):
        #todo
        sampler = SamplerFactory(config).get()
        queue = tio.Queue(
            subjects_dataset=self,
            max_length=100,
            samples_per_volume=10,
            sampler=sampler,
            num_workers=config.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True, 
            start_background=False,
        )
        loader = DataLoader(queue, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
        return loader
