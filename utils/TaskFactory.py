from utils.TrainFactory import *

class TaskFactory:
    def __init__(self, config, debug=False):
        self.name = config.experiment.name
        self.config = config
        self.debug = debug

    def get(self):
        if self.name == 'Segmentation':
            experiment = Segmentation(self.config, self.debug)
        elif self.name == 'Generation':
            experiment = Generation(self.config, self.debug)
        else:
            raise ValueError(f'Experiment \'{self.name}\' not found')
        return experiment

class Segmentation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)

class Generation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)