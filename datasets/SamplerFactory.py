import torchio as tio

class SamplerFactory:
    def __init__(self, config):
        self.config = config
        # the config is not whole config, it is "self.config.data_loader"
        self.sampler_type = config.sampler_type

    def get(self):
        if self.sampler_type == 'UniformSampler':
             sampler = tio.UniformSampler(patch_size=self.config.patch_shape)
        # elif self.sampler_type == 'WeightedSampler':
        #     sampler = tio.WeightedSampler(patch_size=self.config.patch_shape, probability_map='sampling_map')
        elif self.sampler_type == 'WeightedSampler':
            probabilities = {0: 0.1, 1: 0.9}
            sampler = tio.LabelSampler(patch_size=self.config.patch_shape, label_name='label', label_probabilities=probabilities)

        return sampler