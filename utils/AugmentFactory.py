import logging
import torchio as tio


class AugFactory:
    def __init__(self, aug_list):
        self.aug_list = aug_list
        self.transforms = self.factory(self.aug_list, [])
        logging.info('Augmentations: {}'.format(self.aug_list))

    def factory(self, auglist, transforms):
        if auglist == None: return []
        for aug in auglist: 
            if aug == 'OneOf':
                transforms.append(tio.OneOf(self.factory(auglist[aug], [])))
            else:
                try:
                    kwargs = {}
                    for param, value in auglist[aug].items():
                        kwargs[param] = value
                    else:
                        t = getattr(tio, aug)(**kwargs)
                    transforms.append(t)
                except:
                    raise Exception(f"this transform is not valid: {aug}")
        return transforms

    def get_transform(self):
        """
        return the transform object
        :return:
        """
        transf = tio.Compose(self.transforms)
        return transf