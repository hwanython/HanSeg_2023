from torch import nn


class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch

    def get(self):
        if self.model_name == 'AttentionUnet3D':
            from libs.models.AttentionUnet3D import Attention_UNet3D
            return Attention_UNet3D(n_classes=self.num_classes, in_channels=self.in_ch)
        elif self.model_name == 'VNetLight':
            from libs.models.VNet import VNetLight
            return VNetLight(n_classes=self.num_classes, in_channels=self.in_ch)
        elif self.model_name == 'HighResNet3D':
            from libs.models.HighResNet3D import HighResNet3D
            return HighResNet3D(classes=self.num_classes, in_channels=self.in_ch)
        elif self.model_name == 'Unet3D':
            from libs.models.Unet3D import UNet3D
            return UNet3D(n_classes=self.num_classes, in_channels=self.in_ch)
        elif self.model_name == 'MedNeXt':
            from libs.models.mednextv1.create_mednext_v1 import create_mednext_v1
            return create_mednext_v1(num_input_channels = self.in_ch, num_classes = self.num_classes, model_id = 'B', kernel_size = 3, deep_supervision = True)
        elif self.model_name == 'ER_Net':
            from libs.models.ER_Net import ER_Net
            return ER_Net(classes= self.num_classes, channels=self.in_ch)
        elif self.model_name == 'nonlocalUnet3D':
            from libs.models.nonlocalUnet3D import unet_nonlocal_3D
            return unet_nonlocal_3D(n_classes= self.num_classes, in_channels=self.in_ch)
        else:
            raise ValueError(f'Model {self.model_name} not found')
