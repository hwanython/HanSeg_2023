import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class unet_nonlocal_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=1, is_batchnorm=True,
                 nonlocal_mode='embedded_gaussian', nonlocal_sf=4):
        super(unet_nonlocal_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.nonlocal2 = NONLocalBlock3D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.nonlocal3 = NONLocalBlock3D(in_channels=filters[2], inter_channels=filters[2] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        nl2 = self.nonlocal2(conv2)
        maxpool2 = self.maxpool2(nl2)

        conv3 = self.conv3(maxpool2)
        nl3 = self.nonlocal3(conv3)
        maxpool3 = self.maxpool3(nl3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(nl3,   up4)
        up2 = self.up_concat2(nl2,   up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    

class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
    

def init_weights(net, init_type='kaiming'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        pass
    elif init_type == 'xavier':
        pass
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        pass
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0) 



class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample_factor=4, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation', 'concat_proper', 'concat_proper_down']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, list) else [sub_sample_factor]

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation', 'concat_proper', 'concat_proper_down']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode in ['concatenation']:
                self.wf_phi = nn.Linear(self.inter_channels, 1, bias=False)
                self.wf_theta = nn.Linear(self.inter_channels, 1, bias=False)
            elif mode in ['concat_proper', 'concat_proper_down']:
                self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1,
                                     padding=0, bias=True)

        if mode == 'embedded_gaussian':
            self.operation_function = self._embedded_gaussian
        elif mode == 'dot_product':
            self.operation_function = self._dot_product
        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        elif mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concat_proper':
            self.operation_function = self._concatenation_proper
        elif mode == 'concat_proper_down':
            self.operation_function = self._concatenation_proper_down
        else:
            raise NotImplementedError('Unknown operation function.')

        if any(ss > 1 for ss in self.sub_sample_factor):
            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample_factor))
            if self.phi is None:
                self.phi = max_pool(kernel_size=sub_sample_factor)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample_factor))
            if mode == 'concat_proper_down':
                self.theta = nn.Sequential(self.theta, max_pool(kernel_size=sub_sample_factor))

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample_factor > 1:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, thw/s**2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw/s**2, 0.5c)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        # theta => (b, thw, 0.5c) -> (b, thw, 1) -> (b, 1, thw) -> (expand) (b, thw/s**2, thw)
        # phi => (b, thw/s**2, 0.5c) -> (b, thw/s**2, 1) -> (expand) (b, thw/s**2, thw)
        # f=> RELU[(b, thw/s**2, thw) + (b, thw/s**2, thw)] = (b, thw/s**2, thw)
        f = self.wf_theta(theta_x).permute(0, 2, 1).repeat(1, phi_x.size(1), 1) + \
            self.wf_phi(phi_x).repeat(1, 1, theta_x.size(1))
        f = F.relu(f, inplace=True)

        # Normalise the relations
        N = f.size(-1)
        f_div_c = f / N

        # g(x_j) * f(x_j, x_i)
        # (b, 0.5c, thw/s**2) * (b, thw/s**2, thw) -> (b, 0.5c, thw)
        y = torch.matmul(g_x, f_div_c)
        y = y.contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation_proper(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, thw/s**2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw/s**2)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # theta => (b, 0.5c, thw) -> (expand) (b, 0.5c, thw/s**2, thw)
        # phi => (b, 0.5c, thw/s**2) ->  (expand) (b, 0.5c, thw/s**2, thw)
        # f=> RELU[(b, 0.5c, thw/s**2, thw) + (b, 0.5c, thw/s**2, thw)] = (b, 0.5c, thw/s**2, thw)
        f = theta_x.unsqueeze(dim=2).repeat(1,1,phi_x.size(2),1) + \
            phi_x.unsqueeze(dim=3).repeat(1,1,1,theta_x.size(2))
        f = F.relu(f, inplace=True)

        # psi -> W_psi^t * f -> (b, 1, thw/s**2, thw) -> (b, thw/s**2, thw)
        f = torch.squeeze(self.psi(f), dim=1)

        # Normalise the relations
        f_div_c = F.softmax(f, dim=1)

        # g(x_j) * f(x_j, x_i)
        # (b, 0.5c, thw/s**2) * (b, thw/s**2, thw) -> (b, 0.5c, thw)
        y = torch.matmul(g_x, f_div_c)
        y = y.contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation_proper_down(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, thw/s**2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw/s**2)
        theta_x = self.theta(x)
        downsampled_size = theta_x.size()
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # theta => (b, 0.5c, thw) -> (expand) (b, 0.5c, thw/s**2, thw)
        # phi => (b, 0.5, thw/s**2) ->  (expand) (b, 0.5c, thw/s**2, thw)
        # f=> RELU[(b, 0.5c, thw/s**2, thw) + (b, 0.5c, thw/s**2, thw)] = (b, 0.5c, thw/s**2, thw)
        f = theta_x.unsqueeze(dim=2).repeat(1,1,phi_x.size(2),1) + \
            phi_x.unsqueeze(dim=3).repeat(1,1,1,theta_x.size(2))
        f = F.relu(f, inplace=True)

        # psi -> W_psi^t * f -> (b, 0.5c, thw/s**2, thw) -> (b, 1, thw/s**2, thw) -> (b, thw/s**2, thw)
        f = torch.squeeze(self.psi(f), dim=1)

        # Normalise the relations
        f_div_c = F.softmax(f, dim=1)

        # g(x_j) * f(x_j, x_i)
        # (b, 0.5c, thw/s**2) * (b, thw/s**2, thw) -> (b, 0.5c, thw)
        y = torch.matmul(g_x, f_div_c)
        y = y.contiguous().view(batch_size, self.inter_channels, *downsampled_size[2:])

        # upsample the final featuremaps # (b,0.5c,t/s1,h/s2,w/s3)
        y = F.upsample(y, size=x.size()[2:], mode='trilinear')

        # attention block output
        W_y = self.W(y)
        z = W_y + x

        return z
    

class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample_factor=2, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample_factor=sub_sample_factor,
                                              bn_layer=bn_layer)
        