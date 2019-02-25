import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False):
        super(conv_block, self).__init__()
        self.upsample = upsample
        
        if (upsample):
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d((kernel_size-1)//2)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        
        if (self.upsample):
            out = torch.nn.functional.interpolate(out, scale_factor=2)

        out = self.reflection(out)
        out = self.conv(out)
            
        return out
    
class network_optical_flow(nn.Module):
    def __init__(self, n_channels=32):
        super(network_optical_flow, self).__init__()
        self.A01 = conv_block(2, n_channels)
        
        self.C01 = conv_block(n_channels, 2*n_channels, stride=2)
        self.C02 = conv_block(2*n_channels, 2*n_channels)
        self.C03 = conv_block(2*n_channels, 2*n_channels)
        self.C04 = conv_block(2*n_channels, 2*n_channels)
        
        self.C11 = conv_block(2*n_channels, 2*n_channels)
        self.C12 = conv_block(2*n_channels, 2*n_channels)
        self.C13 = conv_block(2*n_channels, 2*n_channels)
        self.C14 = conv_block(2*n_channels, 2*n_channels)
        
        self.C21 = conv_block(2*n_channels, 4*n_channels, stride=2)
        self.C22 = conv_block(4*n_channels, 4*n_channels)
        self.C23 = conv_block(4*n_channels, 4*n_channels)
        self.C24 = conv_block(4*n_channels, 4*n_channels)
        
        self.C31 = conv_block(4*n_channels, 8*n_channels, stride=2)
        self.C32 = conv_block(8*n_channels, 8*n_channels)
        self.C33 = conv_block(8*n_channels, 8*n_channels)
        self.C34 = conv_block(8*n_channels, 8*n_channels)
        
        self.C41 = conv_block(8*n_channels, 4*n_channels, stride=2, upsample=True)
        self.C42 = conv_block(4*n_channels, 4*n_channels)
        self.C43 = conv_block(4*n_channels, 4*n_channels)
        self.C44 = conv_block(4*n_channels, 4*n_channels)
        
        self.C51 = conv_block(4*n_channels, 2*n_channels, stride=2, upsample=True)
        self.C52 = conv_block(2*n_channels, 2*n_channels)
        self.C53 = conv_block(2*n_channels, 2*n_channels)
        self.C54 = conv_block(2*n_channels, 2*n_channels)

        self.C61 = conv_block(2*n_channels, n_channels, stride=2, upsample=True)
        self.C62 = conv_block(n_channels, n_channels)
        self.C63 = conv_block(n_channels, n_channels)        
        
        self.C64 = nn.Conv2d(n_channels, 2, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.C64.weight)
        nn.init.constant_(self.C64.bias, 0.1)

        self.tanh = nn.Tanh()
                
    def forward(self, x):
        A01 = self.A01(x)

        # N -> N/2
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)        
        C04 = self.C04(C03)
        C04 += C01
        
        # N/2 -> N/2
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)        
        C14 = self.C14(C13)
        C14 += C11
        
        # N/2 -> N/4
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)        
        C24 = self.C24(C23)
        C24 += C21
        
        # N/4 -> N/8
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)        
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        C41 += C24
        C42 = self.C42(C41)
        C43 = self.C43(C42)
        C44 = self.C44(C43)
        C44 += C41

        C51 = self.C51(C44)
        C51 += C14
        C52 = self.C52(C51)
        C53 = self.C53(C52)
        C54 = self.C54(C53)
        C54 += C51
        
        C61 = self.C61(C54)        
        C62 = self.C62(C61)
        C63 = self.C63(C62)
        out = self.C64(C63)      

        return self.tanh(out)
    
class network_spatial_transformer(nn.Module):
    def __init__(self, n_pixel, device):
        super(network_spatial_transformer, self).__init__()

        self.device = device
        self.n_pixel = n_pixel

        x = torch.linspace(-1,1,n_pixel)
        y = torch.linspace(-1,1,n_pixel)
        X, Y = torch.meshgrid([x, y])

        self.reference = torch.zeros((n_pixel, n_pixel, 2))
        self.reference[:,:,0] = Y
        self.reference[:,:,1] = X

        self.reference = self.reference.to(self.device)
        
    def forward(self, x, flow):
        flow_reshape = flow.transpose(1,2).transpose(2,3)

        out = torch.nn.functional.grid_sample(x, flow_reshape + self.reference[None,:,:,:])
        
        return out
    
class network(nn.Module):
    def __init__(self, n_pixel, device):
        super(network, self).__init__()
        self.n_pixel = n_pixel
        self.optical_flow = network_optical_flow(n_channels=8)
        self.deformation = network_spatial_transformer(n_pixel, device)
        
    def forward(self, x, backward=True):
        
        
        flow_forward = self.optical_flow(x)
        out_forward = self.deformation(x[:,0:1,:,:], flow_forward * 64.0 / self.n_pixel)

        if (backward):
            x_flip = torch.flip(x, [1])
            flow_backward = self.optical_flow(x_flip)
            out_backward = self.deformation(x[:,1:2,:,:], flow_backward * 64.0 / self.n_pixel)
        
            return out_forward, out_backward, flow_forward, flow_backward

        else:
            return out_forward, flow_forward