from lib import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x

class xgan_classifier2(nn.Module):
    def __init__(self, latent_len, num_class=2):
        super(xgan_classifier2,self).__init()
        self.classifier = nn.Linear(latent_len, num_class)
    def forward(self, input):
        x = input.view(input.size(0),-1)
        return self.classifier(x)

class xgan_classifier(nn.Module):
    def __init__(self, latent_len=1024,num_class=2):
        super(xgan_classifier,self).__init__()
        self.classifier = nn.Linear(latent_len,num_class)
        self.norm = nn.InstanceNorm1d(latent_len)
    def forward(self,input):
#         x = input.reshape(input.shape[0],1,-1)
        x = input.view(input.size(0),1,-1)
        x = self.norm(x)
#         x = x.reshape(x.shape[0],-1)
        x = x.view(x.size(0),-1)
        output = self.classifier(x)
        return output

class xgan_generator2(nn.Module):
    def __init__(self, in_nc, out_nc, nf=16, nb=4, n_downsampling=4):
        super(xgan_generator2, self).__init__()
        self.mult = 1

        model_conv_s2t = [nn.Conv2d(in_nc, nf, 7, 1, 3),
                 nn.InstanceNorm2d(nf),
                 nn.ReLU(True)]
        # n_downsampling = 2
        for i in range(n_downsampling//4*3):
            model_conv_s2t += [nn.Conv2d(nf*self.mult, nf*self.mult*2, 3, 2, 1),
                      nn.InstanceNorm2d(nf*self.mult*2),
                      nn.ReLU(True)]
            self.mult *= 2
        self.conv_s2t = nn.Sequential(*model_conv_s2t)
        model_conv_t2s = model_conv_s2t.copy()
        self.conv_t2s = nn.Sequential(*model_conv_t2s)

        model_conv_sharing = []
        for i in range(n_downsampling//4):
            model_conv_sharing += [nn.Conv2d(nf*self.mult, nf*self.mult*2, 3, 2, 1),
                      nn.InstanceNorm2d(nf*self.mult*2),
                      nn.ReLU(True)]
            self.mult *= 2
        for i in range(nb//2):
            model_conv_sharing += [resnet_block(nf * self.mult, 3, 1, 1)]
        self.conv_sharing = nn.Sequential(*model_conv_sharing)

        model_deconv_sharing = []
        for i in range(nb//2):
            model_deconv_sharing += [resnet_block(nf * self.mult, 3, 1, 1)]
        for i in range(n_downsampling//4):
            model_deconv_sharing += [nn.ConvTranspose2d(nf * self.mult, nf * self.mult//2, 3, 2, 1),
                                   nn.InstanceNorm2d(nf * self.mult// 2),
                                   nn.ReLU(True)]
            self.mult = self.mult//2
        self.deconv_sharing = nn.Sequential(*model_deconv_sharing)

        model_deconv_s2t = []
        for i in range(n_downsampling//4*3):
            model_deconv_s2t += [nn.ConvTranspose2d(nf * self.mult, nf * self.mult//2, 3, 2, 1),
                                   nn.InstanceNorm2d(nf * self.mult//2),
                                   nn.ReLU(True)]
            self.mult = self.mult//2

        model_deconv_s2t += [nn.Conv2d(nf, out_nc, 7, 1, 3),
                      nn.Tanh()]
        self.deconv_s2t = nn.Sequential(*model_deconv_s2t)
        model_deconv_t2s = model_deconv_s2t.copy()
        self.deconv_t2s = nn.Sequential(*model_deconv_t2s)

        utils.initialize_weights(self)

    def enc_s2t(self, input):
        return self.conv_sharing(self.conv_s2t(input))
    def enc_t2s(self, input):
        return self.conv_sharing(self.conv_t2s(input))
    def dec_s2t(self, input):
        return self.deconv_s2t(self.deconv_sharing(input))
    def dec_t2s(self, input):
        return self.deconv_t2s(self.deconv_sharing(input))
class xgan_generator(nn.Module):
    def __init__(self, in_nc, out_nc, nf=16, fcn=1024):
        super(xgan_generator,self).__init__()
        #input 128*128*3
        self.conv_s2t = nn.Sequential(
                nn.Conv2d(in_nc,nf,3,2,1),
                nn.InstanceNorm2d(nf),
                nn.ReLU(True), #64*64*nf
                nn.Conv2d(nf,2*nf,3,2,1),
                nn.InstanceNorm2d(2*nf),
                nn.ReLU(True), #32*32*2nf
                nn.Conv2d(2*nf,4*nf,3,2,1),
                nn.InstanceNorm2d(4*nf),
                nn.ReLU(True) #16*16*4nf
            )
        self.conv_t2s = nn.Sequential(
                nn.Conv2d(in_nc,nf,3,2,1),
                nn.InstanceNorm2d(nf),
                nn.ReLU(True), #64*64*nf
                nn.Conv2d(nf,2*nf,3,2,1),
                nn.InstanceNorm2d(2*nf),
                nn.ReLU(True), #32*32*2nf
                nn.Conv2d(2*nf,4*nf,3,2,1),
                nn.InstanceNorm2d(4*nf),
                nn.ReLU(True) #16*16*4nf
            )
        self.conv_sharing = nn.Sequential(
                nn.Conv2d(4*nf,8*nf,3,2,1),
                nn.InstanceNorm2d(8*nf),
                nn.ReLU(True), #8*8*8nf
                nn.Conv2d(8*nf,16*nf,3,2,1),
                nn.InstanceNorm2d(16*nf),
                nn.ReLU(True) #4*4*16nf               
            )
        self.fc_sharing = nn.Sequential(
                nn.Linear(4*4*16*nf,1*1*fcn), 
                nn.Linear(1*1*fcn,1*1*fcn)
            )
        self.deconv_sharing = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(fcn,32*nf,3,1,1),
                nn.InstanceNorm2d(32*nf),
                nn.ReLU(True), #4*4*32nf
                nn.ConvTranspose2d(32*nf,16*nf,4,2,1),
                nn.InstanceNorm2d(16*nf),
                nn.ReLU(True) #8*8*16nf                
            )
        self.deconv_s2t = nn.Sequential(
                nn.ConvTranspose2d(16*nf,8*nf,4,2,1),
                nn.InstanceNorm2d(8*nf),
                nn.ReLU(True), #16*16*8nf
                nn.ConvTranspose2d(8*nf,4*nf,4,2,1),
                nn.InstanceNorm2d(4*nf),
                nn.ReLU(True), #32*32*4nf   
                nn.ConvTranspose2d(4*nf,2*nf,4,2,1),
                nn.InstanceNorm2d(2*nf),
                nn.ReLU(True), #64*64*2nf   
                nn.ConvTranspose2d(2*nf,nf,4,2,1),#128*128*nf
                nn.Conv2d(nf,out_nc,3,1,1), #128*128*out_nc
                nn.Tanh() 
            )
        self.deconv_t2s = nn.Sequential(
                nn.ConvTranspose2d(16*nf,8*nf,4,2,1),
                nn.InstanceNorm2d(8*nf),
                nn.ReLU(True), #16*16*8nf
                nn.ConvTranspose2d(8*nf,4*nf,4,2,1),
                nn.InstanceNorm2d(4*nf),
                nn.ReLU(True), #32*32*4nf   
                nn.ConvTranspose2d(4*nf,2*nf,4,2,1),
                nn.InstanceNorm2d(2*nf),
                nn.ReLU(True), #64*64*2nf   
                nn.ConvTranspose2d(2*nf,nf,4,2,1),#128*128*nf
                nn.Conv2d(nf,out_nc,3,1,1), #128*128*out_nc
                nn.Tanh() 
            )
        utils.initialize_weights(self)

    def enc_s2t(self,input):
        x = self.conv_sharing(self.conv_s2t(input))
#         x = x.reshape(x.shape[0],-1)
        x = x.view(x.size(0),-1)

        out = self.fc_sharing(x)
        return out

    def enc_t2s(self,input):
        x = self.conv_sharing(self.conv_t2s(input))
#         x = x.reshape(x.shape[0],-1)
        x = x.view(x.size(0),-1)

        out = self.fc_sharing(x)
        return out        

    def dec_s2t(self,input):
#         x = input.reshape(input.shape[0],-1,1,1)
        x = input.view(input.size(0),-1,1,1)        
        x = self.deconv_sharing(x)
        out = self.deconv_s2t(x)
        return out

    def dec_t2s(self,input):
#         x = input.reshape(input.shape[0],-1,1,1)
        x = input.view(input.size(0),-1,1,1)
        x = self.deconv_sharing(x)
        out = self.deconv_t2s(x)
        return out

class generator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(in_nc, nf, 7, 1, 3),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1),
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3),
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        # input = torch.cat((input1, input2), 1)
        output = self.convs(input)

        return output


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x
