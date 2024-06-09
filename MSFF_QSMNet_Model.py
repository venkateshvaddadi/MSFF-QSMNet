import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src
#%%
#making it for 3D
class REBNCONV_for_3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV_for_3D, self).__init__()

        self.conv_s1 = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


def _upsample_like_for_3D(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=False)
    return src


#%%



### RSU-7 ###
class RSU7_for_3D(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=32):
        super(RSU7_for_3D,self).__init__()

        self.rebnconvin = REBNCONV_for_3D(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV_for_3D(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV_for_3D(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        #####print('hx.shape',hx.shape)
        hxin = self.rebnconvin(hx)
        #####print('hxin.shape',hxin.shape)

        hx1 = self.rebnconv1(hxin)
        #####print('hx1.shape',hx1.shape)

        hx = self.pool1(hx1)
        #####print('hx.shape',hx.shape)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        #####print('hx2.shape',hx2.shape)
        #####print('hx.shape',hx.shape)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        #####print('hx3.shape',hx3.shape)
        #####print('hx.shape',hx.shape)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        #####print('hx4.shape',hx4.shape)
        #####print('hx.shape',hx.shape)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        #####print('hx5.shape',hx5.shape)
        #####print('hx.shape',hx.shape)

        hx6 = self.rebnconv6(hx)
        #####print('hx6.shape',hx6.shape)

        hx7 = self.rebnconv7(hx6)
        #####print('hx7.shape',hx7.shape)


        # decoder
        #####print('Decoding started...')
        
        #####print('concatnated dimension of torch.cat((hx7,hx6),1)',torch.cat((hx7,hx6),1).shape)
        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        #####print('hx6d.shape',hx6d.shape)

        hx6dup = _upsample_like_for_3D(hx6d,hx5)
        #####print('hx6dup.shape',hx6dup.shape)
        #####print('concatnated dimension of torch.cat((hx6dup,hx5),1)',torch.cat((hx6dup,hx5),1).shape)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        #####print('hx5d.shape',hx5d.shape)

        hx5dup = _upsample_like_for_3D(hx5d,hx4)
        #####print('hx5dup.shape',hx5dup.shape)

        #####print('concatnated dimension of torch.cat((hx5dup,hx4),1)',torch.cat((hx5dup,hx4),1).shape)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        #####print('hx4d.shape',hx4d.shape)

        hx4dup = _upsample_like_for_3D(hx4d,hx3)
        #####print('hx4dup.shape',hx4dup.shape)


        #####print('concatnated dimension of torch.cat((hx4dup,hx3),1)',torch.cat((hx4dup,hx3),1).shape)


        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        #####print('hx3d.shape',hx3d.shape)

        hx3dup = _upsample_like_for_3D(hx3d,hx2)
        #####print('hx3dup.shape',hx3dup.shape)

        #####print('concatnated dimension of torch.cat((hx3dup,hx2),1)',torch.cat((hx3dup,hx2),1).shape)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        #####print('hx2d.shape',hx2d.shape)

        hx2dup = _upsample_like_for_3D(hx2d,hx1)
        #####print('hx2dup.shape',hx2dup.shape)

        #####print('concatnated dimension of torch.cat((hx2dup,hx1),1)',torch.cat((hx2dup,hx1),1).shape)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        #####print('hx1d.shape',hx1d.shape)

        return hx1d + hxin

# model = RSU7_for_3D(in_ch=1, mid_ch=16, out_ch=64).float().cuda()

# temp_input=torch.randn(1,1,64,64,64).float().cuda()
# #####print('temp_input.shape: ',temp_input.shape)

# temp_output=model(temp_input)
# #####print('temp_output.shape: ',temp_output.shape)


# torch.save(model.state_dict(), 'RSU7_for_3D.pth')

#%%






### RSU-6 ###
class RSU6_for_3D(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6_for_3D,self).__init__()

        self.rebnconvin = REBNCONV_for_3D(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV_for_3D(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV_for_3D(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like_for_3D(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like_for_3D(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like_for_3D(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like_for_3D(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


# model = RSU6_for_3D(in_ch=64, mid_ch=16, out_ch=64).float().cuda()

# temp_input=torch.randn(1,64,64,64,64).float().cuda()
# #####print('temp_input.shape: ',temp_input.shape)

# temp_output=model(temp_input)
# #####print('temp_output.shape: ',temp_output.shape)


# torch.save(model.state_dict(), 'RSU6_for_3D.pth')
#%%
### RSU-5 ###
class RSU5_for_3D(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5_for_3D,self).__init__()

        self.rebnconvin = REBNCONV_for_3D(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV_for_3D(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV_for_3D(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like_for_3D(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like_for_3D(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like_for_3D(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin



# model = RSU5_for_3D(in_ch=64, mid_ch=16, out_ch=64).float().cuda()

# temp_input=torch.randn(1,64,64,64,64).float().cuda()
# #####print('temp_input.shape: ',temp_input.shape)

# temp_output=model(temp_input)
# #####print('temp_output.shape: ',temp_output.shape)


# torch.save(model.state_dict(), 'RSU5_for_3D.pth')
#%%


### RSU-4 ###
class RSU4_for_3D(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4_for_3D,self).__init__()

        self.rebnconvin = REBNCONV_for_3D(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV_for_3D(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV_for_3D(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like_for_3D(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like_for_3D(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


model = RSU4_for_3D(in_ch=64, mid_ch=16, out_ch=64).float().cuda()

temp_input=torch.randn(1,64,64,64,64).float().cuda()
#####print('temp_input.shape: ',temp_input.shape)

temp_output=model(temp_input)
#####print('temp_output.shape: ',temp_output.shape)


torch.save(model.state_dict(), 'RSU4_for_3D.pth')
#%%

### RSU-4F ###
class RSU4F_for_3D(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F_for_3D,self).__init__()

        self.rebnconvin = REBNCONV_for_3D(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV_for_3D(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV_for_3D(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV_for_3D(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV_for_3D(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


# model = RSU4F_for_3D(in_ch=64, mid_ch=16, out_ch=64).float().cuda()

# temp_input=torch.randn(1,64,64,64,64).float().cuda()
# #####print('temp_input.shape: ',temp_input.shape)

# temp_output=model(temp_input)
# #####print('temp_output.shape: ',temp_output.shape)


# torch.save(model.state_dict(), 'RSU4F_for_3D.pth')
#%%


### 3D U2NET implementation with more customization###
class U2NETP_for_3D_cus_ch(nn.Module):

    def __init__(self,in_ch=1,out_ch=1,cus_ch=64,mid_ch=16):
        super(U2NETP_for_3D_cus_ch,self).__init__()

        self.stage1 = RSU7_for_3D(in_ch,mid_ch,cus_ch)
        self.pool12 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6_for_3D(cus_ch,mid_ch,cus_ch)
        self.pool23 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5_for_3D(cus_ch,mid_ch,cus_ch)
        self.pool34 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4_for_3D(cus_ch,mid_ch,cus_ch)
        self.pool45 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F_for_3D(cus_ch,mid_ch,cus_ch)
        self.pool56 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F_for_3D(cus_ch,mid_ch,cus_ch)

        # decoder
        self.stage5d = RSU4F_for_3D(2*cus_ch,mid_ch,cus_ch)
        self.stage4d = RSU4_for_3D(2*cus_ch,mid_ch,cus_ch)
        self.stage3d = RSU5_for_3D(2*cus_ch,mid_ch,cus_ch)
        self.stage2d = RSU6_for_3D(2*cus_ch,mid_ch,cus_ch)
        self.stage1d = RSU7_for_3D(2*cus_ch,mid_ch,cus_ch)

        self.side1 = nn.Conv3d(cus_ch,out_ch,3,padding=1)
        self.side2 = nn.Conv3d(cus_ch,out_ch,3,padding=1)
        self.side3 = nn.Conv3d(cus_ch,out_ch,3,padding=1)
        self.side4 = nn.Conv3d(cus_ch,out_ch,3,padding=1)
        self.side5 = nn.Conv3d(cus_ch,out_ch,3,padding=1)
        self.side6 = nn.Conv3d(cus_ch,out_ch,3,padding=1)

        self.outconv = nn.Conv3d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like_for_3D(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like_for_3D(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like_for_3D(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like_for_3D(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like_for_3D(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like_for_3D(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like_for_3D(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like_for_3D(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like_for_3D(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like_for_3D(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))


        return d0,d1,d2,d3,d4,d5,d6

#%%
# proposed MSFF_QSMNet model
class MSFF_QSMNet(nn.Module):

    def __init__(self,in_ch=1,out_ch=1):
        super(MSFF_QSMNet,self).__init__()

        self.stage1 = RSU7_for_3D(in_ch,16,64)
        self.pool12 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6_for_3D(64,16,64)
        self.pool23 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5_for_3D(64,16,64)
        self.pool34 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4_for_3D(64,16,64)
        self.pool45 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F_for_3D(64,16,64)
        self.pool56 = nn.MaxPool3d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F_for_3D(64,16,64)

        # decoder
        self.stage5d = RSU4F_for_3D(128,16,64)
        self.stage4d = RSU4_for_3D(128,16,64)
        self.stage3d = RSU5_for_3D(128,16,64)
        self.stage2d = RSU6_for_3D(128,16,64)
        self.stage1d = RSU7_for_3D(128,16,64)

        self.side1 = nn.Conv3d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv3d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv3d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv3d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv3d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv3d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv3d(7*out_ch,out_ch,1)

        self.final_stage_of_half_UNet = RSU7_for_3D(64,16,64)
        self.outconv_Half_of_UNet = nn.Conv3d(64,out_ch,1)
        self.decision=nn.Sigmoid()


    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        #print('hx1.shape:',hx1.shape)
        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        #print('hx2.shape:',hx2.shape)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        #print('hx3.shape:',hx3.shape)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        #print('hx4.shape:',hx4.shape)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        #print('hx5.shape:',hx5.shape)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like_for_3D(hx6,hx5)
        #print('hx6.shape:',hx6.shape)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like_for_3D(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like_for_3D(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like_for_3D(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like_for_3D(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like_for_3D(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like_for_3D(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like_for_3D(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like_for_3D(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like_for_3D(d6,d1)
        
        #halfunet style
        hx2_upsample_to_orig_shape = _upsample_like_for_3D(hx2,hx1)
        hx3_upsample_to_orig_shape = _upsample_like_for_3D(hx3,hx1)
        hx4_upsample_to_orig_shape = _upsample_like_for_3D(hx4,hx1)
        hx5_upsample_to_orig_shape = _upsample_like_for_3D(hx5,hx1)
        hx6_upsample_to_orig_shape = _upsample_like_for_3D(hx6,hx1)

        #print(hx2_upsample_to_orig_shape.shape)
        #print(hx3_upsample_to_orig_shape.shape)
        #print(hx4_upsample_to_orig_shape.shape)
        #print(hx5_upsample_to_orig_shape.shape)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape
        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)
        d7=self.outconv_Half_of_UNet(temp_output_1)


        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6,d7),1))
        #print('d0.shape',d0.shape)
        #print('d1.shape',d1.shape)
        #print('d2.shape',d2.shape)
        #print('d3.shape',d3.shape)
        #print('d4.shape',d4.shape)
        #print('d5.shape',d5.shape)
        #print('d6.shape',d6.shape)
        #print('d7.shape',d7.shape)

        return d0,d1,d2,d3,d4,d5,d6,d7
if __name__ == "__main__":

    model=MSFF_QSMNet(in_ch=1,out_ch=1).float().cuda()
    model=model.eval()
    
    
    with torch.no_grad():
        for i in range(1):
            # x= torch.rand(6,1,64,64,64)
            x= torch.rand(1,1,64,64,64)
        
            x=x.cuda()
            x=x.float()
            yy=model(x)
            print('Out Shape :', yy[0].shape)
    from torchsummary import summary
    from pthflops import count_ops
    torch.save(model.state_dict(), 'U2NETP_along_with_Half_UNET_for_3D.pth')
