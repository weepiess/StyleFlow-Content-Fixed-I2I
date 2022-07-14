import torch
from torch import nn


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class ToStyleHead(nn.Module):
    def __init__(self,input_dim=512, out_dim=512):
        super(ToStyleHead, self).__init__()
        self.out_dim = out_dim

        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(input_dim,affine=True),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class SAN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SAN, self).__init__()
        self.encoder = ToStyleHead(input_dim=input_dim, out_dim=768)
        self.code_encode_mean = nn.Linear(1920,1024)
        self.act_code_encode_mean = nn.PReLU()

        self.code_encode_mean_nl = nn.Linear(1024,512)
        self.act_code_encode_mean_nl = nn.PReLU()

        self.code_encode_mean_nl2 = nn.Linear(512,2*output_dim)

        self.fc_mean_enc = nn.Linear(768,2*output_dim)
        self.act_fc_mean_enc = nn.PReLU()

        self.fc_mean = nn.Linear(output_dim,output_dim,bias=False)

        self.fc_std_enc = nn.Linear(768,output_dim)
        self.act_fc_std_enc = nn.PReLU()

        self.fc_std = nn.Linear(output_dim,output_dim,bias=False)

        self.fc_mean_nl2 = nn.Linear(output_dim*4,output_dim)
        self.act_fc_mean_nl2 = nn.PReLU()
        #self.fc_mean2 = nn.Linear(output_dim,output_dim)

        self.fc_std_nl2= nn.Linear(output_dim*4,output_dim)
        self.act_fc_std_nl2 = nn.PReLU()
        #self.fc_std2 = nn.Linear(output_dim,output_dim)

    def forward(self, x, code):

        x = self.encoder(x)
        x = torch.flatten(x, 1)

        code_c_mean = self.act_code_encode_mean(self.code_encode_mean(code))
        code_c_mean = self.act_code_encode_mean_nl(self.code_encode_mean_nl(code_c_mean))
        code_c_mean = torch.tanh(self.code_encode_mean_nl2(code_c_mean))

        x_mean = self.act_fc_mean_enc(self.fc_mean_enc(x))
        
        merge = torch.cat([x_mean,code_c_mean],dim=1)

        mean = self.act_fc_mean_nl2(self.fc_mean_nl2(merge))
        std = self.act_fc_std_nl2(self.fc_std_nl2(merge))

        mean = self.fc_mean(mean)
        std = self.fc_std(std)

        return mean,std
        
class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, content, style):
        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    

class AdaIN_SET(nn.Module):
    def __init__(self):
        super().__init__()
        #self.activate = nn.ReLU(inplace=True)

    def forward(self, content, style_mean, style_std):
        assert style_mean is not None
        assert style_std is not None

        size = content.size()

        content_mean, content_std = calc_mean_std(content)

        style_mean = style_mean.reshape(size[0],content_mean.shape[1],1,1)
        style_std = style_std.reshape(size[0],content_mean.shape[1],1,1)
        # style_mean = style_mean.repeat(size[0],1,1,1)
        # style_std = style_std.repeat(size[0],1,1,1)
        
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        sum_mean = style_mean.expand(size)# + content_mean.expand(size)#torch.clamp(style_mean.expand(size) + content_mean.expand(size),0)
        sum_std = style_std.expand(size)# + content_std.expand(size)#torch.clamp(style_std.expand(size)+content_std.expand(size),0)
        return normalized_feat*sum_std + sum_mean
        #return normalized_feat * (style_std.expand(size)+content_std.expand(size)) + style_mean.expand(size) + content_mean.expand(size)

class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc
        