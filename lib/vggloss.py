import numpy as np 
import torch 
import torchvision
import TorchSUL.Model as M
import torch.nn.functional as F

class MultiOutput(M.Model):
    def initialize(self):
        # net = torchvision.models.squeezenet1_1(pretrained=True)
        self.net = torchvision.models.vgg16(pretrained=True).float()
        for param in self.net.parameters():
            param.requires_grad = False
        print(self.net)
        self.net = list(self.net.children())[0]
        self.submodules = list(self.net.children())

    def forward(self, x, max_layer=99):
        results = []
        for mod in self.submodules:
            x = mod(x)
            # print(x.shape)
            results.append(x)
            if len(results)==max_layer:
                break
        return results

class VGGLoss(M.Model):
    Content_layer = [3, 8]
    Style_layer = [8, 15]
    Max_layer = 16
    def initialize(self, content_w=1.0, style_w=0.1):
        self.content_w = content_w
        self.style_w = style_w
        self.net = MultiOutput()
        BGR2RGB = np.float32([[0,0,1], [0,1,0], [1,0,0]])
        self.register_buffer('BGR2RGB', torch.from_numpy(BGR2RGB))
        IMAGENET_MEAN = np.float32([0.485, 0.456, 0.406]).reshape([3, 1, 1])
        IMAGENET_STD = np.float32([0.229, 0.224, 0.225]).reshape([3, 1, 1])
        self.register_buffer('IMAGENET_MEAN', torch.from_numpy(IMAGENET_MEAN))
        self.register_buffer('IMAGENET_STD', torch.from_numpy(IMAGENET_STD))

    def preprocess(self, x):
        # x = x / 2 + 0.5 
        x = torch.einsum('ijkl,mj->imkl', x, self.BGR2RGB)
        x = (x - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return x 

    def forward(self, x, y, norm=True, need_resize=False):
        if norm:
            x = self.preprocess(x)
            y = self.preprocess(y)
        if need_resize:
            x = F.interpolate(x, (224, 224), mode='bilinear')
            y = F.interpolate(y, (224, 224), mode='bilinear')
        feats_x = self.net(x, max_layer=self.Max_layer)
        with torch.no_grad():
            feats_y = self.net(y, max_layer=self.Max_layer)
        style_losses = []
        for idx in self.Style_layer:
            ls = self.style_loss(feats_x[idx], feats_y[idx])
            style_losses.append(ls)
        stl_ls = sum(style_losses) / len(style_losses)
        content_losses = []
        for idx in self.Content_layer:
            ls = self.content_loss(feats_x[idx], feats_y[idx])
            content_losses.append(ls)
        ct_ls = sum(content_losses) / len(content_losses)
        return stl_ls * self.style_w + ct_ls * self.content_w

    def gram_matrix(self, x):
        shape = x.shape
        feat_flatten = x.view([shape[0], shape[1], shape[2]*shape[3]]) 
        result = torch.matmul( feat_flatten, torch.transpose(feat_flatten, 1,2)) 
        return result               # [N, C, C]

    def style_loss(self, x, y):
        gram_x = self.gram_matrix(x)
        gram_y = self.gram_matrix(y)
        return torch.pow(gram_x - gram_y, 2).mean() / (4 * x.shape[2] * x.shape[3] * x.shape[2] * x.shape[3])

    def content_loss(self, x, y):
        return torch.pow(x-y, 2).mean()


if __name__=='__main__':
    model = VGGLoss()
    x1 = torch.zeros(2, 3, 224, 224)
    x2 = torch.ones(2, 3, 224, 224)
    y = model(x1, x2)
    print(y.shape, y)
