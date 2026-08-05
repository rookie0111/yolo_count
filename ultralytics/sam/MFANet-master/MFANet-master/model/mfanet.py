import torch
from torch import nn
import torch.nn.functional as F
import math
import model.resnet as models
import model.vgg as vgg_models

class TPM(nn.Module):
    def __init__(self, in_channels, sub_sample=True, bn_layer=True):
        super(TPM, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.peak_pool = nn.AdaptiveMaxPool2d(1)
        self.cos_similarity = nn.CosineSimilarity()
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=(1, 1), stride=(1, 1), padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=(1, 1), stride=(1, 1), padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g,
                                   nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi,
                                     nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        batch_size = x.size(0)

        x1 = self.global_pool(x)
        x2 = self.peak_pool(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        x3 = self.avg_pool(z)
        out = torch.mean(torch.stack([x1,x2,x3], dim=1),dim=1)
        return out

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

class channel_attention(nn.Module):
    def __init__(self, channel=256, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = 7 // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_pool_out,_ = torch.max(x, dim=1, keepdim=True)  # channel max pooling
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)  # channel mean pooling
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x

class Decoder(nn.Module):
    def __init__(self,reduce_dim = 256):
        super().__init__()
        self.Done1 = nn.Sequential(
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=(2, 2), stride=2, padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.Done2 = nn.Sequential(
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=(2, 2), stride=2, padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=(1, 1),  padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),

        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(reduce_dim,reduce_dim,kernel_size=(3,3),padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls_head= nn.Sequential(
            nn.Conv2d(reduce_dim,reduce_dim,kernel_size=(3,3),padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(reduce_dim, 2, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.rb1 = nn.Sequential(
            RB(256,256),
            RB(256,256),
        )
        self.rb2 = nn.Sequential(
            RB(256,256),
            RB(256,256),
        )
        self.channel_attention = channel_attention(channel=256, ratio=16)
        self.spacial_attention = spatial_attention(kernel_size=7)
        self._init_weight()

    def forward(self, x):
        _,_,h,w = x.size()
        x1 = self.Done1(x)
        x1_up = F.interpolate(x1, size=(h,w), mode='bilinear',align_corners=True)
        x2 = self.Done2(x1)
        x2_up = F.interpolate(x2, size=(h,w), mode='bilinear',align_corners=True)
        out = torch.cat([x, x1_up, x2_up], dim=1)
        out = self.conv_cat(out)
        out1 = self.rb1(out)
        out2 = self.channel_attention(out)
        out2 = self.spacial_attention(out2)
        out2 = self.rb2(out2)
        out = out1+out2
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SAM(nn.Module):
    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat
def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class mfanet(nn.Module):
    def __init__(self, args):
        super(mfanet, self).__init__()
        from torch.nn import BatchNorm2d as BatchNorm

        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        self.classes = args.classes
        self.pretrained = True
        models.BatchNorm = BatchNorm
        self.layers = args.layers

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('>>>>>>>>> Using ResNet {}<<<<<<<<<'.format(self.layers))
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        self.tpm = TPM(reduce_dim)
        self.decoder = Decoder(reduce_dim)
        self.sam = SAM(reduce_dim)
        self.rb = nn.Sequential(
            RB(reduce_dim,reduce_dim),
            RB(reduce_dim,reduce_dim),
            nn.Conv2d(reduce_dim,2,kernel_size=1,padding=0,bias=False)
        )
        self.rb_merge = nn.Sequential(
            RB(reduce_dim,reduce_dim),
            RB(reduce_dim,reduce_dim),
        )

        if self.vgg:
            fea_dim = 512 + 256

        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 3 + 1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        if self.vgg:
            self.res1_simple = nn.Sequential(
                nn.Conv2d(512, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )

            self.res2_simple = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.cls_simple = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            )
        else:
            self.res1_simple = nn.Sequential(
                nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),

            )
            self.res2_simple = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.cls_simple = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            )


        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def get_optim(self, args, LR):
        optimizer = torch.optim.SGD(
            [
                {'params': self.down_query.parameters()},
                {'params': self.down_supp.parameters()},
                {'params': self.init_merge.parameters()},
                {'params': self.tpm.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.sam.parameters()},
                {'params': self.rb.parameters()},
                {'params': self.rb_merge.parameters()},

            ],
            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        return optimizer

    def forward(self,
                x,
                s_x=torch.FloatTensor(4, 1, 3, 200, 200).cuda(),
                s_y=torch.FloatTensor(4, 1, 200, 200).cuda(),
                y=None,
                ):

        x_size = x.size()
        h = int(x_size[-1])
        w = int(x_size[-2])

        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)


            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2,
                                             size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear',
                                             align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], dim=1)
        query_feat = self.down_query(query_feat)
        mask_list = []
        supp_ref_list = []
        supp_att_list = []

        final_supp_list = []
        for i in range(self.shot):
            supp_gt = (s_y[:, i, :, :] == 1).float().unsqueeze(1)

            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4 = self.layer4(supp_feat_3)

                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2,
                                                size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear',
                                                align_corners=True)

                mask = F.interpolate(supp_gt, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)

                final_supp_list.append(supp_feat_4)  # prior
                mask_list.append(supp_gt)  # prior and prototype

                # --- support fusion --- #
                supp_feat = torch.cat([supp_feat_3, supp_feat_2], dim=1)
                supp_feat_tem = self.down_supp(supp_feat)  # [4,256,25,25]

                 # ---support REF  --- #
                supp_ref = supp_feat_tem * mask
                supp_ref = self.tpm(supp_ref)  # [1,256,1,1]
                supp_ref_list.append(supp_ref)

                # --- support Attention  --- #
                supp_att = self.sam(supp_feat_tem, mask)
                supp_att_list.append(supp_att)

        if self.shot>1:
            supp_ref = supp_ref_list[0]
            for i in range(1, len(supp_ref_list)):
                supp_ref = supp_ref + supp_ref_list[i]
            supp_ref = supp_ref / len(supp_ref_list)
        else:
            supp_ref = supp_ref_list[0]
        supp_ref = supp_ref.expand_as(query_feat)
        if self.shot>1:
            supp_att = supp_att_list[0]
            for i in range(1, len(supp_att_list)):
                supp_att = supp_att + supp_att_list[i]
            supp_att = supp_att / len(supp_att_list)
        else:
            supp_att = supp_att_list[0]
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)

        merge_feat_bin = torch.cat([query_feat, supp_ref, supp_att, corr_query_mask], dim=1)   # 支持集特征，查询集特征 ，支持集先验，边缘特征
        merge_feat = self.init_merge(merge_feat_bin)
        merge_feat = self.rb_merge(merge_feat)

        merge_feat = F.interpolate(merge_feat, size=(h, w), mode='bilinear', align_corners=True)
        output_fin = self.decoder(merge_feat)
        output_fin = self.rb(output_fin)
        if self.training == False:
            output_fin = self.max_pool(output_fin)
            output_fin = -self.max_pool(-output_fin)

        if self.training:
            main_loss = self.criterion(output_fin, y.long())

            return output_fin.max(1)[1], main_loss
        else:
            return output_fin
