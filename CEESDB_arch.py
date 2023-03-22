import torch
import torch.nn as nn
import numpy as np


class CAR(nn.Module):
    def __init__(self, nf_in, nf_cond, ca_type):
        super().__init__()
        self.ca_type = ca_type
        if ca_type == 'ECA':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        elif ca_type == 'GP':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(nf_in, nf_in, 1, bias=False)
        elif ca_type == 'CE':
            self.conv = nn.Conv2d(nf_cond, nf_in, 1, bias=False)
        elif ca_type == 'CEAug':
            self.res = nn.Sequential(
            *[nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, stride=1, padding=1, bias=True),
              nn.ReLU(inplace=False),
              nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, stride=1, padding=1, bias=True)
              ]
        )
            self.gamma_conv = nn.Conv2d(nf_cond, nf_in, 1, bias=False)
            self.beta_conv = nn.Conv2d(nf_cond, nf_in, 1, bias=False)
            self.tanh = nn.Tanh()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(nf_in+nf_cond, nf_in, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        if self.ca_type == 'CEAug':
            gamma = self.sigmoid(self.gamma_conv(cond))
            beta = self.tanh(self.beta_conv(cond))
            res = gamma * self.res(x) + beta
            return x + res
        elif self.ca_type == 'ECA':
            logic = self.avg_pool(x)
            logic = self.conv(logic.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        elif self.ca_type == 'GP':
            logic = self.avg_pool(x)
            logic = self.conv(logic)
        elif self.ca_type == 'CE':
            logic = self.conv(cond)
        else:
            logic = self.avg_pool(x)
            logic = self.conv(torch.cat([logic, cond], dim=1))

        logic = self.sigmoid(logic)
        out = x * logic
        return out


class Up(nn.Module):
    def __init__(self, nf_in_up, nf_in, nf_out, nf_cond, ca_type='GP_CE'):
        super().__init__()
        self.up = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=nf_in_up,
                out_channels=nf_out,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

        self.relu1 = nn.ReLU(inplace=False)
        self.ca1 = CAR(nf_in, nf_cond, ca_type)
        self.conv1 = nn.Conv2d(
            in_channels=nf_in,
            out_channels=nf_out,
            kernel_size=3,
            padding=1,
        )

        self.relu2 = nn.ReLU(inplace=False)
        self.ca2 = CAR(nf_out, nf_cond, ca_type)
        self.conv2 = nn.Conv2d(
            in_channels=nf_out,
            out_channels=nf_out,
            kernel_size=3,
            padding=1,
        )

    def forward(self, small_x, normal_x_lst, cond):
        # print(small_x.shape)
        # print(normal_x_lst[0].shape)
        f = self.up(small_x)
        f = torch.cat([f]+normal_x_lst, dim=1)
        f = self.relu1(f)
        f = self.ca1(f, cond)
        f = self.conv1(f)
        f = self.relu2(f)
        f = self.ca2(f, cond)
        f = self.conv2(f)
        return f


class Down(nn.Module):
    def __init__(self, nf_in, nf_out, nf_cond, ca_type='GP_CE'):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.ca1 = CAR(nf_in, nf_cond, ca_type)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nf_in, out_channels=nf_out, kernel_size=3, padding=1, stride=2),
        )
        self.relu2 = nn.ReLU(inplace=False)
        self.ca2 = CAR(nf_out, nf_cond, ca_type)
        self.conv2 = nn.Conv2d(in_channels=nf_out, out_channels=nf_out, kernel_size=3, padding=1)

    def forward(self, x, cond):
        f = self.relu1(x)
        f = self.ca1(f, cond)
        f = self.conv1(f)
        f = self.relu2(f)
        f = self.ca2(f, cond)
        f = self.conv2(f)
        return f



class CEESDBNet(nn.Module):
    '''
    dense UNet arch as rbqe
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, cond_dim=1, ca_type='GP_CE', order=0):
        super(CEESDBNet, self).__init__()

        self.head = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.shared_body = nn.Sequential(*[
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ])

        # cond alone body
        self.cond_body = nn.Sequential(*[
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.condfc = nn.Sequential(*[
            nn.Conv2d(nf, nf // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 4, cond_dim, 1),
            nn.Sigmoid()
        ])
        if order > 0:
            for p in self.parameters():
                p.requires_grad = False

        # restoration body
        nlevel = 5
        self.nlevel = nlevel
        for idx_level in range(nlevel):
            setattr(self, f'down_{idx_level}', Down(
                nf_in=nf,
                nf_out=nf,
                nf_cond=cond_dim,
                ca_type=ca_type,
            ))

            if idx_level < order - 1:
                down = getattr(self, f'down_{idx_level}')
                for p in down.parameters():
                    p.requires_grad = False

            for idx_up in range(idx_level+1):
                setattr(self, f'up_{idx_level}_{idx_up}', Up(
                    nf_in_up=nf,
                    nf_in=nf*(2+idx_up),
                    nf_out=nf,
                    nf_cond=cond_dim,
                    ca_type=ca_type,
                ))

                if idx_level < order - 1:
                    up = getattr(self, f'up_{idx_level}_{idx_up}')
                    for p in up.parameters():
                        p.requires_grad = False

        # out
        self.out_relu = nn.ModuleList([nn.ReLU(inplace=False) for _ in range(nlevel)])
        self.out_ca = nn.ModuleList([CAR(nf_in=nf, nf_cond=cond_dim, ca_type=ca_type) for _ in range(nlevel)])
        self.out_conv = nn.ModuleList([nn.Conv2d(in_channels=nf, out_channels=out_nc, kernel_size=3, padding=1)
                                       for _ in range(nlevel)])

        for i in range(order - 1):
            ca = self.out_ca[i]
            for p in ca.parameters():
                p.requires_grad = False
            conv = self.out_conv[i]
            for p in conv.parameters():
                p.requires_grad = False

    def forward(self, x, mode='train'):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 32) * 32 - h)
        paddingRight = int(np.ceil(w / 32) * 32 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        f_head = self.head(x)
        f_shared = self.shared_body(f_head)
        f_cond = self.cond_body(f_shared)
        f_cond = self.avg_pool(f_cond)
        f_cond2 = self.condfc(f_cond)
        cond = f_cond2.view(f_cond2.size(0), -1)
        # if cond < 0.01:
        #     return x[..., :h, :w], cond
        f_lst = [f_shared]
        f_lst_lst = []
        out_t_lst = []
        for idx_level in range(self.nlevel):
            f_lst_lst.append(f_lst)

            down = getattr(self, f'down_{idx_level}')
            f = down(f_lst_lst[-1][0], f_cond2)
            f_lst = [f]
            for idx_up in range(idx_level+1):
                inp_lst = []
                for pre_f_lst in f_lst_lst:
                    ndepth = idx_level + 1 - idx_up
                    if len(pre_f_lst) >= ndepth:
                        inp_lst.append(pre_f_lst[-ndepth])

                up = getattr(self, f'up_{idx_level}_{idx_up}')
                f = up(f_lst[-1], inp_lst, f_cond2)
                f_lst.append(f)

            out = self.out_relu[idx_level](f_lst[-1])
            out = self.out_ca[idx_level](out, f_cond2)
            out = self.out_conv[idx_level](out)
            out += x
            out = out[..., :h, :w]
            out_t_lst.append(out)

            if mode == 'val' and cond * 5 < (idx_level + 1):
                return out, cond
        if mode == 'val':
            return out, cond
        out1, out2, out3, out4, out5 = out_t_lst
        return out1, out2, out3, out4, out5, cond


class ResBlock(nn.Module):
    def __init__(self, in_nc=64, out_nc=64):
        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            *[nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=True),
              nn.ReLU(inplace=False),
              nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=True)
              ]
        )

    def forward(self, x):
        res = self.res(x)
        return x + res


class QFAttention(nn.Module):
    def __init__(self, in_nc=64, out_nc=64, cond_dim=64):
        super(QFAttention, self).__init__()

        self.res = nn.Sequential(
            *[nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=True),
              nn.ReLU(inplace=False),
              nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=True)
              ]
        )

        self.gamma_conv = nn.Sequential(
            *[nn.Conv2d(cond_dim, out_nc, 1, bias=False),
              nn.Sigmoid()
              ]
        )

        self.beta_conv = nn.Sequential(
            *[nn.Conv2d(cond_dim, out_nc, 1, bias=False),
              nn.Tanh()
              ]
        )

    def forward(self, x, cond):
        gamma = self.gamma_conv(cond)
        beta = self.beta_conv(cond)

        res = gamma * self.res(x) + beta
        return x + res


class simpleUNet(nn.Module):
    def __init__(self, nf=[64, 128, 256, 512], cond_dim=64):
        super(simpleUNet, self).__init__()
        self.m_down1 = nn.Sequential(
            *[ResBlock(nf[0], nf[0]),
              nn.Conv2d(in_channels=nf[0], out_channels=nf[1], kernel_size=2, stride=2, padding=0, bias=True),
              nn.ReLU(inplace=True)
              ]
        )
        self.m_down2 = nn.Sequential(
            *[ResBlock(nf[1], nf[1]),
              nn.Conv2d(in_channels=nf[1], out_channels=nf[2], kernel_size=2, stride=2, padding=0, bias=True),
              nn.ReLU(inplace=True)
              ]
        )
        self.m_down3 = nn.Sequential(
            *[ResBlock(nf[2], nf[2]),
              nn.Conv2d(in_channels=nf[2], out_channels=nf[3], kernel_size=2, stride=2, padding=0, bias=True),
              nn.ReLU(inplace=True)
              ]
        )

        self.m_body = nn.Sequential(
            *[ResBlock(nf[3], nf[3]) for _ in range(2)]
        )

        self.m_up3 = nn.Sequential(
            *[nn.ConvTranspose2d(in_channels=nf[3], out_channels=nf[2], kernel_size=2, stride=2, padding=0, bias=True),
              nn.ReLU(inplace=True)
              ]
        )

        self.up3_qa = QFAttention(nf[2], nf[2], cond_dim)

        self.m_up2 = nn.Sequential(
            *[nn.ConvTranspose2d(in_channels=nf[2], out_channels=nf[1], kernel_size=2, stride=2, padding=0, bias=True),
              nn.ReLU(inplace=True)
              ]
        )

        self.up2_qa = QFAttention(nf[1], nf[1], cond_dim)

        self.m_up1 = nn.Sequential(
            *[nn.ConvTranspose2d(in_channels=nf[1], out_channels=nf[0], kernel_size=2, stride=2, padding=0, bias=True),
              nn.ReLU(inplace=True)
              ]
        )

        self.up1_qa = QFAttention(nf[0], nf[0], cond_dim)

    def forward(self, x1, cond):
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x = self.m_down3(x3)
        x = self.m_body(x)
        x = self.m_up3(x)
        x = self.up3_qa(x, cond)
        x = x + x3
        x = self.m_up2(x)
        x = self.up2_qa(x, cond)
        x = x + x2
        x = self.m_up1(x)
        x = self.up1_qa(x, cond)
        x = x + x1
        return x


class CEESDBNet2(nn.Module):
    '''
    cascaded UNet arch to compare with fbcnn
    '''

    def __init__(self, in_nc=3, out_nc=3, nf=[64, 128, 256, 512], cond_dim=1, order=1):
        super(CEESDBNet2, self).__init__()

        self.head = nn.Conv2d(in_nc, nf[0], 3, 1, 1)

        self.shared_body = nn.Sequential(*[
            nn.Conv2d(nf[0], nf[0], 3, stride=1, padding=1),
            nn.Conv2d(nf[0], nf[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf[0], nf[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ])

        # cond alone body
        self.cond_body = nn.Sequential(*[
            nn.Conv2d(nf[0], nf[0], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf[0], nf[0], 3, 1, 1),
            nn.ReLU(inplace=True),
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.condfc = nn.Sequential(*[
            nn.Conv2d(nf[0], nf[0] // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf[0] // 4, cond_dim, 1),
            nn.Sigmoid()
        ])
        if order > 0:
            for p in self.parameters():
                p.requires_grad = False

        self.stage1 = simpleUNet(nf, cond_dim=cond_dim)
        self.tail1 = nn.Conv2d(nf[0], out_nc, 3, 1, 1)
        self.stage2 = simpleUNet(nf, cond_dim=cond_dim)
        self.tail2 = nn.Conv2d(nf[0], out_nc, 3, 1, 1)
        self.stage3 = simpleUNet(nf, cond_dim=cond_dim)
        self.tail3 = nn.Conv2d(nf[0], out_nc, 3, 1, 1)
        self.stage4 = simpleUNet(nf, cond_dim=cond_dim)
        self.tail4 = nn.Conv2d(nf[0], out_nc, 3, 1, 1)

        for i in range(1, order):
            stage = getattr(self, f'stage{i}')
            for p in stage.parameters():
                p.requires_grad = False
            tail = getattr(self, f'tail{i}')
            for p in tail.parameters():
                p.requires_grad = False

    def forward(self, x, mode='train', qf_input=None):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        f_head = self.head(x)
        f_shared = self.shared_body(f_head)
        f_cond = self.cond_body(f_shared)
        f_cond = self.avg_pool(f_cond)
        f_cond2 = self.condfc(f_cond) if qf_input is None else qf_input
        cond = f_cond2.view(f_cond2.size(0), -1)

        f1 = self.stage1(f_shared, f_cond2)
        f = f_head + f1
        out1 = self.tail1(f)
        out1 = out1[..., :h, :w]
        if mode == 'val' and cond < 0.25:
            return out1, cond

        f2 = self.stage2(f1, f_cond2)
        f = f_head + f2
        out2 = self.tail2(f)
        out2 = out2[..., :h, :w]
        if mode == 'val' and cond < 0.5:
            return out2, cond

        f3 = self.stage3(f2, f_cond2)
        f = f_head + f3
        out3 = self.tail3(f)
        out3 = out3[..., :h, :w]
        if mode == 'val' and cond < 0.75:
            return out3, cond

        f4 = self.stage4(f3, f_cond2)
        f = f_head + f4
        out4 = self.tail4(f)
        out4 = out4[..., :h, :w]
        if mode == 'val':
            return out4, cond
        return out1, out2, out3, out4, cond
