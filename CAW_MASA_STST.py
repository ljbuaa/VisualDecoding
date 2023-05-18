'''
CAW_MASA_STST
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,with_bn=True,with_relu=True,stride=1,padding=0,bias=True):
        super().__init__()
        self.with_bn=with_bn
        self.with_relu=with_relu
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,stride=stride,padding=padding,bias=bias)
        self.batchNorm=None
        self.relu=None
        if with_bn:
            self.batchNorm=nn.BatchNorm2d(out_channels)
        if with_relu:
            self.relu=nn.ELU()
    def forward(self, x):
        out=self.conv2d(x)
        if self.with_bn:
            out=self.batchNorm(out)
        if self.with_relu:
            out=self.relu(out)
        return out

class GCN(nn.Module):
    def __init__(self, chns,feas):
        super().__init__()  
        self.a=nn.Parameter(torch.rand((chns,chns)))
        self.k=2
        self.num_of_filters=feas
        self.Theta=nn.Parameter(torch.randn((self.k,feas, self.num_of_filters)))
        self.adj=None
    def forward(self, x):
        x = x.squeeze()
        b, c, l = x.size()
        fea_matrix = x
        # Similarity Matrix
        self.diff = (x.expand([c,b,c,l]).permute(2,1,0,3)-x).permute(1,0,2,3)
        self.diff=torch.abs(self.diff).sum(-1)
        self.diff=F.normalize(self.diff,dim=0)
        tmpS = torch.exp(torch.relu((1-self.diff)*self.a))
        # Laplacian matrix 
        self.adj = tmpS / torch.sum(tmpS,axis=1,keepdims=True)
        D = torch.diag_embed(torch.sum(self.adj,axis=1))
        L = D - self.adj
        # Chebyshev graph convolution
        firstOrder=torch.eye(c).cuda()
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - firstOrder
        cheb_polynomials = [firstOrder, L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        output = torch.zeros(b, c, self.num_of_filters).cuda()
        for kk in range(self.k):
            T_k = cheb_polynomials[kk].expand([b,c,c])
            rhs = torch.bmm(T_k.permute(0, 2, 1), fea_matrix)
            output = output + torch.matmul(rhs, self.Theta[kk])
        output=torch.relu(output)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)") 
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Channel Attention Weighting
class CAW(nn.Module):
    def __init__(self, channel, reduction = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace  = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape)==3:
            b, c, t = x.size()
            xstd=((x-x.mean(-1).view(b,c,1))**2)
            xstd = F.normalize(xstd.sum(-1),dim=-1)
            attn = self.fc(xstd).view(b, c, 1)
        else:
            b, c, s, t = x.size()
            xstd=((x-x.mean(-1).view(b,c,s,1))**2)
            xstd = F.normalize(xstd.sum(-1),dim=-1)
            attn = self.fc(xstd).view(b, c, s, 1)
        out = x * attn.expand_as(x)
        return out, attn


class STSTransformerBlock(nn.Module):
    def __init__(self, emb_size1,emb_size2,num_heads=5,drop_p=0.5,forward_expansion=4,forward_drop_p=0.5):
        super().__init__()
        self.emb_size = emb_size1
        self.att_drop1 = nn.Dropout(drop_p)
        self.projection1 = nn.Linear(emb_size1, emb_size1)
        self.projection2 = nn.Linear(emb_size1, emb_size1)
        self.drop1=nn.Dropout(drop_p)
        self.drop2=nn.Dropout(drop_p)

        self.layerNorm1=nn.LayerNorm(emb_size1)
        self.layerNorm2=nn.LayerNorm(emb_size2)
 
        self.queries1 = nn.Linear(emb_size1, emb_size1)
        self.values1 = nn.Linear(emb_size1, emb_size1)
        self.keys2 = nn.Linear(emb_size2, emb_size2)
        self.values2 = nn.Linear(emb_size2, emb_size2)

        self.layerNorm3=nn.LayerNorm(emb_size1+emb_size2)
        self.mha=MultiHeadAttention(emb_size1+emb_size2, num_heads, 0.5)
        self.drop3=nn.Dropout(drop_p)

        self.ffb=nn.Sequential(
            nn.LayerNorm(emb_size1+emb_size2),
            FeedForwardBlock(
                emb_size1+emb_size2, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self, x1, x2):
        x1=rearrange(x1, 'b e (h) (w) -> b (h w) e ')
        x2=rearrange(x2, 'b e (h) (w) -> b (h w) e ')
        res1=x1
        res2=x2

        x1 = self.layerNorm1(x1)
        x2 = self.layerNorm2(x2)
        queries1 = self.queries1(x1) 
        values1 = self.values1(x1)
        keys2 = self.keys2(x2)
        values2 = self.values2(x2)

        energy = torch.einsum('bqd, bkd -> bqk', keys2, queries1)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop1(att)

        out1 = torch.einsum('bal, blv -> bav ', att, values1)
        out1 = self.projection1(out1)
        x1 = self.drop1(out1)
        x1+=res1

        out2 = torch.einsum('bal, blv -> bav ', att, values2)
        out2 = self.projection2(out2)
        x2 = self.drop2(out2)
        x2+=res2

        x=torch.cat((x1,x2),dim=-1)
        res = x
        x=self.layerNorm3(x)
        x=self.mha(x)
        x=self.drop3(x)
        x += res

        res = x
        x = self.ffb(x)
        x += res
        x = rearrange(x, 'b t e -> b e 1 t')
        return x

class CAW_MASA_STST(nn.Module):
    def __init__(self,classNum,channelNum,chan_spe,tlen=32):
        super().__init__()
        self.chunks=5
        self.conv1s=[]
        self.speWidth=chan_spe//self.chunks
        for i in range(self.chunks):
            ASA= nn.Sequential(
                ConvBlock(channelNum, 25, (self.speWidth,1)),
                Rearrange("a b c d -> a (c d) b"),
                GCN(tlen,25),
                Rearrange("a b (c d) -> a c d b",d=1),
                ConvBlock(25, 2, (1,1)),
            )
            self.conv1s.append(ASA.cuda())

        self.spe1 =ConvBlock(channelNum,30,(chan_spe,1))
        self.spe2 = nn.Sequential(
            ConvBlock(40,30,(1,13)),
            ConvBlock(30,10,(1,11)),
        )
        self.speAvgPool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )
        self.spa1 = ConvBlock(1,40,(channelNum,1))
        self.spa2 = nn.Sequential(
            ConvBlock(40,30,(1,13)),
            ConvBlock(30,10,(1,11)),
        )
        self.spaAvgPool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        self.fuseConv1=nn.Sequential(
            ConvBlock(80,70,(1,13)),
            ConvBlock(70,80,(1,11)),
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        self.fuseAvgPool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )
        self.fusinTB=STSTransformerBlock(40,40)
       
        self.caw=CAW(chan_spe,2)

        self.feaLen=(100)*8
        self.classify = nn.Sequential(
            nn.Linear(self.feaLen, classNum),
            nn.Softmax(dim=1),
        )
    def forward(self, x, xcwt):
        xcwt,_=self.caw(xcwt)
        #MASA
        xcwts=xcwt.chunk(self.chunks,dim=2)
        out1s=[]
        for i in range(self.chunks):
            out1=self.conv1s[i](xcwts[i])
            out1s.append(out1)
        gcn_out=torch.cat(out1s,1)

        spe1_out= self.spe1(xcwt) 
        spe1_out= torch.cat((spe1_out,gcn_out), dim=1)

        spe2_out= self.spe2(spe1_out)
        spe_out=self.speAvgPool(spe2_out)
        spe_out = spe_out.squeeze()

        spa1_out= self.spa1(x.unsqueeze(1))
        spa2_out1= self.spa2[0](spa1_out)
        spa2_out2= self.spa2[1](spa2_out1)
        spa_out=self.spaAvgPool(spa2_out2)
        spa_out = spa_out.squeeze()

        fuse_out1 = self.fusinTB(spe1_out,spa1_out)
        fuse_out1 = self.fuseConv1(fuse_out1).squeeze()

        out = torch.cat((spe_out,spa_out,fuse_out1),dim=1)

        out = self.classify(out.reshape(-1,self.feaLen))
        return out

if __name__ == '__main__':
    x=torch.randn(10,64,30).cuda() # EEG data with 64 channel x 30 timepoint
    x_spe=torch.randn(10,64,20,30).cuda() #  time-frequency images of EEG with 64 channel x 20 frequency scale x 30 timepoint
    model=CAW_MASA_STST(2,64,20,30).cuda()
    pre_y=model(x,x_spe)
    print("pre_y.shape:",pre_y.shape)