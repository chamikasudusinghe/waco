import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 32 
    PLANES = (16,32,64,64)

    def __init__(self, in_channels, out_channels, D=3, layer_nums=14, weighted=False):
        nn.Module.__init__(self)
        self.D = D
        self.layer_nums = layer_nums
        self.weighted = weighted

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Sparse Matrix Query 
        self.inplanes = self.INIT_DIM
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layers = nn.ModuleList([self.layer1])
        for _ in range(1, self.layer_nums):
            layer = nn.Sequential(
                ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
                ME.MinkowskiReLU(inplace=True)
            )
            self.layers.append(layer)

        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiToFeature())
    
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

        self.feature = nn.Sequential(
          nn.Linear(3, 64),
          nn.ReLU(),
          nn.Linear(64,32),
        )
        
        self.matrix_embedding = nn.Sequential(
          nn.Linear(self.INIT_DIM * self.layer_nums + 32, 256),
          nn.ReLU(),
          nn.Linear(256,128),
        )

        # Super Schedule
        self.isplit = nn.Embedding(18, 32)
        self.ksplit = nn.Embedding(20, 32)
        self.jsplit = nn.Embedding(9, 32)
        self.order = nn.Linear(36, 32) #6x6 Permutation
        self.format1 = nn.Embedding(2, 32)
        self.format2 = nn.Embedding(2, 32)
        self.format3 = nn.Embedding(2, 32)
        self.format4 = nn.Embedding(2, 32)
        self.parchunk = nn.Embedding(10, 32)
        
        self.gpu1 = nn.Embedding(5, 32)
        self.gpu2 = nn.Embedding(5, 32)
        self.gpu3 = nn.Embedding(5, 32)
        
        self.spade1 = nn.Embedding(2, 32)
        self.spade2 = nn.Embedding(2, 32)
        self.spade3 = nn.Embedding(2, 32)
        
        self.device = nn.Embedding(3, 32)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*16,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # Final Layer
        self.final = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        );

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
   
    def embed_sparse_matrix(self, x1: ME.SparseTensor, x2) :
        y_list = []
        y = x1
        for layer in self.layers:
            y = layer(y)
            y_list.append(self.glob_pool(y))
        y = torch.cat(y_list, dim=1)
        #y = F.normalize(torch.cat(y_list, dim=1))
        x2 = self.feature(x2[:, :3])
        x1x2 = torch.cat((y,x2), dim=1)
        x1x2 = self.matrix_embedding(x1x2)
        #x1x2 = F.normalize(x1x2)

        return x1x2

    def embed_super_schedule(self, y) :
        # Super Schedule
        isplit = self.isplit(y[:, 0].long())
        ksplit = self.ksplit(y[:, 1].long())
        jsplit = self.jsplit(y[:, 2].long())
        order = self.order(y[:, 3:39])
        
        f1 = self.format1(y[:, 39].long())
        f2 = self.format2(y[:, 40].long())
        f3 = self.format3(y[:, 41].long())
        f4 = self.format3(y[:, 42].long())
        pchk = self.parchunk(y[:, 43].long())
        
        gp1 = self.gpu1(y[:, 44].long())
        gp2 = self.gpu2(y[:, 45].long())
        gp3 = self.gpu3(y[:, 46].long())
        
        sp1 = self.spade1(y[:, 47].long())
        sp2 = self.spade2(y[:, 48].long())
        sp3 = self.spade3(y[:, 49].long())
        
        d = self.device(y[:, 50].long())
        
        y = torch.cat((isplit,ksplit,jsplit,order,f1,f2,f3,f4,pchk,gp1,gp2,gp3,sp1,sp2,sp3,d), dim=1)
        y = self.schedule_embedding(y)
        #y = F.normalize(y)
        
        return y

    def forward_after_query(self, x, y):
        y = self.embed_super_schedule(y)
        if self.weighted == True:
            wx = self.weight1 * x
            wy = self.weight2 * y
            xy = torch.cat((wx,wy), dim=1)
        else:
            xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy
    
    def forward(self, x1: ME.SparseTensor, x2, y):
        x = self.embed_sparse_matrix(x1,x2)
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy

class ResNet14(ResNetBase):
    LAYERS = (1, 1, 1, 1)

