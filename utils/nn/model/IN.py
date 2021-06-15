import itertools
import numpy as np
import torch
import torch.nn as nn

# only vertex-particle branch
class INTagger(nn.Module):

    def __init__(self,
                 pf_dims,
                 sv_dims,
                 num_classes,
                 pf_features_dims,
                 sv_features_dims,
                 hidden, De, Do,
                 **kwargs):
        super(INTagger, self).__init__(**kwargs)
        self.P = pf_features_dims
        self.N = pf_dims
        self.S = sv_features_dims
        self.Nv = sv_dims
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.De = De
        self.Do = Do
        self.n_targets = num_classes
        self.hidden = hidden
        self.assign_matrices()
        self.assign_matrices_SV()
        self.batchnorm_x = nn.BatchNorm1d(self.P)
        self.batchnorm_y = nn.BatchNorm1d(self.S)

        self.fr = nn.Sequential(nn.Linear(2 * self.P, self.hidden),
                                nn.BatchNorm1d(self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.hidden),
                                nn.BatchNorm1d(self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.De),
                                nn.BatchNorm1d(self.De),
                                nn.ReLU())

        self.fr_pv = nn.Sequential(nn.Linear(self.S + self.P, self.hidden),
                                   nn.BatchNorm1d(self.hidden),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden, self.hidden),
                                   nn.BatchNorm1d(self.hidden),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden, self.De),
                                   nn.BatchNorm1d(self.De),
                                   nn.ReLU())
        
        self.fo = nn.Sequential(nn.Linear(self.P + (2 * self.De), self.hidden),
                                nn.BatchNorm1d(self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.hidden),
                                nn.BatchNorm1d(self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.Do),
                                nn.BatchNorm1d(self.Do),
                                nn.ReLU())
        
        self.fc_fixed = nn.Linear(self.Do, self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1

    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1

    def edge_conv(self, x):
        Orr = torch.matmul(x, self.Rr.to(device=x.device)) # [batch, P, Nr]
        Ors = torch.matmul(x, self.Rs.to(device=x.device)) # [batch, P, Nr]
        B = torch.cat([Orr, Ors], dim=-2) # [batch, 2*P, Nr]
        B = B.transpose(-1, -2).contiguous() # [batch, Nr, 2*P]
        E = self.fr(B.view(-1, 2*self.P)).view(-1, self.Nr, self.De) # [batch, Nr, De]
        E = E.transpose(-1, -2).contiguous() # [batch, De, Nr]
        Ebar_pp = torch.einsum('bij,kj->bik', E, self.Rr.to(device=x.device)) # [batch, De, N]
        return Ebar_pp

    def edge_conv_SV(self, x, y):
        Ork = torch.matmul(x, self.Rk.to(device=x.device)) # [batch, P, Nt]
        Orv = torch.matmul(y, self.Rv.to(device=x.device)) # [batch, S, Nt]
        B = torch.cat([Ork, Orv], dim=-2) # [batch, P+S, Nt]
        B = B.transpose(-1, -2).contiguous() # [batch, Nt, P+S]
        E = self.fr_pv(B.view(-1, self.P+self.S)).view(-1, self.Nt, self.De) # [batch, Nt, De]
        E = E.transpose(-1, -2).contiguous() # [batch, De, Nt]
        Ebar_pv = torch.einsum('bij,kj->bik', E, self.Rk.to(device=x.device)) # [batch, De, N]
        return Ebar_pv
        
    def forward(self, x, y):
        x = self.batchnorm_x(x) # [batch, P, N]
        y = self.batchnorm_y(y) # [batch, S, Nv]
        
        # pf - pf
        Ebar_pp = self.edge_conv(x) # [batch, De, N]
        
        # sv - pf
        Ebar_pv = self.edge_conv_SV(x, y) # [batch, De, N]
        
        # Final output matrix
        C = torch.cat([x, Ebar_pp, Ebar_pv], dim=-2) # [batch, P + 2*De, N]
        C = C.transpose(-1, -2).contiguous() # [batch, N, P + 2*De]
        O = self.fo(C.view(-1, self.P+2*self.De)).view(-1, self.N, self.Do) # [batch, N, Do]
        O = O.transpose(-1, -2).contiguous() # [batch, Do, N]
        
        # Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=-1) # [batch, Do]

        # Classification MLP
        N = self.fc_fixed(N) # [batch, Do]

        return N
