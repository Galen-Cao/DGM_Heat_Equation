import torch
import torch.nn as nn 
import tqdm
import os

from lib.function import Bell

def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_laplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2[:,d].view(-1,1))    
    hess_diag = torch.cat(hess_diag,1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian

class PDE_DGM(nn.Module):

    def __init__(
        self,
        d,
        hidden_dim: int,
        ts: torch.Tensor = None,
    ):
        super().__init__()
        self.ts = ts
        self.d = d

        self.net_dgm = Net_DGM(d, hidden_dim, activation='Tanh')
        #self.net_dgm = Net(NL=4, NN=16)

    @staticmethod
    def write(msg, logfile, pbar):
        pbar.write(msg)
        with open(logfile, "a") as f:
            f.write(msg)
            f.write("\n")

    def fit(self, max_updates: int, batch_size: int, final: Bell, device, base_dir):
        logfile = os.path.join(base_dir, "log.txt")
        if os.path.exists(logfile):
            os.remove(logfile)
        
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)
        loss_fn = nn.MSELoss()

        pbar = tqdm.tqdm(total=max_updates)
        for it in range(max_updates):
            optimizer.zero_grad()

            # uniform between [-2,2]
            #input_domain = -2+4*torch.rand(batch_size, self.d, device=device, requires_grad=True) 

            # normal distribution with mean 0, sigma 8. 
            input_domain = torch.randn(batch_size, self.d, requires_grad=True) * 8 

            # t ~ uinf([0,T])
            t0, T = self.ts[0], self.ts[-1]
            t = t0 + T*torch.rand(batch_size, 1, device=device, requires_grad=True)
            # u(t,X,S,P,Q)
            u_of_tx = self.net_dgm(t, input_domain)

            grad_u_x = get_gradient(u_of_tx, input_domain)
            grad_u_t = get_gradient(u_of_tx, t)
            laplacian = get_laplacian(grad_u_x, input_domain)
            target_functional = torch.zeros_like(u_of_tx)
            
            pde = grad_u_t + 0.5 * laplacian
            MSE_functional = loss_fn(pde, target_functional)

            # Terminal Condition
            #input_terminal = -2+4*torch.rand(batch_size, self.d, device=device, requires_grad=True)
            input_terminal = torch.randn(batch_size, self.d, requires_grad=True) * 8 
            t = torch.ones(batch_size, 1, device=device) * T
            u_of_tx = self.net_dgm(t, input_terminal)

            target_terminal = final(input_terminal)  # (batch_size, 1)
            MSE_terminal = loss_fn(u_of_tx, target_terminal)
            loss = MSE_functional + MSE_terminal
            # loss = MSE_terminal
            loss.backward()
            optimizer.step()
            scheduler.step()
            if it%10 == 0:
                pbar.update(10)
                self.write("Iteration: {}/{}\t MSE functional: {:.4f}\t MSE terminal: {:.4f}\t Total Loss: {:.4f}".format(it, max_updates, MSE_functional.item(), MSE_terminal.item(), loss.item()), logfile, pbar)
                #pbar.write("Iteration: {}/{}\t MSE terminal: {:.4f}\t Total Loss: {:.4f}".format(it, max_updates, MSE_terminal.item(), loss.item()))

    def drift(self, x):
        """
        """
        return 0

    def diffusion(self, x):
        """
        """
        return torch.ones_like(x)

    def sdeint(self, ts, x0):
        """
        Euler scheme to solve SDE that the states follow

        Args:
            x0: the initial state, (batch_size, (X0, S0, P0, Q0))
            ts: time step
        """
        # (batch_size, N, dim), N is the total time step
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            x_new = x[:,-1,:] + self.drift(x[:,-1,:])*h + self.diffusion(x[:,-1,:])*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments

    
    def MC_value(self, ts: torch.Tensor, x0: torch.Tensor, final: Bell, MC_samples: int):
        """
        Do monte carlo samples 
        """
        assert x0.shape[0] == 1
        x0 = x0.repeat(MC_samples, 1)
        with torch.no_grad():
            x, brownian_increments = self.sdeint(ts, x0)
        final_value = final(x[:,-1,:]) # (batch_size, 1)
        return final_value


    def unbiased_price(self, ts, x0):
        assert x0.shape[0] == 1
        t = torch.ones(1, 1) * ts[0]
        u_of_tx = self.net_dgm(t, x0)
    
        return u_of_tx


class Net_DGM(nn.Module):

    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        output = self.output_layer(S4)
        return output


class DGM_Layer(nn.Module):
    
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
            
        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)
            
    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut), self.activation)
        return l
    
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        
        output = ((1-G))*H + Z*S
        return output


class Net(nn.Module):

    def __init__(self , NL  , NN  ):
        super(Net, self).__init__()
        
        self.NL = NL
        self.NN = NN
        # dimension of the problem (t, X, S, P , Q) 
        self.Input = 2 + 1
        
        self.fc_input = nn.Linear(self.Input, self.NN)
        #torch.nn.init.xavier_uniform_(self.fc_input.weight)
        
        
        self.linears = nn.ModuleList([nn.Linear(self.NN, self.NN) for i in range(self.NL)])
        #for i, l in enumerate(self.linears):    
        #    torch.nn.init.xavier_uniform_(l.weight)
        
        self.fc_output = nn.Linear(self.NN,1)
        #torch.nn.init.xavier_uniform_(self.fc_output.weight)
        
        self.act = torch.tanh
        
    def forward(self, t, x):
        x = torch.cat([t,x],1)
        h = self.act(self.fc_input(x))

        for i, l in enumerate(self.linears):
            h = self.act( l(h) )
        
        out = self.fc_output(h)
        
        return out 