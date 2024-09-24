import numpy as np
import torch
import matplotlib.pyplot as plt

# input X, label=None, k=1, th=1e-2, steps, lr
# output projected X


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_entropy(x, k=1, th=1e-1):
    n = x.shape[0] # the number of data points
    d = x.shape[1] # dimension
    
    Vd = torch.pi**(d/2)/torch.tensor(1+d/2).lgamma().exp()
    gamma = 0.577216
    Psi = -gamma + torch.sum(1/torch.tensor([i for i in range(1,k)]))
    H_const = torch.log(Vd) + torch.log(torch.tensor(n-1)) - Psi
        
    dist = torch.linalg.norm(x[None,:,:] - x[:,None,:], dim=2)
    rho = torch.sort(dist, dim=1).values[:,k]

    H_rho = d * torch.mean(torch.log(rho+th))
    H_total = H_rho + H_const
    return H_total

def get_entropy_with_labels(x, labels, k=1, th=1e-2):
    n_class = len(set(list(labels.cpu().detach().numpy())))
    
    entropy = 0.
    for i in range(n_class):
        member = torch.where(labels == i)[0]
        entropy_i = get_entropy(x[member], k=k, th=th)
        entropy += entropy_i
        
    return entropy

def find_projection(x, target_dim, labels=None, plot_labels=None, k=1, th=1e-2, n_steps=2000, lr=1e-3, log=500, cheat=False, plot=True):
    
    d_total = x.shape[1]
    A = torch.nn.Linear(target_dim, d_total).to(x.device)

    if cheat:
        W = torch.zeros(d_total, target_dim)
        for i in range(target_dim):
            W[i,i] = 1.
        A.weight.data = W

    U = torch.nn.utils.parametrizations.orthogonal(A)
    optimizer = torch.optim.SGD(A.parameters(), lr=lr)

    for i in range(n_steps):

        optimizer.zero_grad()
        x_proj = torch.matmul(x, U.weight)
        if labels == None:
            entropy = get_entropy(x_proj, k=k, th=th)
        else:
            entropy = get_entropy_with_labels(x_proj, labels, k=k, th=th)
            
        entropy.backward()
        optimizer.step()

        if i % log == 0:
            print('step=%d, entropy=%.3f'%(i, entropy))

        if i == 0 or i == n_steps-1:
            if plot:
                x_proj = torch.matmul(x, U.weight)
                x_proj = x_proj.cpu().detach().numpy()
                plt.figure(figsize=(3,3))
                if plot_labels == None:
                    plt.scatter(x_proj[:,0], x_proj[:,[1]])
                else:
                    plt.scatter(x_proj[:,0], x_proj[:,[1]], c=plot_labels.cpu().detach().numpy())
                plt.title(f'step={i}')
                plt.show()
            
    return x_proj, U.weight, entropy
            
def get_data():
    n = 100
    d_structure = 2
    d_random = 8
    d = d_structure + d_random
    n_class = 3
    sigma_inter_cluster = 1.
    sigma_intra_cluster = 0.1
    sigma_random = 0.5

    cluster_centers = torch.randn(n_class, d_structure) * sigma_inter_cluster
    labels = torch.from_numpy(np.random.choice(n_class, n)).long()

    x_structure = cluster_centers[labels] + torch.randn(n, d_structure) * sigma_intra_cluster
    x_random = torch.randn(n, d_random) * sigma_random
    x = torch.cat([x_structure, x_random], dim=1)
    return x, labels


