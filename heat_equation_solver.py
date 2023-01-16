import torch
import numpy as np
import argparse
import os
import pandas as pd

from lib.dgm import PDE_DGM
from lib.function import Bell


def train(
    T,
    n_steps,
    d,
    hidden_dim,
    max_updates,
    batch_size, 
    base_dir,
    device
):
    ts = torch.linspace(0,T,n_steps+1, device=device)
    final = Bell()
    pde_solver = PDE_DGM(
        d=d,
        hidden_dim=hidden_dim,
        ts=ts,
    )
    pde_solver.to(device)
    pde_solver.fit(
        max_updates=max_updates,
        batch_size=batch_size, 
        final=final,
        device=device,
        base_dir=base_dir,
    )

    # Save the trained network 
    PATH = os.path.join(base_dir, 'Heat_Model.pt')
    torch.save(pde_solver.state_dict(), PATH)

    # Load the model
    model = PDE_DGM(        
        d=d,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(torch.load(PATH))
    model.eval()
    # Test
    x0 = torch.zeros(1,d, device=device)
    mc_value = model.MC_value(ts=ts, x0=x0, final=final, MC_samples=5000)
    unbiased_v = model.unbiased_price(ts, x0)
    results = {'unbiased value function':unbiased_v.item(), 
                'MC value function':mc_value.mean().item()
            }
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(base_dir, 'results.csv'))


def visualise(    
    T,
    n_steps,
    d,
    hidden_dim,
    base_dir,
    device,
    x_steps: int = 1000,
):
    assert (d == 1 or d == 2)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation

    PATH = os.path.join(base_dir, 'Heat_Model.pt')

    # Load the model
    model = PDE_DGM(        
        d=d,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(torch.load(PATH))
    model.eval()

    if d == 1:
        # 1-D heat equation: 3D plot with (t, x, net(t,x))
        with torch.no_grad():
            ts = torch.linspace(0,T,n_steps+1)
            xs = torch.linspace(-5, 5, x_steps+1)
            input_t = torch.zeros(1, 1, device=device)
            input_x = torch.zeros(1, d, device=device)

        est_solution = []
        for t in ts:
            input_t[0,0] = t
            for x in xs:
                input_x[0,0] = x
                est_solution.append(model.net_dgm(input_t, input_x).detach().cpu().numpy())

        est_solution = np.reshape(est_solution, (n_steps+1, x_steps+1))

        ts = ts.cpu().detach().numpy()
        xs = xs.cpu().detach().numpy()
        t, x = np.meshgrid(ts, xs, indexing='ij')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_zlim([0, 1])
        ax.set_title('1-D Heat Equation')
        ax.set_xlabel('Time')
        ax.set_ylabel('x')
        ax.set_zlabel('value')
        ax.plot_surface(t,x, est_solution, cmap=cm.RdYlBu_r, edgecolor='blue',linewidth=0.0003, antialiased=True)
        fig.savefig(
            os.path.join(base_dir, 'demo.png')
        )   
        plt.close(fig)

    if d == 2:
        with torch.no_grad():
            ts = torch.linspace(0,T,n_steps+1)
            x0 = torch.linspace(-5, 5, x_steps+1)
            x1 = torch.linspace(-5, 5, x_steps+1)
            X0, X1 = torch.meshgrid([x0, x1])
            X = torch.cat([X0.reshape(-1,1), X1.reshape(-1,1)],1)
            t_coarse = ts[::n_steps//50]
            X = X.unsqueeze(1).repeat(1,len(t_coarse),1)
            t = t_coarse.reshape(-1,1).repeat(X.shape[0],1)
            X = X.reshape(-1,2)

        est_solution = model.net_dgm(t, X).detach().cpu().numpy()
        t = t.cpu().detach().numpy()
        X = X.cpu().detach().numpy()
        x0 = x0.cpu().detach().numpy()
        x1 = x1.cpu().detach().numpy()
        x0_grid, x1_grid = np.meshgrid(x0, x1, indexing='ij')
        frn = len(t_coarse)

        df = pd.DataFrame({
            "time": t[:,0], 
            "x0": X[:,0],
            "x1": X[:,1],
            "value": est_solution[:,0]
        })
        value = np.zeros((x_steps+1, x_steps+1, frn))

        for idx, t in enumerate(t_coarse):
            data = df[np.round(df['time'],2) == np.round(float(t),2)]
            value[:,:,idx] = np.array(data.value).reshape(x_steps+1, x_steps+1)

        def update_graph(num, value, plot):
            plot[0].remove()
            plot[0] = ax.plot_surface(x0_grid, x1_grid, value[:,:,num], cmap=cm.RdYlBu_r, edgecolor='blue',linewidth=0.0003, antialiased=True)
            ax.set_title('2-D Heat Equation, with time = {}'.format(num * T/(frn-1)))

        
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_zlim([0, 1])
        ax.set_title('2-D Heat Equation, with time = 0')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('value')

        plot = [ax.plot_surface(x0_grid, x1_grid, value[:,:,0], cmap=cm.RdYlBu_r, edgecolor='blue',linewidth=0.0003, antialiased=True)]
        ani = animation.FuncAnimation(fig, update_graph, frn, fargs=(value, plot), interval=400)
        ani.save(os.path.join(base_dir, 'demo.gif'), dpi=80)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--hidden_dim', default=20, type=int, help="hidden sizes of ffn networks approximations")
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    results_path = os.path.join(args.base_dir, "{}D_Heat_Equation".format(args.d), "DGM")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(
        T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        hidden_dim=args.hidden_dim,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        base_dir=results_path,
        device=device,
    )

    if args.d == 1:
        x_steps = 1000

    elif args.d == 2:
        x_steps = 200
    
    visualise(
        T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        hidden_dim=args.hidden_dim,
        base_dir=results_path,
        device=device,
        x_steps=x_steps,
    )
