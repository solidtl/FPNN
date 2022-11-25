import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.ksLoader import KSLoader

'''FSM for physical control constraints of KS system '''

class KS_FFT():
    def __init__(self,L,tau,J):
        self.init(L,tau,J)

    def init(self,L,tau,N):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tau = tau  #time step
        I = complex(0, 1)
        k = 2 * np.pi / L * np.array(
            [I * n for n in range(0, int(N / 2))] + [0] + [I * n for n in range(-int(N / 2) + 1, 0)])
        self.k = torch.tensor(k).to(device)
        self.k2 = self.k ** 2
        self.k4 = self.k ** 4

        self.k = self.k.to(device)
        self.k2 = self.k2.to(device)
        self.k4 = self.k4.to(device)


    def forward(self,v):
        v_hat = torch.fft.fft(v)  # convert to fourier space
        # v_hat = v_hat-0.5*k*tau*np.fft.fft(v**2)-k2*tau*v_hat-k4*tau*v_hat
        # backward Euler timestepping
        v_hat = (v_hat - 0.5*self.k * self.tau * torch.fft.fft(v ** 2)) / (1 + self.k2 * self.tau + self.k4 * self.tau)
        # FE timestepping
        # v_hat = v_hat + tau * alpha * k2 * v_hat
        # Crank-Niclson
        # v_hat =((1-0.5*(self.k2+self.k4)* self.tau) *v_hat-0.5*self.k*torch.fft.fft(v ** 2)* self.tau)/\
        #        (1+(self.k2+self.k4)*self.tau)
        v = torch.fft.ifft(v_hat)  # convert back to real space
        return torch.real(v)

if __name__ =="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = KS_FFT(22*torch.pi,0.1,96)
    data = np.zeros((1000, 96))
    N = 96
    L = 22 * torch.pi
    x, h = np.linspace(0, L, N, retstep=True)

    # Initial conditions
    # v = np.cos(x / 16) * (1 + np.sin(x / 16))
    # v = torch.tensor(v)
    data_dir = "../solver/validation_data_demo"
    ksLoader = KSLoader()
    test_cases = np.arange(1, 6, 1).astype(int)  # Some validation/test data
    test_loader = ksLoader.createTestingLoader(data_dir, test_cases, batch_size=5)
    for i, (input0, uTarget) in enumerate(test_loader):
        v = input0[:,-1,:].to(device)
    print(v.shape)

    alpha = 0.5
    tmax = 100
    tau = 0.1
    nplots = int(round(tmax / tau))
    t = np.linspace(0, tmax, int(tmax / tau))

    data = torch.zeros((5,len(t),N))

    # v = v.repeat((10,1))
    for i in range(len(t)):
        v = model.forward(v)
        data[:,i,:] = v
    xx, tt = np.meshgrid(x, t)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx, tt, data[3], cmap=plt.get_cmap('rainbow'))
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()
