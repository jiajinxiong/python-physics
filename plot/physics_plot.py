import numpy as np
import matplotlib.pyplot as plt

class Schrodinger (object):
    def __init__(self,potensial_fun,x_min, x_max , N , hbar=1,mass = 1):

        self.x = np.linspace(x_min,x_max,N)
        self.U = np.diag(potensial_fun(self.x),0)               # 将势能离散化
        self.Lap = self.laplace(N)
        self.H = - hbar ** 2/(2*mass)*self.Lap + self.U
        self.eigE,self.eigV = self.eig_solve()

    def laplace(self,N):
        '''构造二阶微分算子 : Laplacian'''
        dx = self.x[1]-self.x[0]
        return (-2*np.diag(np.ones(N,np.float32),0) + np.diag(np.ones(N-1,np.float32),1)
                + np.diag(np.ones(N-1,np.float32),-1)
                )/(dx**2)

    def eig_solve(self):
        w,v = np.linalg.eig(self.H)
        idx_sorted = np.argsort(w)
        return w[idx_sorted],v[:,idx_sorted]

    def eig_value(self,n):
        return format(self.eigE[n],'.0f')

    def eig_vector(self,n):
        dx = self.x[1] - self.x[0]
        norm_factor = np.sqrt(1/(np.sum(self.eigV[:,n]**2)*dx))

        return self.eigV[:,n]*norm_factor

    def plot_potential(self):
        with plt.style.context(['science','ieee']):
            # plt.figure(figsize=(5, 4), dpi=150)
            plt.plot(self.x,np.diag(self.U))

            plt.xlabel(r'$x$')
            plt.ylabel(r'$potential$')

    def plot_wave_fun(self,n):
        with plt.style.context(['science','ieee']):
            xx=self.x
            y = self.eig_vector(n)
            # plt.figure(figsize=(5, 4), dpi=150)
            plt.plot(xx,y)

            plt.xlabel('$x$')
            plt.ylabel(r'$\Psi(x)$')
            plt.legend([np.float16(self.eigE[n])])
    def plot_density_fun(self,n):
        '''画概率密度函数\rho = \psi^* \psi'''
        with plt.style.context(['science','ieee']):
            psi = self.eig_vector(n)
            rho = psi * psi
            plt.plot(self.x,rho)
            plt.xlabel('$x$')
            plt.ylabel(r'$\rho_{%s}(x)=\psi_{%s}^*\psi_{%s}$'%(n,n,n))
            plt.title(r'$E_{%s}=%.2f$'%(n,self.eig_value(n)))


class Update(object):
    def __init__(self,ax,class_func,*coefficient):
        '''

        :param ax:
        :param class_func:
        :param coefficient:输入叠加系数，与叠加态
        '''
        self.ax = ax
        self.line, = ax.plot([],[],'k-')
        self.x = class_func.x
        # self.ax.set_ylim(0,1.1)
        # self.psit = psit
        self.func = class_func
        self.states = coefficient[1]
        self.sup = coefficient[0]

    def psit(self, t):
        super_coefficient = self.sup
        states = self.states
        func = self.func
        num = len(states)
        psi = (func.eig_vector(states[0]) * np.exp(-1j * func.eig_value(states[0]) * t)) * super_coefficient[0]
        norm = np.max(func.eig_vector(states[0]))
        for i in range(1, num):
            psi += (func.eig_vector(states[i]) * np.exp(-1j * func.eig_value(states[i]) * t)) * super_coefficient[i]
            norm += np.max(func.eig_vector(states[i]))
        return psi,norm

    def __call__(self, i):
        time = i * 0.01
        psi,max = self.psit(time)
        density = np.real(np.conjugate(psi) * psi)
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        self.ax.set_xlim(xmin, xmax)

        self.ax.set_ylim(0,max)
        self.line.set_data(self.x, density)
        plt.legend(('t = %s'%format(time,'.2f'),),loc=1)

        return self.line,
