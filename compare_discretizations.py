# Visualize the discretization sparsity pattern

from __future__ import annotations

import pickle
import numpy as np
from numpy import linalg as la
from scipy import integrate as spi
from typing import Callable, List, Optional, Tuple
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

choice = "CGL_seg"  # ZOH | RK4 | CGL | CGL_seg
system = "2ndOrder"  # 1stOrder | 2ndOrder

# ..:: Define the continuous-time dynamical system ::..

if system == "1stOrder":
    alpha = 1.0  # [s] Time constant
    nx = 1  # Number of states
    nu = 1  # Number of inputs
    A = np.array([[-alpha]])
    B = np.array([[1.0]])
    x0 = np.array([1.0])  # Initial condition
elif system == "2ndOrder":
    m = 1.0  # [kg] Mass
    wn = 2.0  # [rad/s] Natural frequency
    zeta = 0.2  # Damping ratio
    nx = 2  # Number of states
    nu = 1  # Number of inputs
    A = np.array([[0, 1], [-wn**2, -2*zeta*wn]])
    B = np.array([[0], [1/m]])
    x0 = np.array([0.0, 0.0])  # Initial condition

Ts = 0.2  # [s] Discretization time interval
Tsim = 10.0  # [s] Simulation duration
Ntsol = 100  # Number of points to store the solution
machine_eps = np.finfo(float).eps  # Machine precision
down_step_time = (1.0//Ts)*Ts  # Time of input step end

if choice == "CGL_seg":
    t_knot = down_step_time  # [s] Time at which to split collocation segments

def input(t: np.float) -> np.float:
    """Input to the system."""
    step = t <= down_step_time-np.sqrt(machine_eps)
    u = np.array([1.0]) if step else np.array([0.0])
    return u


def dynamics_ct(t: np.float, x: np.ndarray) -> np.ndarray:
    """Compute dynamics time derivative."""
    x = x.flatten()
    u = input(t)
    x_dot = A@x+B@u
    return x_dot


# ..:: Integration routines ::..

rtol = 1e-12  # Relative integration accuracy tolerance
atol = 1e-12  # Absolute integration accuracy tolerance


def integrate_final(f: Callable[[np.float, np.ndarray], np.ndarray],
                    x0: np.ndarray, tf: np.float) -> np.ndarray:
    """Integrate an ODE, return the final value."""
    sol = spi.solve_ivp(f, (0, tf), x0, vectorized=True, rtol=rtol, atol=atol)
    Xf = sol.y[:, -1]
    return Xf


def integrate_sys_ct(f: Callable[[np.float, np.ndarray], np.ndarray],
                     x0: np.ndarray, tspan: np.ndarray) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
    """Integrate an ODE, return time history."""
    sol = spi.solve_ivp(f, (0, tspan[-1]), x0, t_eval=tspan, vectorized=True,
                        rtol=rtol, atol=atol)
    return sol.t, sol.y


def integrate_sys_dt(f: Callable[[int, np.ndarray], np.ndarray],
                     x0: np.ndarray, tf: np.float) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    """Compute a different equation, return time history."""
    t = np.linspace(0, tf, round(tf/Ts+1))
    x = np.empty((nx, t.size))
    x[:, 0] = x0
    for k in range(0, t.size-1):
        x[:, k+1] = f(k, x[:, k])
    return t, x


# ..:: Perform discretization ::..

def F(t: np.float, x: np.ndarray) -> np.ndarray:
    """ODE with vector state. Return time derivative (vector)."""
    x = x.flatten()
    # ..:: Convert vector to matrix components ::..
    Phi, _ = extract(x)
    # ..:: Compute time derivatives ::..
    Phi_dot = A@Phi
    Btilde_dot = la.solve(Phi, B)
    # ..:: Concatenate and return ::..
    x_dot = vectorize((Phi_dot, Btilde_dot))
    return x_dot


def extract(x: np.ndarray) -> Tuple[np.ndarray,
                                    np.ndarray]:
    """Convert vectorized form to matrix form."""
    Phi = np.reshape(x[:nx**2], (nx, nx))
    Btilde = np.reshape(x[nx**2:], (nx, nu))
    return Phi, Btilde


def vectorize(matrices: Tuple[np.ndarray,
                              np.ndarray]) -> np.ndarray:
    """Convert matrices to vectorized form."""
    x = np.concatenate([np.reshape(M, -1) for M in matrices])
    return x


def cgl_nodes(N: int) -> np.ndarray:
    """CGL discretization nodes."""
    return -np.cos(np.pi*np.arange(N)/(N-1))


def make_differentiation_matrix(Ng: int, eta: np.array) -> np.ndarray:
    """Make CGL differentiation matrix."""
    def c(j):
        return 2.0 if (j == 0 or j == Ng-1) else 1.0
    D = np.zeros((Ng, Ng))
    for j in range(Ng):
        for i in range(Ng):
            if j != i:
                D[j, i] = (c(j)/c(i))*(-1.0)**(i+j)/(eta[j]-eta[i])
            elif j == i:
                if j == 0:
                    D[j, i] = -(2*(Ng-1)**2+1)/6
                elif j == Ng-1:
                    D[j, i] = (2*(Ng-1)**2+1)/6
                else:
                    D[j, i] = -eta[j]/(2*(1-eta[j]**2))
    return D


def combine_segments(M_seg: List[np.ndarray],
                     shift: Optional[int] = 0) -> np.ndarray:
    """Combine differentiation matrices of individual segments into one."""
    M = M_seg[0]
    for i in range(1, len(M_seg)):
        O1 = np.zeros((M.shape[0], M_seg[i].shape[1]-shift))
        O2 = np.zeros((M_seg[i].shape[0], M.shape[1]-shift))
        M = np.block([[M, O1], [O2, M_seg[i]]])
    return M


if choice == "ZOH":
    Phi_0 = np.eye(nx)
    Btilde_0 = np.zeros((nx, nu))
    X0 = vectorize((Phi_0, Btilde_0))
    Xf = integrate_final(F, X0, Ts)
    Phi_f, Btilde_f = extract(Xf)
    Ak = Phi_f
    Bk = Ak@Btilde_f
elif choice == "RK4":
    # Matrix coefficient of xk
    k1 = A
    k2 = A@(np.eye(nx)+Ts/2*k1)
    k3 = A@(np.eye(nx)+Ts/2*k2)
    k4 = A@(np.eye(nx)+Ts*k3)
    Ak = np.eye(nx)+Ts/6*(k1+2*k2+2*k3+k4)
    # Matrix coefficient of uk
    k1 = B
    k2 = Ts/2*A@k1+B/2
    k3 = Ts/2*A@k2+B/2
    k4 = Ts*A@k3
    Bkm = Ts/6*(k1+2*k2+2*k3+k4)
    # Matrix coefficient of u(k+1)
    k2 = B/2
    k3 = Ts/2*A@k2+B/2
    k4 = Ts*A@k3+B
    Bkp = Ts/6*(2*k2+2*k3+k4)
elif choice == "CGL":
    Ng = round(Tsim/Ts+1)  # Number of grid nodes
    eta = cgl_nodes(Ng)
    D = make_differentiation_matrix(Ng, eta)
elif choice == "CGL_seg":
    # Two segments: one during, and one after the input step
    num_seg = 2
    Ng_seg = [round(t_knot/Ts+1), round((Tsim-t_knot)/Ts+1)]
    eta_seg = [cgl_nodes(Ng_seg[i]) for i in range(num_seg)]
    print(eta_seg)
    D_seg = [make_differentiation_matrix(Ng_seg[i], eta_seg[i])
             for i in range(num_seg)]


def dynamics_dt(k: int, xk: np.ndarray) -> np.ndarray:
    """Compute discrete-time update"""
    tk = k*Ts
    if choice == "ZOH":
        uk = input(tk)
        x_kp1 = Ak@xk+Bk@uk
    elif choice == "RK4":
        tkp1 = (k+1)*Ts
        uk = input(tk)
        ukp1 = input(tkp1)
        x_kp1 = Ak@xk+Bkm@uk+Bkp@ukp1
    return x_kp1


def lagrange_poly(i: int, t: np.float, eta: np.ndarray) -> np.float:
    """Lagrange interpolating polynomial value at pseudo time eta in [-1,1]."""
    value = np.prod([(t-eta[j])/(eta[i]-eta[j])
                     for j in range(len(eta)) if j != i])

    return value


def cgl_trajectory(x: np.ndarray, tspan: np.ndarray) -> np.ndarray:
    """Collocated trajectory using Chebyshev-Gauss-Lobatto."""
    x_cgl = np.zeros((nx, tspan.size))
    if choice == "CGL":
        for k in range(tspan.size):
            tk = tspan[k]
            for i in range(Ng):
                x_cgl[:, k] += x[:, i]*lagrange_poly(i, 2*tk/Tsim-1, eta)
    elif choice == "CGL_seg":
        for k in range(tspan.size):
            tk = tspan[k]
            j = 0 if tk <= t_knot else 1
            _Ng = Ng_seg[j]
            _eta = eta_seg[j]
            offset = 0 if j == 0 else Ng_seg[0]-1
            T = t_knot if j == 0 else Tsim-t_knot
            t0 = 0 if j == 0 else t_knot
            for i in range(_Ng):
                x_cgl[:, k] += x[:, offset+i]*lagrange_poly(
                    i, 2*(tk-t0)/T-1, _eta)
    return x_cgl


# ..:: Simulate continuous- and discrete-time systems ::..

# >> Continuous-time <<
times = np.linspace(0, Tsim, Ntsol)
t_c, x_c = integrate_sys_ct(dynamics_ct, x0, times)
if system == "1stOrder":
    p_c = x_c[0]
elif system == "2ndOrder":
    p_c, v_c = x_c

# >> Discrete-time <<
if choice == "ZOH" or choice == "RK4":
    t_d, x_d = integrate_sys_dt(dynamics_dt, x0, Tsim)
elif choice == "CGL" or choice == "CGL_seg":
    # "Solve" the approximated dynamics
    if choice == "CGL":
        Bbar = np.kron(np.eye(Ng), B)
        t_d = Tsim*(eta+1)/2
        Abar = np.kron(np.eye(Ng), A)
        Dbar = np.kron(2/Tsim*D, np.eye(nx))
        U = np.concatenate([input(tk) for tk in t_d])
    else:
        Bbar = combine_segments([
            np.kron(np.eye(Ng_seg[i]), B) for i in range(num_seg)],
                                shift=0)
        t_d = np.concatenate([
            t_knot*(eta_seg[0]+1)/2,
            (Tsim-t_knot)*(eta_seg[1][1:]+1)/2+t_knot])
        Abar = combine_segments([np.kron(np.eye(Ng_seg[i]), A)
                                 for i in range(num_seg)], shift=nx)
        Dbar = combine_segments([
            np.kron(2/t_knot*D_seg[0], np.eye(nx)),
            np.kron(2/(Tsim-t_knot)*D_seg[1], np.eye(nx))], shift=nx)
        U0 = np.array([input(t) for t in t_knot*(eta_seg[0]+1)/2])
        U0[-1] = input(down_step_time-np.sqrt(machine_eps))
        U1 = np.array([input(t) for t in
                       (Tsim-t_knot)*(eta_seg[1]+1)/2+t_knot])
        U = np.concatenate([U0, U1]).flatten()
    F = Dbar-Abar
    F0 = F[:, :nx]
    F1 = F[:, nx:]
    R = -F0@x0+Bbar@U
    X = la.pinv(F1)@R
    X = np.concatenate([x0, X])
    x_d = np.vstack([X[nx*i:nx*(i+1)] for i in range(Ng)]).T
    # Collocate back into continuous time
    t_d = np.linspace(0, Tsim, Ntsol)
    x_d = cgl_trajectory(x_d, t_d)

if system == "1stOrder":
    p_d = x_d[0]
    p_cd = np.interp(t_d, t_c, p_c)
elif system == "2ndOrder":
    p_d, v_d = x_d
    p_cd = np.interp(t_d, t_c, p_c)
    v_cd = np.interp(t_d, t_c, p_c)

p_err = p_d-p_cd

# >> Plot <<
ct_style = dict(color="black")
if choice == "ZOH" or choice == "RK4":
    dt_style = dict(color="red", linestyle="none", marker=".", markersize=7)
    error_style = dict(color="red", linestyle="none", marker=".", markersize=7)
elif choice == "CGL" or choice == "CGL_seg":
    dt_style = dict(color="red")
    error_style = dict(color="red")

fig = plt.figure(1)
plt.clf()

# Position plot
ax = fig.add_subplot(211)
ax.plot(t_c, p_c, **ct_style)
ax.plot(t_d, p_d, **dt_style)
ax.autoscale(tight=True)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position [m]")

# Position error plot
ax = fig.add_subplot(212)
ax.plot(t_d, p_err, **error_style)
ax.autoscale(tight=True)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position error [m]")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

plt.tight_layout()

fig.savefig("simulation.png", bbox_inches='tight')

data = dict(t_c=t_c,
            y_c=p_c,
            t_d=t_d,
            y_d=p_d,
            t_err=t_d,
            y_err=p_err)
with open("sim_{0}.pkl".format(choice), "wb") as f:
    pickle.dump(data, f)

# ..:: Show sparsity pattern ::..


class BlockMatrix:
    def __init__(self,
                 N: int,
                 M: int,
                 n: Optional[int] = None,
                 m: Optional[int] = None,
                 value: Optional[np.ndarray] = None) -> np.ndarray:
        """Make an [N,M] block matrix where each block is of size [n,m]."""
        if value is not None:
            self._n = N/value.shape[0]
            self._m = M/value.shape[1]
            self._N = N
            self._M = M
            self._F = value
        else:
            self._n = n
            self._m = m
            self._N = N
            self._M = M
            self._F = np.zeros((N*n, M*m))

    def set(self, i: int, j: int, B: np.ndarray) -> None:
        """Set block (i,j) to value."""
        assert B.shape == (self._n, self._m)
        assert i < self._N and j < self._M
        rows = slice(i*self._n, (i+1)*self._n)
        cols = slice(j*self._m, (j+1)*self._m)
        self._F[rows, cols] = B

    def spy(self) -> None:
        """Print the sparsity pattern."""
        rows, cols = self._F.shape
        for i in range(rows):
            for j in range(cols):
                val = "* " if self._F[i, j] != 0 else ". "
                print(val, end="")
            print("\n", end="")

    def __sub__(self, other: BlockMatrix) -> BlockMatrix:
        """Subtraction."""
        new_value = self._F-other.value
        result = BlockMatrix(self._N, self._M, value=new_value)
        return result

    @property
    def value(self) -> np.ndarray:
        """Return matrix value"""
        return self._F.copy()


_Ng = 4  # Number of time grid points
print("{0} (F*X = G*U)".format(choice))

if choice == "ZOH" or choice == "RK4":
    # State coefficient matrix sparsity
    Eye = BlockMatrix(_Ng, _Ng, nx, nx)
    for i in range(_Ng):
        Eye.set(i, i, np.eye(nx))
    Abar = BlockMatrix(_Ng, _Ng, nx, nx)
    Abar.set(0, 0, np.eye(nx))
    for i in range(1, _Ng):
        Abar.set(i, i-1, Ak)
    F = Eye-Abar

    # State coefficient matrix sparsity
    if choice == "ZOH":
        G = BlockMatrix(_Ng, _Ng-1, nx, nu)
        for k in range(0, _Ng-1):
            G.set(k+1, k, Bk)
    elif choice == "RK4":
        G = BlockMatrix(_Ng, _Ng, nx, nu)
        for k in range(0, _Ng-1):
            G.set(k+1, k, Bkm)
            G.set(k+1, k+1, Bkp)
elif choice == "CGL":
    eta = cgl_nodes(_Ng)
    D = make_differentiation_matrix(_Ng, eta)
    Dbar = 2/Tsim*np.kron(D, np.eye(nx))
    Abar = np.kron(np.eye(_Ng), A)
    Bbar = np.kron(np.eye(_Ng), B)
    F = BlockMatrix(_Ng, _Ng, value=Dbar-Abar)
    G = BlockMatrix(_Ng, _Ng, value=Bbar)
elif choice == "CGL_seg":
    _Ng = 3
    _Ng_2 = 2
    Ng_seg = [_Ng-_Ng_2+1, _Ng_2]
    eta_seg = [cgl_nodes(Ng_seg[i]) for i in range(num_seg)]
    D_seg = [make_differentiation_matrix(Ng_seg[i], eta_seg[i])
             for i in range(num_seg)]
    Dbar = combine_segments([
        np.kron(2/t_knot*D_seg[0], np.eye(nx)),
        np.kron(2/(Tsim-t_knot)*D_seg[1], np.eye(nx))], shift=nx)
    Abar = combine_segments([np.kron(np.eye(Ng_seg[i]), A)
                             for i in range(num_seg)], shift=nx)
    Bbar = combine_segments([
        np.kron(np.eye(Ng_seg[i]), B) for i in range(num_seg)], shift=0)
    F = BlockMatrix(_Ng+1, _Ng, value=Dbar-Abar)
    G = BlockMatrix(_Ng+1, _Ng+1, value=Bbar)

print("F=")
F.spy()
print("G=")
G.spy()

data = dict(F=F.value,
            G=G.value,
            Nrow_F=_Ng if choice != "CGL_seg" else _Ng+1,
            Ncol_F=_Ng,
            Nrow_G=_Ng if choice != "CGL_seg" else _Ng+1,
            Ncol_G=((_Ng if choice != "ZOH" else _Ng-1)
                    if choice != "CGL_seg" else _Ng+1),
            split=None if choice != "CGL_seg" else Ng_seg[0],
            nx=nx,
            nu=nu)
with open("data_{0}.pkl".format(choice), "wb") as f:
    pickle.dump(data, f)
