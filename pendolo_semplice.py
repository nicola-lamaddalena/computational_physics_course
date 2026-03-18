import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

"M=l times Fg"
"M=l mg sin(theta), con verso entrante in O"
"L=l times mv -> L=lmv=lml thetapunto=l^2 m thetapunto con verso uscente"
"M=dL/dt -> M=-dl^2 m thetapunto/dt -> l m g sin(theta)=-l^2 m thetaduepunti"
"thetaduepunti=-g sin(theta)/l, II ordine"
"passando al primo ordine: dtheta/dt=omega e domega/dt=-g sin(theta)/l"
"energia meccanica totale (rispetto al punto più basso): z=l-lcos(theta)=l(1-cos(theta))"
"E=ml^2v^2/2+mgl(1-cos(theta))"
"approssimazione picoole oscillazioni: sin(theta)=theta, frequenza angolare: omega0=sqrt(g/l) e T=2pi sqrt(l/g)"
"theta(t)=theta0 cos(omega0 t)+(omega_in/omega0)sin(omega0 t)"

def eulero(x0: float, dt: float, times: list[float], f: callable) -> list[float]:
    points = [x0]
    for t in times:
        f = f(x0, t)
        xn = x0 + dt * f
        points.append(xn)
        x0 = xn

    return points

def eulero_cromer(x0: float, v0: float, dt: float, times: list[float], f: callable, g: callable) -> tuple[list[float], list[float]]:
    pos, vel = [], []
    for t in times:
        xn = x0 + dt*f(t, x0, v0)
        vn = v0 + dt*g(t, xn, v0)
        pos.append(xn)
        vel.append(vn)
        x0, v0 = xn, vn
    return pos, vel

def rk4(x0: float, dt: float, times: list[float], f: callable) -> list[float]:
    points = []
    for t in times:
        f1 = f(t, x0)
        f2 = f(t + dt/2, x0 + f1*dt/2)
        f3 = f(t + dt/2, x0 + f2*dt/2)
        f4 = f(t + dt, x0 + h*f3)

        xn = x0 + dt/6 * (f1 + 2*f2 + 2*f3 + f4)
        points.append(xn)
        x0 = xn

def verlet(x0: float, a: callable, dt: float, times: list[float]) -> tuple[list[float], list[float]]:
    pos = []
    vel = []
    x1 = eulero(x0, dt, [1], a(1, x0))
    pos.append(x0)
    pos.append(x1)

    for t in times:
        xn = 2*pos[-1] + pos[-2] + dt**2 * a(t, pos[-1])
        vn = (xn - pos[-2])/(2*dt)
        pos.append(xn)
        vel.append(vn)
    
    return pos, vel

def f_analitica(theta0: float, t: float, omega_in: float = 0.0) -> list[float]:
    g = 9.81
    l = 1
    omega0 = (g/l)**0.5
    theta = theta0 + np.cos(omega0 * t) + (omega_in/omega0)*np.sin(omega0*t)
    return theta

x0 = 0.1
times = np.arange(0, 20, 0.01)
plt.plot(f_analitica(x0, times))
plt.plot(eulero(x0=x0, dt=0.01, times=times, f=f_analitica))
plt.show()
