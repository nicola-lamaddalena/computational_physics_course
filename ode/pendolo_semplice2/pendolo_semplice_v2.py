import numpy as np
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

L, theta0, t1, t2, h, g = 1, 0.5, 20, 500, 0.01, 9.81
times = np.arange(0, t1, h)
long_times = np.arange(0, t2, h)

def vel_verlet(theta0: float, h: float, times: list[float]) -> tuple[list[float], list[float]]:
    omega0 = 0
    a0 = -(g/L)*np.sin(theta0)
    pos, vel = [theta0], [omega0]

    for _ in times[:-1]:
        theta = theta0 + h*omega0 + 0.5 * h**2 * a0
        a = -(g/L)*np.sin(theta)
        omega = omega0 + 0.5 * h * (a0 + a)

        pos.append(theta)
        vel.append(omega)
        a0, theta0, omega0 = a, theta, omega

    return pos, vel

def rk4(theta0: float, h: float, times: list[float]) -> tuple[list[float], list[float]]:
    omega0 = 0
    pos, vel = [theta0], [omega0]

    for _ in times[:-1]:
        k1t, k1o = omega0, -(g/L) * np.sin(theta0)
        k2t, k2o = omega0 + 0.5 * h * k1o, -(g/L) * np.sin(theta0 + 0.5 * h * k1t)
        k3t, k3o = omega0 + 0.5 * h * k2o, -(g/L) * np.sin(theta0 + 0.5 * h * k2t)
        k4t, k4o = omega0 + h * k3o, -(g/L) * np.sin(theta0 + h * k3t)

        theta = theta0 + (h/6) * (k1t + 2*k2t + 2*k3t + k4t)
        omega = omega0 + (h/6) * (k1o + 2*k2o + 2*k3o + k4o)
        pos.append(theta)
        vel.append(omega)
        theta0, omega0 = theta, omega

    return pos, vel

def positions(ver_pos: list[float], rk4_pos: list[float], times: list[float], color1: str, color2: str, label1: str, label2: str, xlabel: str, ylabel: str, title: str):
    plt.plot(times, ver_pos, color=color1, label=label1)
    plt.plot(times, rk4_pos, "--", color=color2, label=label2)
    plt.xlabel(r"$t\ [s]$")
    plt.ylabel(r"$\theta(t)\ [rad]$")
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.show()



def main():
    ver_pos, ver_vel = vel_verlet(theta0=theta0, h=h, times=times)
    rk4_pos, rk4_vel = rk4(theta0=theta0, h=h, times=times)

    positions(ver_pos=ver_pos, rk4_pos=rk4_pos, times=times, color1="red", color2="blue", label1="Velocity Verlet", label2="Runge-Kutta 4", xlabel=r"$t [s]$", ylabel=r"$\theta(t)\ [rad]$", title="Velocity Verlet vs Runge-Kutta 4 con t=20 s e h=0.01 s")

    
    l_ver_pos, l_ver_vel = vel_verlet(theta0=theta0, h=h, times=long_times)
    l_rk4_pos, l_rk4_vel = rk4(theta0=theta0, h=h, times=long_times)
    positions(ver_pos=l_ver_pos, rk4_pos=l_rk4_pos, times=long_times, color1="red", color2="blue", label1="Velocity Verlet", label2="Runge-Kutta 4", xlabel=r"$t [s]$", ylabel=r"$\theta(t)\ [rad]$", title="Velocity Verlet vs Runge-Kutta 4 con t=20 s e h=0.01 s")
if __name__ == "__main__":
    main()
