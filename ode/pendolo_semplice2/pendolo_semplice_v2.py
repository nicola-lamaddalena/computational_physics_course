import numpy as np
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

L, theta0, omega0, t1, t2, h, g, m = 1, 0.5, 0.0, 20, 500, 0.01, 9.81, 1
times = np.arange(0, t1, h)
long_times = np.arange(0, t2, h)
hs = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
t3 = 10

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

    return np.array(pos), np.array(vel)

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

    return np.array(pos), np.array(vel)

def positions(ver_pos: list[float], rk4_pos: list[float], times: list[float], color1: str, color2: str, label1: str, label2: str, xlabel: str, ylabel: str, title: str):
    plt.plot(times, ver_pos, color=color1, label=label1)
    plt.plot(times, rk4_pos, "--", color=color2, label=label2)
    plt.xlabel(r"$t\ [s]$")
    plt.ylabel(r"$\theta(t)\ [rad]$")
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.show()

def convergenza(f1: callable, f2: callable, hs: list[float], t: float) -> None:
    t10 = np.arange(0, 10, 0.0001)
    rk4_pos, _ = rk4(theta0=theta0, h=0.0001, times=t10) 
    camp = rk4_pos[-1]
    ver_p, rk4_p = [], []
    for h in hs:
        steps = int(t / h)
        times = np.zeros(steps)
        for t in times:
            pos1, _ = f1(theta0=theta0, h=h, times=times)
            pos2, _ = f2(theta0=theta0, h=h, times=times)

        ver_p.append(np.mean(pos1)-camp)
        rk4_p.append(np.mean(pos2)-camp)
    plt.plot(hs, ver_p, label=str(h))
    plt.plot(hs, rk4_p, label=str(h))
    plt.xscale("log")
    plt.xscale("log")

    plt.legend()
    plt.show()

def main():
    ver_pos, ver_vel = vel_verlet(theta0=theta0, h=h, times=times)
    rk4_pos, rk4_vel = rk4(theta0=theta0, h=h, times=times)
    l_ver_pos, l_ver_vel = vel_verlet(theta0=theta0, h=h, times=long_times)
    l_rk4_pos, l_rk4_vel = rk4(theta0=theta0, h=h, times=long_times)
    ver_energy = 0.5 * m * L**2 * l_ver_vel**2 + m * g * L * (1 - np.cos(l_ver_pos))
    rk4_energy = 0.5 * m * L**2 * l_rk4_vel**2 + m * g * L * (1 - np.cos(l_rk4_pos))
    energy = [0.5 * m * L**2 * omega0**2 + m * g * L * (1 - np.cos(theta0)) for _ in long_times]
    delta_ver_energy = abs(ver_energy - energy)
    delta_rk4_energy = abs(rk4_energy - energy)

    #positions(ver_pos=ver_pos, rk4_pos=rk4_pos, times=times, color1="red", color2="blue", label1="Velocity Verlet", label2="Runge-Kutta 4", xlabel=r"$t [s]$", ylabel=r"$\theta(t)\ [rad]$", title="Velocity Verlet vs Runge-Kutta 4 con t=20 s e h=0.01 s")
    #positions(ver_pos=l_ver_pos, rk4_pos=l_rk4_pos, times=long_times, color1="red", color2="blue", label1="Velocity Verlet", label2="Runge-Kutta 4", xlabel=r"$t [s]$", ylabel=r"$\theta(t)\ [rad]$", title="Velocity Verlet vs Runge-Kutta 4 con t=500 s e h=0.01 s")
    #plt.plot(l_ver_pos, l_ver_vel, color="red")
    #plt.plot(l_rk4_pos, l_rk4_vel, color="blue")
    #plt.xlabel(r"$\theta\ [rad]$")
    #plt.ylabel(r"$\omega\ [rad/s]$")
    #plt.title("RK4: spazio delle fasi")
    #plt.grid()
    #plt.show()
    convergenza(vel_verlet, rk4, hs, t3)
if __name__ == "__main__":
    main()
