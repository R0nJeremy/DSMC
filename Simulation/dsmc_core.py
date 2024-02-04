import numpy as np
from pydantic import BaseModel, computed_field, PositiveFloat, PositiveInt
import matplotlib.pyplot as plt

#import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Serif"})
plt.rcParams.update({"font.size": 18})


class ParticleParameters(BaseModel):
    radius: PositiveFloat
    rho: PositiveFloat
    eps: PositiveFloat

    @computed_field
    @property
    def mass(self) -> float:
        return (4 * np.pi / 3) * self.rho * self.radius**3


class WorldParameters(BaseModel):
    N: PositiveInt
    n: PositiveFloat
    Nx: PositiveInt
    Ny: PositiveInt
    Nz: PositiveInt
    T0: PositiveFloat

    @computed_field
    @property
    def V(self) -> float:
        return self.N / self.n

    @computed_field
    @property
    def V_box(self) -> float:
        return self.V / (self.Nx * self.Ny * self.Nz)

    @computed_field
    @property
    def L_box(self) -> float:
        return np.power(self.V_box, 1 / 3)

    @computed_field
    @property
    def Lx(self) -> float:
        return self.Nx * self.L_box

    @computed_field
    @property
    def Ly(self) -> float:
        return self.Ny * self.L_box

    @computed_field
    @property
    def Lz(self) -> float:
        return self.Nz * self.L_box


class DSMC_Core:
    def __init__(self) -> None:
        self.pp = ParticleParameters(
            radius=0.05,
            rho=900,
            eps=0.6,
        )
        self.wp = WorldParameters(
            N=100_000,
            n=0.05,
            Nx=20,
            Ny=15,
            Nz=5,
            T0=24.0,
        )

    @property
    def temperature(self) -> float:
        v_T = (self.vx**2 + self.vy**2 + self.vz**2).mean()
        return (self.pp.mass / 3) * v_T

    def initialize(self) -> None:
        """
        Initializes all the parameters of the simulation, after the
        Particle and World constants are given
        """
        # particles' positions are uniformly distributed
        self.rx = np.random.default_rng().uniform(0, self.wp.Lx, self.wp.N)
        self.ry = np.random.default_rng().uniform(0, self.wp.Ly, self.wp.N)
        self.rz = np.random.default_rng().uniform(0, self.wp.Lz, self.wp.N)
        # particles' velocities are normally distributed with T0
        self.vx = np.random.default_rng().normal(
            0, np.sqrt(self.wp.T0 / self.pp.mass), self.wp.N
        )
        self.vy = np.random.default_rng().normal(
            0, np.sqrt(self.wp.T0 / self.pp.mass), self.wp.N
        )
        self.vz = np.random.default_rng().normal(
            0, np.sqrt(self.wp.T0 / self.pp.mass), self.wp.N
        )
        # remove the artifact mean velocity
        self.vx -= self.vx.mean()
        self.vy -= self.vy.mean()
        self.vz -= self.vz.mean()
        # data structures
        self.box_particles = np.ndarray(
            (self.wp.Nx, self.wp.Ny, self.wp.Nz), dtype=object
        )
        self.box_vmax = np.ndarray(
            (self.wp.Nx, self.wp.Ny, self.wp.Nz), dtype=np.float64
        )
        self.box_nc_error = np.ndarray(
            (self.wp.Nx, self.wp.Ny, self.wp.Nz), dtype=np.float64
        )
        self.box_nc_error.fill(0.0)

        # simulation characteristic constants and variables
        self.t = 0.0  # simulation time
        self.dt = 0.1  # simulation time-step
        self.nc = 0  # cumulative number of collisions

        self.T0 = self.temperature  # actual initial temperature
        self.C_vmax = 5  # ratio of the maximal speed over thermal speed
        # DSMC constant
        self.C_ncoll = 4 * np.pi * self.pp.radius**2 / self.wp.V_box
        # Charateristic inverse time
        self.t_c_inv = (
            8
            * self.wp.n
            * self.pp.radius**2
            * np.sqrt(np.pi * self.T0 / self.pp.mass)
        )

        self.fill_data_structures()

    def fill_data_structures(self) -> None:
        for idx, _ in np.ndenumerate(self.box_particles):
            self.box_particles[idx] = np.ndarray((0), dtype=np.int64)

        for p_idx in range(self.wp.N):
            i = (self.rx[p_idx] / self.wp.L_box).astype(int)
            j = (self.ry[p_idx] / self.wp.L_box).astype(int)
            k = (self.rz[p_idx] / self.wp.L_box).astype(int)
            self.box_particles[i, j, k] = np.append(self.box_particles[i, j, k], p_idx)

        for idx, p_list in np.ndenumerate(self.box_particles):
            if p_list.size == 0:
                temperature_box = 0.0
            else:
                v_T = (
                    self.vx[p_list] ** 2 + self.vy[p_list] ** 2 + self.vz[p_list] ** 2
                ).mean()
                temperature_box = (self.pp.mass / 3) * v_T
            v_thermal = np.sqrt(2 * temperature_box / self.pp.mass)
            self.box_vmax[idx] = self.C_vmax * v_thermal

    def translations(self) -> None:
        self.rx = (self.rx + self.vx * self.dt + self.wp.Lx) % self.wp.Lx
        self.ry = (self.ry + self.vy * self.dt + self.wp.Ly) % self.wp.Ly
        self.rz = (self.rz + self.vz * self.dt + self.wp.Lz) % self.wp.Lz
        self.fill_data_structures()

    def binary_collision(self, p1: int, p2: int, vmax: float) -> int:
        # relative velocity
        dvx = self.vx[p1] - self.vx[p2]
        dvy = self.vy[p1] - self.vy[p2]
        dvz = self.vz[p1] - self.vz[p2]
        # generate a random unit vector for the collision geometry
        phi = np.random.default_rng().uniform(0, 2 * np.pi)
        costheta = np.random.default_rng().uniform(-1, 1)
        sintheta = np.sqrt(1 - costheta**2)
        ndx = np.cos(phi) * sintheta
        ndy = np.sin(phi) * sintheta
        ndz = costheta
        dv_norm = dvx * ndx + dvy * ndy + dvz * ndz
        r = np.random.default_rng().uniform(0, 1)

        if np.fabs(dv_norm) < r * vmax:
            return 0
        else:
            h = (1 + self.pp.eps) * dv_norm / 2
            self.vx[p1] -= h * ndx
            self.vy[p1] -= h * ndy
            self.vz[p1] -= h * ndz
            self.vx[p2] += h * ndx
            self.vy[p2] += h * ndy
            self.vz[p2] += h * ndz
            return 1

    def collisions(self) -> None:
        n_collisions = 0
        for idx, p_list in np.ndenumerate(self.box_particles):
            n_box_particles = p_list.size
            if n_box_particles > 1:
                Nc_float = (
                    self.C_ncoll
                    * n_box_particles
                    * (n_box_particles - 1)
                    * self.box_vmax[idx]
                    * self.dt
                    + self.box_nc_error[idx]
                )
                Nc = int(Nc_float)
                self.box_nc_error[idx] = Nc_float - Nc
                for coll_idx in range(Nc):
                    p_index_1, p_index_2 = np.random.default_rng().choice(
                        p_list, 2, replace=False
                    )
                    n_collisions += self.binary_collision(
                        p_index_1, p_index_2, self.box_vmax[idx]
                    )
        self.nc += n_collisions

    def make_step(self) -> None:
        v_T = np.sqrt(2 * self.temperature / self.pp.mass)
        self.dt = self.wp.L_box / (self.C_vmax * v_T)
        self.collisions()
        self.translations()
        self.t += self.dt

    # theoretical curves
    def haff_cooling(self, t: float) -> float:
        tc_inv = self.t_c_inv * (1 - self.pp.eps**2) / 3
        return self.T0 / (1 + t * tc_inv) ** 2

    def ncol_analytic(self, t: float) -> float:
        C = (1 - self.pp.eps**2) / 3
        return (self.wp.N / C) * np.log(1 + C * self.t_c_inv * t)

    def __repr__(self) -> str:
        dump = f"""
        -------------------------------------------------------------------------
                                  World Parameters
        -------------------------------------------------------------------------
                 Number of particles:          N = {self.wp.N:_}
                      Number density:          n = {self.wp.n:.3f} 1/m^3
              Simulation area volume:          V = {self.wp.V:_.3f} m^3
                     Number of boxes: Nx, Ny, Nz = ({self.wp.Nx}, {self.wp.Ny}, {self.wp.Nz})
        Simulation area linear sizes: Lx, Ly, Lz = ({self.wp.Lx:.3f}, {self.wp.Ly:.3f}, {self.wp.Lz:.3f}) m
           Volume of the cubical box:      V_box = {self.wp.V_box:.3f} m^3
              Linear size of the box:      L_box = {self.wp.L_box:.3f} m
        -------------------------------------------------------------------------
                                  Particle Parameters
        -------------------------------------------------------------------------
                              Radius:          R = {self.pp.radius:.3f} m
                        Mass density:        rho = {self.pp.rho:.3f} kg/m^3
                                Mass:          M = {self.pp.mass:.3f} kg
          Coefficient of restitution:        eps = {self.pp.eps}
        -------------------------------------------------------------------------
                                  Simulation Parameters
        -------------------------------------------------------------------------
                 Initial temperature:         T0 = {self.wp.T0:.3f} J
          Actual initial temperature:         T0 = {self.T0:.3f} J
         Avg. N of particles per box:      N_box = {self.wp.N/(self.wp.Nx*self.wp.Ny*self.wp.Nz):.3f}
            Constant for coll. freq.:          C = {self.C_ncoll:.6f} 1/m
         Characteristic inverse time:    t_c_inv = {self.t_c_inv:.6f} 1/s
         
        """
        return dump


def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print(
        "\r[",
        "#" * int(left),
        " " * int(right),
        "]",
        f" {percent:.0f}%",
        sep="",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    sim = DSMC_Core()
    sim.initialize()
    print(sim)

    time_array = np.asarray([sim.t])
    temp_array = np.asarray([sim.T0])
    ncol_array = np.asarray([sim.nc])
    sim_end = 20.0

    while sim.t < sim_end:
        sim.make_step()
        time_array = np.append(time_array, sim.t)
        temp_array = np.append(temp_array, sim.temperature)
        ncol_array = np.append(ncol_array, sim.nc)
        progress(100 * sim.t / sim_end)
        # print(f" Simulating: {100*sim.t/sim_end:3.2f}%, time: {sim.t:.3f}, dt: {sim.dt:.3f}, temperature: {sim.temperature:.3f}, total collisions: {sim.nc}", end="\r", flush=True)
    print("")
    print("Finished simulation")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    ax1.plot(time_array, temp_array, "o", label="simulation")
    ax1.plot(time_array, sim.haff_cooling(time_array), label="analytic", color="red")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Granular temperature (J)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(time_array, ncol_array / sim.wp.N, "o", label="simulation")
    ax2.plot(
        time_array,
        sim.ncol_analytic(time_array) / sim.wp.N,
        label="analytic",
        color="red",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cumulative number of collisions (x100 000)")
    ax2.legend()
    plt.show()
