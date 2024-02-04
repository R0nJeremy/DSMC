import numpy as np


class DSMC_Mono:
    def __init__(self) -> None:
        # * simulation space constants
        self.N_particles = 1_000_00  # number of particles in the system
        self.n = 0.02  # number density (meter^-3)
        self.Nx, self.Ny, self.Nz = 20, 20, 20  # number of boxes in each dimension

        self.V = self.N_particles / self.n  # volume of the entire system (meter^3)
        self.L_box = np.power(
            self.V / (self.Nx * self.Ny * self.Nz), 1 / 3
        )  # linear size of a box (meter)
        # linear sizes of the simulation space (meters)
        self.Lx = self.L_box * self.Nx
        self.Ly = self.L_box * self.Ny
        self.Lz = self.L_box * self.Nz
        self.V_box = self.L_box**3

        # * particle constants
        self.eps = 0.6  # coefficient of restitution
        self.R = 0.05  # radius of a particle (meter)
        self.rho = (
            900  # mass density of a particle. The value is for water-ice (kg/meter^3)
        )
        self.m = (4 * np.pi / 3) * self.rho * self.R**3  # mass of a particle (kg)

        # * data structures
        # spacial positions of particles
        self.rx = np.ndarray((self.N_particles), dtype=np.float64)
        self.ry = np.ndarray((self.N_particles), dtype=np.float64)
        self.rz = np.ndarray((self.N_particles), dtype=np.float64)
        # velocities of particles
        self.vx = np.ndarray((self.N_particles), dtype=np.float64)
        self.vy = np.ndarray((self.N_particles), dtype=np.float64)
        self.vz = np.ndarray((self.N_particles), dtype=np.float64)
        # data containers for each box
        self.box_particles = np.ndarray(
            (self.Nx, self.Ny, self.Nz), dtype=object
        )  # list of indices of particles in a box
        self.box_vmax = np.ndarray(
            (self.Nx, self.Ny, self.Nz), dtype=np.float64
        )  # vmax per box
        self.box_nc_error = np.ndarray(
            (self.Nx, self.Ny, self.Nz), dtype=np.float64
        )  # rounding error for number of collisions per box

        # * system states and their initial conditions
        self.t = 0.0  # simulation time starts at t=0 (seconds)
        self.dt = 0.1  # simulation time-step (seconds).
        self.T0 = 10.0
        self.temperature = self.T0  # temperature of the system (J*meter^3)
        self.nc = 0  # number of collisions since the start of the simulation

        # * DSMC simulation constants
        self.C_vmax = 5
        self.C_ncoll = 4 * np.pi * self.R**2 / self.V_box

        # * Constants for analytic tests
        self.nu_c = 8 * self.n * self.R**2 * np.sqrt(np.pi * self.T0 / self.m)

    def reset_data_structures(self) -> None:
        # * system parameters
        self.L_box = np.power(self.V / (self.Nx * self.Ny * self.Nz), 1 / 3)
        self.V_box = self.V_box**3
        self.Lx = self.L_box * self.Nx
        self.Ly = self.L_box * self.Ny
        self.Lz = self.L_box * self.Nz
        self.rx = np.ndarray((self.N_particles), dtype=np.float64)
        self.ry = np.ndarray((self.N_particles), dtype=np.float64)
        self.rz = np.ndarray((self.N_particles), dtype=np.float64)
        # velocities of particles
        self.vx = np.ndarray((self.N_particles), dtype=np.float64)
        self.vy = np.ndarray((self.N_particles), dtype=np.float64)
        self.vz = np.ndarray((self.N_particles), dtype=np.float64)
        # data containers for each box
        self.box_particles = np.ndarray(
            (self.Nx, self.Ny, self.Nz), dtype=object
        )  # list of indices of particles in a box
        self.box_vmax = np.ndarray(
            (self.Nx, self.Ny, self.Nz), dtype=np.float64
        )  # vmax per box
        self.box_nc_error = np.ndarray(
            (self.Nx, self.Ny, self.Nz), dtype=np.float64
        )  # rounding error for number of collisions per box
        # * DSMC simulation constants
        self.C_vmax = 5
        self.C_ncoll = 4 * np.pi * self.R**2 / self.V_box
        # * Constants for analytic tests
        self.nu_c = 8 * self.n * self.R**2 * np.sqrt(np.pi * self.T0 / self.m)

    def initialize(self) -> None:
        self.reset_data_structures()
        print("Initializing system...")
        print("...")
        print(f"Number of simulated particles:       N       = {self.N_particles:_}")
        print(
            f"Linear sizes of the system:          Lx      = {self.Lx:.2f} (m), Ly = {self.Ly:.2f} (m), Lz = {self.Lz:.2f} (m)"
        )
        print(f"Volume of the simulated space:       V       = {self.V:_.2f} (m^3)")
        print(
            f"Number of boxes:                     Nx      = {self.Nx}, Ny = {self.Ny}, Nz = {self.Nz}"
        )
        print(f"Linear size of a box:                L_box   = {self.L_box:.2f} (m)")
        print(f"Volume of a box:                     V_box   = {self.V_box:_.2f} (m^3)")
        print(
            f"Total number of boxes:               N_total = {self.Nx * self.Ny * self.Nz:_}"
        )
        print(
            f"Average number of particles per box: Nbox    = {self.N_particles/(self.Nx*self.Ny*self.Nz):.2f}"
        )
        print(f"Number density:                      n       = {self.n:.2f} (1/m^3)")
        print(f"Particles' radii:                    R       = {self.R:.2f} (m)")
        print(f"Particles' mass densities:           rho     = {self.rho:.2f} (kg/m^3)")
        print(f"Particles' masses:                   m       = {self.m:.2f} (kg)")
        print(f"Coefficient of restitution:          eps     = {self.eps:.2f}")
        print("...")

        self.box_nc_error.fill(0.0)

        # uniform distribution of particles in the simulation space
        self.rx = np.random.default_rng().uniform(0, self.Lx, self.N_particles)
        self.ry = np.random.default_rng().uniform(0, self.Ly, self.N_particles)
        self.rz = np.random.default_rng().uniform(0, self.Lz, self.N_particles)

        # thermal distribution of velocities
        self.vx = np.random.default_rng().normal(
            0, np.sqrt(self.T0 / self.m), self.N_particles
        )
        self.vy = np.random.default_rng().normal(
            0, np.sqrt(self.T0 / self.m), self.N_particles
        )
        self.vz = np.random.default_rng().normal(
            0, np.sqrt(self.T0 / self.m), self.N_particles
        )

        print("Coordinates and velocities are distributed")

        # fill the data structures
        self.set_system_state()
        self.T0 = self.temperature
        self.nu_c = 8 * self.n * self.R**2 * np.sqrt(np.pi * self.T0 / self.m)
        print("...")
        print("Data structure are initialized")
        print("...")
        print(f"Initial temperature:                 T0      = {self.T0:.5f} (J)")
        print(
            f"Sampled temperature:                 T       = {self.temperature:.5f} (J)"
        )
        print(
            f"Mass density of the system:          rho     = {self.m*self.N_particles/self.V:.5f} (kg/m^3)"
        )
        print(
            f"Pressure:                            P       = {self.n*self.temperature:.5f} (Pa)"
        )
        print(f"Characteristics time (inverse):      t_c^-1  = {self.nu_c:.5f} (1/s)")
        print("...")
        print("Ready for simulation!")
        print("---")

    # fills all data structures and temperature of the system
    def set_system_state(self) -> None:
        self.temperature = (self.m / 3) * (
            self.vx**2 + self.vy**2 + self.vz**2
        ).mean()

        for box_idx, _ in np.ndenumerate(self.box_particles):
            self.box_particles[box_idx] = np.ndarray((0), dtype=int)

        for p_idx in range(self.N_particles):
            i = (self.rx[p_idx] / self.L_box).astype(int)
            j = (self.ry[p_idx] / self.L_box).astype(int)
            k = (self.rz[p_idx] / self.L_box).astype(int)
            self.box_particles[i, j, k] = np.append(self.box_particles[i, j, k], p_idx)

        for box_idx, box_list in np.ndenumerate(self.box_particles):
            if box_list.size == 0:
                temperature_box = 0.0
            else:
                temperature_box = (self.m / 3) * (
                    self.vx[box_list] ** 2
                    + self.vy[box_list] ** 2
                    + self.vz[box_list] ** 2
                ).mean()
            v_thermal = np.sqrt(2 * temperature_box / self.m)
            self.box_vmax[box_idx] = self.C_vmax * v_thermal

    # streaming motion of particles
    def translations(self) -> None:
        self.rx = (self.rx + self.vx * self.dt + self.Lx) % self.Lx
        self.ry = (self.ry + self.vy * self.dt + self.Ly) % self.Ly
        self.rz = (self.rz + self.vz * self.dt + self.Lz) % self.Lz

    # sample collisions
    def collisions(self) -> None:
        n_collisions = 0
        for box_idx, box_list in np.ndenumerate(self.box_particles):
            n_box_particles = box_list.size
            if n_box_particles > 1:
                Nc_float = (
                    self.C_ncoll
                    * n_box_particles
                    * (n_box_particles - 1)
                    * self.box_vmax[box_idx]
                    * self.dt
                    + self.box_nc_error[box_idx]
                )
                Nc = int(Nc_float)
                self.box_nc_error[box_idx] = Nc_float - Nc
                for coll_idx in range(Nc):
                    p_index_1, p_index_2 = np.random.default_rng().choice(
                        box_list, 2, replace=False
                    )
                    n_collisions += self.binary_collision(
                        p_index_1, p_index_2, self.box_vmax[box_idx]
                    )
        self.nc += n_collisions

    # binary collision
    def binary_collision(self, p1, p2, vmax) -> int:
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
            h = (1 + self.eps) * dv_norm / 2
            self.vx[p1] -= h * ndx
            self.vy[p1] -= h * ndy
            self.vz[p1] -= h * ndz
            self.vx[p2] += h * ndx
            self.vy[p2] += h * ndy
            self.vz[p2] += h * ndz
            return 1

    # simulation step
    def make_step(self) -> None:
        v_T = np.sqrt(2 * self.temperature / self.m)
        self.dt = self.L_box / (self.C_vmax * v_T)
        self.collisions()
        self.translations()
        self.set_system_state()
        self.t += self.dt

    # theoretical curves
    def haff_cooling(self, t: float) -> float:
        tc_inv = (
            (8 / 3)
            * (1 - self.eps**2)
            * self.n
            * self.R**2
            * np.sqrt(np.pi * self.T0 / self.m)
        )
        return self.T0 / (1 + t * tc_inv) ** 2

    def ncol_analytic(self, t: float) -> float:
        C = (1 - self.eps**2) / 3
        return (self.N_particles / C) * np.log(1 + C * self.nu_c * t)
