# packages
using Unitful
using Statistics

#* Simulation space constants
N_particles = 1_000_00
n = 0.02u"1/m^3" # number density
V = N_particles / n 
Nx = 20; Ny = 20; Nz = 20
L_box = (V/(Nx*Ny*Nz))^(1/3)
V_box = L_box^3
Lx = Nx*L_box; Ly = Ny*L_box; Lz = Nz*L_box

#* Particle constants
ϵ = 0.6 # coefficient of restitution
R = 0.05u"m" # radius of a particle
ρ = 900u"kg/m^3" # mass density (ice)
m = (4π/3)*ρ*R^3 # mass of a particle

#* Initial conditions, simulation variables and constants
T₀ = 10.0u"kg*m^2/s^2" # initial temperature
T = T₀ # current temperature
t = 0.0u"s" # simulation time 
Δt = 0.1u"s" # time step (adaptive)
C_vmax = 5.0 # thermal speed upper threshold

#* Particles' positions and velocitites
# positions of particles
rx = Lx*rand(N_particles)
ry = Ly*rand(N_particles)
rz = Lz*rand(N_particles)
# velocities of particles
vx = randn(N_particles).*sqrt(T₀/m)
vy = randn(N_particles).*sqrt(T₀/m)
vz = randn(N_particles).*sqrt(T₀/m)

vx .-= mean(vx); vy .-= mean(vy); vz .-= mean(vz)

# Update the temperature with the actual value
T₀ = m*mean(vx.^2+vy.^2+vz.^2)/3
T = T₀

#* Data structures
box_vmax = Array{Float64, 3}(undef, (Nx,Ny,Nz)).*u"m/s"
box_nc_error = Array{Float64, 3}(undef, (Nx,Ny,Nz))
box_particles = Array{Vector{Int64}, 3}(undef, (Nx,Ny,Nz))
for idx in CartesianIndices(box_particles)
    box_vmax[idx] = 0.0u"m/s"
    box_nc_error[idx] = 0.0
    box_particles[idx] = Vector{Int64}(undef, 0)    
end

function set_system_state!(_box_vmax, _box_particles, _T)
    _T = m*mean(vx.^2+vy.^2+vz.^2)/3
    for idx in CartesianIndices(box_particles)
        _box_particles[idx] = Vector{Int64}(undef, 0)
    end

    for p_idx in 1:N_particles
        i = Int(1+floor(rx[p_idx] / L_box))
        j = Int(1+floor(ry[p_idx] / L_box))
        k = Int(1+floor(rz[p_idx] / L_box))
        append!(_box_particles[i,j,k], p_idx)
    end
    
    for (idx, box_list) in pairs(box_particles)
        if isempty(box_list)
            temperature_box = 0.0u"kg*m^2/s^2"
        else
            temperature_box = m*mean(vx[box_list].^2+vy[box_list].^2+vz[box_list].^2)/3
        end
        v_thermal = sqrt(2*temperature_box/m)        
        _box_vmax[idx] = C_vmax * v_thermal
    end    
end



println(rx)
