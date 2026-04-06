import numpy as np
from plant import SolarPlant
from mpc_teacher import MPCTeacher

H = 15

dni = np.load("data/dni_profiles.npy")

plant = SolarPlant()
mpc = MPCTeacher(plant)

X, y = [], []

for i in range(len(dni) - H):
    forecast = dni[i:i+H]
    q = mpc.compute_control(forecast)

    T_in, T_out, _ = plant.step(q, dni[i])

    X.append([
        T_in/500,
        T_out/500,
        *(forecast/900)
    ])
    y.append(q)

np.save("data/X.npy", np.array(X))
np.save("data/y.npy", np.array(y))
