import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from plant import SolarPlant
from mpc_teacher import MPCTeacher
from model import Net

dni = np.load("data/dni_profiles.npy")

plant = SolarPlant()
mpc = MPCTeacher(plant)
net = Net()
net.load_state_dict(torch.load("results/model.pt"))
net.eval()

T, Qm, Qn, P = [], [], [], []

for i in range(len(dni) - 15):
    forecast = dni[i:i+15]
    q_m = mpc.compute_control(forecast)

    X = torch.tensor([[
        plant.T_in/500,
        plant.T_out/500,
        *(forecast/900)
    ]], dtype=torch.float32)

    q_n = net(X).item()

    _, To, power = plant.step(q_n, dni[i])

    T.append(To)
    Qm.append(q_m)
    Qn.append(q_n)
    P.append(power)

t = np.arange(len(T))/60

fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

# ---------- Layer 1: DNI ----------
ax[0].plot(dni[:len(T)], color="orange", linewidth=2, label="DNI (Solar Irradiance)")
ax[0].set_ylabel("DNI (W/m²)")
ax[0].set_title("Layer 1: Direct Normal Irradiance (External Disturbance)")
ax[0].legend(loc="upper right")
ax[0].grid(True)

# ---------- Layer 2: Outlet Temperature ----------
ax[1].plot(T, color="red", linewidth=2, label="Outlet Temperature")
ax[1].axhline(450, color="black", linestyle="--", label="Reference Temperature (450°C)")
ax[1].set_ylabel("Temperature (°C)")
ax[1].set_title("Layer 2: Outlet Temperature (Controlled Variable)")
ax[1].legend(loc="upper right")
ax[1].grid(True)

# ---------- Layer 3: Mass Flow Rate ----------
ax[2].plot(Qm, "k--", linewidth=2, label="MPC Flow Rate")
ax[2].plot(Qn, "r", linewidth=2, label="ANN Flow Rate")
ax[2].set_ylabel("Flow Rate (kg/s)")
ax[2].set_title("Layer 3: Mass Flow Rate (Control Action)")
ax[2].legend(loc="upper right")
ax[2].grid(True)

# ---------- Layer 4: Thermal Power ----------
ax[3].plot(P, color="green", linewidth=2, label="Thermal Power Output")
ax[3].set_ylabel("Power (arb. units)")
ax[3].set_xlabel("Time (minutes)")
ax[3].set_title("Layer 4: Thermal Power Output")
ax[3].legend(loc="upper right")
ax[3].grid(True)

# ---------- Global figure annotation ----------
fig.suptitle(
    "Closed-loop Response of Solar Thermal Plant\n"
    "DNI → Flow & Power Response, Temperature Regulation via MPC and ANN",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/final_result_with_layers.png", dpi=300)
plt.show()


