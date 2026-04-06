import numpy as np
import os
def generate_dni():
    Ts = 60
    Tend = 12 * 3600
    t = np.arange(0, Tend, Ts)

    base = 900 * np.sin(np.pi * t / Tend)
    base[base < 0] = 0

    cloud = 1 - 0.35 * np.exp(-((t - Tend/2) / 1200)**2)
    dni = base * cloud

    return dni

if __name__ == "__main__":
    dni = generate_dni()
    os.makedirs("data", exist_ok=True)
    np.save("data/dni_profiles.npy", dni)
