import numpy as np

class SolarPlant:
    def __init__(self, dt=60):
        self.dt = dt
        self.q_min = 0.3
        self.q_max = 1.5
        self.reset()

    def reset(self):
        self.T_out = 230.0
        self.T_in = 220.0

    def step(self, q, dni):
        q = np.clip(q, self.q_min, self.q_max)

        # solar heating inversely proportional to flow
        heat_in = 0.18 * dni / q
        heat_loss = 0.01 * (self.T_out - 25)

        self.T_out += self.dt * (heat_in - heat_loss)
        self.T_in += 0.02 * (self.T_out - self.T_in)

        # physical bounds
        self.T_out = np.clip(self.T_out, 200, 470)
        self.T_in = np.clip(self.T_in, 180, self.T_out)

        # power tracks DNI and flow
        power = 0.002 * q * dni

        return self.T_in, self.T_out, power

    def predict(self, q_seq, dni_seq):
        T_out = self.T_out
        T_in = self.T_in
        preds = []

        for q, dni in zip(q_seq, dni_seq):
            q = np.clip(q, self.q_min, self.q_max)

            heat_in = 0.18 * dni / q
            heat_loss = 0.01 * (T_out - 25)

            T_out += self.dt * (heat_in - heat_loss)
            T_in += 0.02 * (T_out - T_in)

            T_out = np.clip(T_out, 200, 470)
            T_in = np.clip(T_in, 180, T_out)

            preds.append(T_out)

        return np.array(preds)
