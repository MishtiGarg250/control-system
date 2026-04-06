import numpy as np
from scipy.optimize import minimize

class MPCTeacher:
    def __init__(self, plant, T_ref=450, horizon=15):
        self.plant = plant
        self.T_ref = T_ref
        self.N = horizon

    def q_nominal(self, dni):
        return 0.3 + (dni / 900.0) * (1.5 - 0.3)

    def cost(self, q_seq, dni_seq):
        T_pred = self.plant.predict(q_seq, dni_seq)

        # temperature band constraint
        temp_err = np.maximum(0, np.abs(T_pred - self.T_ref) - 5)
        temp_cost = 5.0 * np.sum(temp_err**2)

        # flow follows DNI (MAIN objective)
        q_ref = np.array([self.q_nominal(d) for d in dni_seq])
        flow_cost = 300.0 * np.sum((q_seq - q_ref)**2)

        # smoothness
        dq = np.diff(q_seq, prepend=q_seq[0])
        smooth_cost = 0.01 * np.sum(dq**2)

        return temp_cost + flow_cost + smooth_cost

    def compute_control(self, dni_forecast):
        q0 = np.ones(self.N) * 0.8
        bounds = [(0.3, 1.5)] * self.N

        res = minimize(
            self.cost,
            q0,
            args=(dni_forecast,),
            bounds=bounds,
            method="SLSQP"
        )
        return res.x[0]

