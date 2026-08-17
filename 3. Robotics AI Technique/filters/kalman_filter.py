
import numpy as np

class ScalarKalmanFilter1D:
    def __init__(
        self,
        initial_estimate=0.0,
        intial_velocity = 1.0,
        initial_uncertainty=10.0,
        process_variance=0.1,
        measurement_variance=9.0,
    ):
        self.x = initial_estimate
        self.P = initial_uncertainty
        self.v = intial_velocity

        self.Q = process_variance
        self.R = measurement_variance

    def predict(self, dt):
        # No explicit motion model yet
        self.x = self.x + self.v * dt
        self.P = self.P + self.Q

    def update(self, measurement):
        # Kalman gain
        K = self.P / (self.P + self.R)

        # State update
        self.x = self.x + K * (measurement - self.x)

        # Covariance update
        self.P = (1 - K) * self.P

        return self.x, self.P, K

class KalmanFilter1D:
    def __init__(
        self,
        initial_estimate=0.0,
        intial_velocity = 1.0,
        control_acceleration = 0.1,
        initial_uncertainty=10.0,
        process_variance=0.1,
        measurement_variance=9.0,
    ):
        self.state = np.array([
            [initial_estimate],
            [intial_velocity],
        ])

        self.P = np.array([
            [initial_uncertainty, 0],
            [0                  , initial_uncertainty],
        ])

        self.R = np.array([
            [measurement_variance],
        ])

        self.Q = np.array([
            [process_variance, 0.0],
            [0.0, process_variance],
        ])

        self.a = np.array([
            [control_acceleration],
        ])
    
    def predict(self, dt):
        F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        B = np.array([
            [0.5 * dt * dt],
            [dt],
        ])

        self.state = F @ self.state + self.a   # x- = F * x_hat + B * accel
        self.P = F @ self.P @ F.T + self.Q   # P- = F * P * FT + Q

    def update(self,measurement):
        # Selection matrix
        H = np.array([
            [1.0, 0.0]
        ])

        z = np.array([
            [measurement]
        ])

        innovation = z - H @ self.state

        S = H @ self.P @ H.T + self.R       # Covariance matrix S
                                            # H * P * HT + R 

        # Calculate K gain
        if (S.shape == (1, 1)):
            K = self.P @ H.T / S[0,0]   # K = P * HT * (S)^-1
        else:
            K = self.P @ H.T @ np.linalg.inv(S)

        # Update 
        self.state = self.state + K * (z - H @ self.state)

        I = np.eye(2)

        self.P = (I - K @ H) @ self.P

        return self.state, self.P, K, innovation
