class KalmanFilter1D:
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