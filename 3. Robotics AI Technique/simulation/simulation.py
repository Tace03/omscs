import random


class Simulation:
    def __init__(
        self,
        initial_position=0.0,
        velocity=1.0,
        acceleration = 0.1,
        measurement_std=3.0,
    ):
        self.position = initial_position
        self.velocity = velocity
        self.acceleration = acceleration
        self.measurement_std = measurement_std

    def step(self, dt):
        # Move the real object
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt * dt
        self.velocity += self.acceleration * dt

        # Sensor measurement with Gaussian noise
        measurement = (
            self.position
            + random.gauss(0, self.measurement_std)
        )

        return self.position, measurement