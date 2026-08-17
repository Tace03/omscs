import time

from filters import KalmanFilter1D
from simulation import Simulation
from visualization import Visualizer


def main():
    simulation = Simulation(
        initial_position=0.0,
        velocity=1.0,
        measurement_std=3.0,
    )

    kalman = KalmanFilter1D(
        initial_estimate=0.0,
        initial_uncertainty=10.0,
        process_variance=0.1,
        measurement_variance=9.0,
    )

    visualizer = Visualizer()

    dt = 0.1
    current_time = 0.0

    while current_time < 20:
        # 1. Simulate reality
        true_position, measurement = simulation.step(dt)

        # 2. Kalman prediction
        kalman.predict(dt)

        # 3. Kalman measurement correction
        estimate, uncertainty, gain = kalman.update(
            measurement
        )

        # 4. Visualize
        visualizer.update(
            current_time,
            true_position,
            measurement,
            estimate,
            uncertainty,
        )

        current_time += dt

        time.sleep(dt)


if __name__ == "__main__":
    main()