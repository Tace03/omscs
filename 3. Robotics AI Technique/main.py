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

    visualizer = Visualizer(
        world_min=0,
        world_max=30,
    )

    dt = 0.1
    current_time = 0.0

    while current_time < 20:
        true_position, measurement = (
            simulation.step(dt)
        )

        kalman.predict(dt)

        estimate, uncertainty, gain = (
            kalman.update(measurement)
        )

        visualizer.update(
            time=current_time,
            true_position=true_position,
            measurement=measurement,
            estimate=estimate,
            uncertainty=uncertainty,
            gain=gain,
        )

        current_time += dt


if __name__ == "__main__":
    main()