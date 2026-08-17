from filters import KalmanFilter1D
from simulation import Simulation
from visualization import ScalarVisualizer,StateVisualizer


def main():
    simulation = Simulation(
        initial_position=0.0,
        velocity=1.0,
        acceleration=0.1,
        measurement_std=3.0,
    )

    kalman = KalmanFilter1D(
        initial_estimate=0.0,
        intial_velocity=1.0,
        control_acceleration=0.1,
        initial_uncertainty=10.0,
        process_variance=0.1,
        measurement_variance=9.0,
    )

    visualizer = StateVisualizer(
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

        state, P, K, innovation = kalman.update(measurement)

        estimated_position = state[0, 0]
        estimated_velocity = state[1, 0]

        position_variance = P[0, 0]
        velocity_variance = P[1, 1]

        position_gain = K[0, 0]
        velocity_gain = K[1, 0]

        visualizer.update(
            time=current_time,
            true_position=true_position,
            true_velocity=simulation.velocity,
            measurement=measurement,
            state=state,
            covariance=P,
            gain=K,
            innovation=innovation,
        )

        current_time += dt


if __name__ == "__main__":
    main()