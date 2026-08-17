import math

import matplotlib.pyplot as plt


class ScalarVisualizer:
    def __init__(
        self,
        world_min=0.0,
        world_max=30.0,
    ):
        self.world_min = world_min
        self.world_max = world_max

        self.times = []
        self.true_positions = []
        self.measurements = []
        self.estimates = []
        self.uncertainties = []

        plt.ion()

        self.fig = plt.figure(figsize=(12, 8))

        # Layout:
        #
        # Physical world      Internals
        # -----------------------------
        # Position history graph
        #
        grid = self.fig.add_gridspec(
            2,
            2,
            height_ratios=[1, 2],
            width_ratios=[3, 1],
        )

        self.ax_world = self.fig.add_subplot(grid[0, 0])
        self.ax_info = self.fig.add_subplot(grid[0, 1])
        self.ax_history = self.fig.add_subplot(grid[1, :])

        self.fig.tight_layout()

    def update(
        self,
        time,
        true_position,
        measurement,
        estimate,
        uncertainty,
        gain,
    ):
        self.times.append(time)
        self.true_positions.append(true_position)
        self.measurements.append(measurement)
        self.estimates.append(estimate)
        self.uncertainties.append(uncertainty)

        self._draw_world(
            true_position,
            measurement,
            estimate,
            uncertainty,
        )

        self._draw_history()

        self._draw_info(
            true_position,
            measurement,
            estimate,
            uncertainty,
            gain,
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _draw_world(
        self,
        true_position,
        measurement,
        estimate,
        uncertainty,
    ):
        self.ax_world.clear()

        self.ax_world.set_title("Physical World")

        self.ax_world.set_xlim(
            self.world_min,
            self.world_max,
        )

        self.ax_world.set_ylim(-1, 1)

        # Horizontal 1D world line
        self.ax_world.axhline(
            y=0,
            linewidth=1,
        )

        # True position
        self.ax_world.scatter(
            true_position,
            0,
            s=120,
            marker="o",
            label="True position",
        )

        # Sensor measurement
        self.ax_world.scatter(
            measurement,
            0.25,
            s=100,
            marker="x",
            label="Measurement",
        )

        # Kalman estimate
        self.ax_world.scatter(
            estimate,
            -0.25,
            s=120,
            marker="o",
            facecolors="none",
            label="Estimate",
        )

        # Convert variance P into standard deviation
        sigma = math.sqrt(max(uncertainty, 0))

        # ±2 sigma confidence region
        lower = estimate - 2 * sigma
        upper = estimate + 2 * sigma

        self.ax_world.plot(
            [lower, upper],
            [-0.25, -0.25],
            linewidth=5,
            alpha=0.3,
        )

        self.ax_world.set_yticks(
            [0.25, 0, -0.25],
            ["Measurement", "Truth", "Estimate"],
        )

        self.ax_world.set_xlabel("Position")

        self.ax_world.legend(
            loc="upper left",
        )

        self.ax_world.grid(axis="x")

    def _draw_history(self):
        self.ax_history.clear()

        self.ax_history.set_title(
            "Position vs Time"
        )

        self.ax_history.plot(
            self.times,
            self.true_positions,
            label="True position",
        )

        self.ax_history.scatter(
            self.times,
            self.measurements,
            s=15,
            alpha=0.5,
            label="Measurements",
        )

        self.ax_history.plot(
            self.times,
            self.estimates,
            label="Kalman estimate",
        )

        #
        # Uncertainty band
        #
        lower_bound = []
        upper_bound = []

        for estimate, uncertainty in zip(
            self.estimates,
            self.uncertainties,
        ):
            sigma = math.sqrt(
                max(uncertainty, 0)
            )

            lower_bound.append(
                estimate - 2 * sigma
            )

            upper_bound.append(
                estimate + 2 * sigma
            )

        self.ax_history.fill_between(
            self.times,
            lower_bound,
            upper_bound,
            alpha=0.15,
            label="±2σ",
        )

        self.ax_history.set_xlabel("Time")
        self.ax_history.set_ylabel("Position")

        self.ax_history.legend()
        self.ax_history.grid()

    def _draw_info(
        self,
        true_position,
        measurement,
        estimate,
        uncertainty,
        gain,
    ):
        self.ax_info.clear()

        self.ax_info.set_title(
            "Kalman Internals"
        )

        self.ax_info.axis("off")

        error = estimate - true_position

        innovation = measurement - estimate

        sigma = math.sqrt(
            max(uncertainty, 0)
        )

        info = (
            f"True x:\n"
            f"{true_position:8.3f}\n\n"

            f"Measurement z:\n"
            f"{measurement:8.3f}\n\n"

            f"Estimate x̂:\n"
            f"{estimate:8.3f}\n\n"

            f"Error:\n"
            f"{error:8.3f}\n\n"

            f"Variance P:\n"
            f"{uncertainty:8.3f}\n\n"

            f"σ = sqrt(P):\n"
            f"{sigma:8.3f}\n\n"

            f"Kalman Gain K:\n"
            f"{gain:8.3f}\n\n"

            f"Innovation:\n"
            f"{innovation:8.3f}"
        )

        self.ax_info.text(
            0.05,
            0.95,
            info,
            transform=self.ax_info.transAxes,
            verticalalignment="top",
            family="monospace",
            fontsize=11,
        )

class StateVisualizer:
    def __init__(
        self,
        world_min=0.0,
        world_max=30.0,
    ):
        self.world_min = world_min
        self.world_max = world_max

        # History
        self.times = []

        self.true_positions = []
        self.measurements = []
        self.estimated_positions = []

        self.true_velocities = []
        self.estimated_velocities = []

        self.position_variances = []
        self.velocity_variances = []

        self.position_gains = []
        self.velocity_gains = []

        plt.ion()

        self.fig = plt.figure(figsize=(13, 10))

        grid = self.fig.add_gridspec(
            3,
            2,
            height_ratios=[1, 2, 2],
            width_ratios=[3, 1],
        )

        # Top-left: physical world
        self.ax_world = self.fig.add_subplot(
            grid[0, 0]
        )

        # Top-right: Kalman internals
        self.ax_info = self.fig.add_subplot(
            grid[0, 1]
        )

        # Middle: position history
        self.ax_position = self.fig.add_subplot(
            grid[1, :]
        )

        # Bottom: velocity history
        self.ax_velocity = self.fig.add_subplot(
            grid[2, :]
        )

        self.fig.tight_layout()

    def update(
        self,
        time,
        true_position,
        true_velocity,
        measurement,
        state,
        covariance,
        gain,
        innovation,
    ):
        # Extract state
        estimated_position = state[0, 0]
        estimated_velocity = state[1, 0]

        # Extract covariance
        position_variance = covariance[0, 0]
        velocity_variance = covariance[1, 1]

        # Extract gain
        position_gain = gain[0, 0]
        velocity_gain = gain[1, 0]

        # Innovation is currently a 1x1 matrix
        innovation_value = innovation[0, 0]

        # Store history
        self.times.append(time)

        self.true_positions.append(
            true_position
        )

        self.measurements.append(
            measurement
        )

        self.estimated_positions.append(
            estimated_position
        )

        self.true_velocities.append(
            true_velocity
        )

        self.estimated_velocities.append(
            estimated_velocity
        )

        self.position_variances.append(
            position_variance
        )

        self.velocity_variances.append(
            velocity_variance
        )

        self.position_gains.append(
            position_gain
        )

        self.velocity_gains.append(
            velocity_gain
        )

        # Draw all panels
        self._draw_world(
            true_position=true_position,
            measurement=measurement,
            estimated_position=estimated_position,
            position_variance=position_variance,
        )

        self._draw_position_history()

        self._draw_velocity_history()

        self._draw_info(
            true_position=true_position,
            true_velocity=true_velocity,
            measurement=measurement,
            estimated_position=estimated_position,
            estimated_velocity=estimated_velocity,
            covariance=covariance,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            innovation=innovation_value,
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _draw_world(
        self,
        true_position,
        measurement,
        estimated_position,
        position_variance,
    ):
        self.ax_world.clear()

        self.ax_world.set_title(
            "Physical World"
        )

        self.ax_world.set_xlim(
            self.world_min,
            self.world_max,
        )

        self.ax_world.set_ylim(
            -1,
            1,
        )

        # World axis
        self.ax_world.axhline(
            y=0,
            linewidth=1,
        )

        # Sensor measurement
        self.ax_world.scatter(
            measurement,
            0.35,
            marker="x",
            s=100,
            label="Measurement",
        )

        # True position
        self.ax_world.scatter(
            true_position,
            0.0,
            marker="o",
            s=120,
            label="True position",
        )

        # Estimated position
        self.ax_world.scatter(
            estimated_position,
            -0.35,
            marker="o",
            facecolors="none",
            s=120,
            label="Kalman estimate",
        )

        # Position uncertainty
        sigma = math.sqrt(
            max(position_variance, 0.0)
        )

        lower = (
            estimated_position
            - 2 * sigma
        )

        upper = (
            estimated_position
            + 2 * sigma
        )

        self.ax_world.plot(
            [lower, upper],
            [-0.35, -0.35],
            linewidth=6,
            alpha=0.25,
            label="±2σ",
        )

        self.ax_world.set_yticks(
            [0.35, 0.0, -0.35],
            [
                "Measurement",
                "Truth",
                "Estimate",
            ],
        )

        self.ax_world.set_xlabel(
            "Position"
        )

        self.ax_world.grid(
            axis="x"
        )

        self.ax_world.legend(
            loc="upper left"
        )

    def _draw_position_history(self):
        self.ax_position.clear()

        self.ax_position.set_title(
            "Position vs Time"
        )

        self.ax_position.plot(
            self.times,
            self.true_positions,
            label="True position",
        )

        self.ax_position.scatter(
            self.times,
            self.measurements,
            s=15,
            alpha=0.5,
            label="Measurements",
        )

        self.ax_position.plot(
            self.times,
            self.estimated_positions,
            label="Estimated position",
        )

        # Build ±2 sigma uncertainty region
        lower_bound = []
        upper_bound = []

        for estimate, variance in zip(
            self.estimated_positions,
            self.position_variances,
        ):
            sigma = math.sqrt(
                max(variance, 0.0)
            )

            lower_bound.append(
                estimate - 2 * sigma
            )

            upper_bound.append(
                estimate + 2 * sigma
            )

        self.ax_position.fill_between(
            self.times,
            lower_bound,
            upper_bound,
            alpha=0.15,
            label="Position ±2σ",
        )

        self.ax_position.set_xlabel(
            "Time"
        )

        self.ax_position.set_ylabel(
            "Position"
        )

        self.ax_position.grid()
        self.ax_position.legend()

    def _draw_velocity_history(self):
        self.ax_velocity.clear()

        self.ax_velocity.set_title(
            "Velocity vs Time"
        )

        self.ax_velocity.plot(
            self.times,
            self.true_velocities,
            label="True velocity",
        )

        self.ax_velocity.plot(
            self.times,
            self.estimated_velocities,
            label="Estimated velocity",
        )

        # Velocity uncertainty band
        lower_bound = []
        upper_bound = []

        for estimate, variance in zip(
            self.estimated_velocities,
            self.velocity_variances,
        ):
            sigma = math.sqrt(
                max(variance, 0.0)
            )

            lower_bound.append(
                estimate - 2 * sigma
            )

            upper_bound.append(
                estimate + 2 * sigma
            )

        self.ax_velocity.fill_between(
            self.times,
            lower_bound,
            upper_bound,
            alpha=0.15,
            label="Velocity ±2σ",
        )

        self.ax_velocity.set_xlabel(
            "Time"
        )

        self.ax_velocity.set_ylabel(
            "Velocity"
        )

        self.ax_velocity.grid()
        self.ax_velocity.legend()

    def _draw_info(
        self,
        true_position,
        true_velocity,
        measurement,
        estimated_position,
        estimated_velocity,
        covariance,
        position_gain,
        velocity_gain,
        innovation,
    ):
        self.ax_info.clear()

        self.ax_info.set_title(
            "Kalman Internals"
        )

        self.ax_info.axis("off")

        position_error = (
            estimated_position
            - true_position
        )

        velocity_error = (
            estimated_velocity
            - true_velocity
        )

        position_sigma = math.sqrt(
            max(covariance[0, 0], 0.0)
        )

        velocity_sigma = math.sqrt(
            max(covariance[1, 1], 0.0)
        )

        info = (
            "STATE\n"
            "----------------\n"
            f"True x:    {true_position:8.3f}\n"
            f"Estimate x:{estimated_position:8.3f}\n"
            f"Error x:   {position_error:8.3f}\n\n"

            f"True v:    {true_velocity:8.3f}\n"
            f"Estimate v:{estimated_velocity:8.3f}\n"
            f"Error v:   {velocity_error:8.3f}\n\n"

            "SENSOR\n"
            "----------------\n"
            f"z:         {measurement:8.3f}\n"
            f"Innovation:{innovation:8.3f}\n\n"

            "COVARIANCE P\n"
            "----------------\n"
            f"Pxx:       {covariance[0, 0]:8.3f}\n"
            f"Pxv:       {covariance[0, 1]:8.3f}\n"
            f"Pvx:       {covariance[1, 0]:8.3f}\n"
            f"Pvv:       {covariance[1, 1]:8.3f}\n\n"

            "UNCERTAINTY\n"
            "----------------\n"
            f"σx:        {position_sigma:8.3f}\n"
            f"σv:        {velocity_sigma:8.3f}\n\n"

            "KALMAN GAIN\n"
            "----------------\n"
            f"Kx:        {position_gain:8.3f}\n"
            f"Kv:        {velocity_gain:8.3f}"
        )

        self.ax_info.text(
            0.02,
            0.98,
            info,
            transform=self.ax_info.transAxes,
            verticalalignment="top",
            family="monospace",
            fontsize=10,
        )