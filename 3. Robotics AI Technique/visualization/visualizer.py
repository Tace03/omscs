import math

import matplotlib.pyplot as plt


class Visualizer:
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