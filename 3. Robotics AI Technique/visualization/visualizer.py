import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self):
        self.times = []
        self.true_positions = []
        self.measurements = []
        self.estimates = []
        self.uncertainties = []

        plt.ion()

        self.fig, self.ax = plt.subplots()

    def update(
        self,
        time,
        true_position,
        measurement,
        estimate,
        uncertainty,
    ):
        self.times.append(time)
        self.true_positions.append(true_position)
        self.measurements.append(measurement)
        self.estimates.append(estimate)
        self.uncertainties.append(uncertainty)

        self.ax.clear()

        self.ax.plot(
            self.times,
            self.true_positions,
            label="True position",
        )

        self.ax.scatter(
            self.times,
            self.measurements,
            label="Measurement",
            s=15,
        )

        self.ax.plot(
            self.times,
            self.estimates,
            label="Kalman estimate",
        )

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Position")
        self.ax.legend()
        self.ax.grid()

        plt.pause(0.01)