#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, threading, time
import numpy as np
import simpy

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QSlider, QCheckBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from models.config import DeviceParams, GlobalParams, WorkloadProfile
from models.device_model import VRDevice
from models.scheduler import CooperativeScheduler, IndependentScheduler
from models.utils import draw_heterogeneity


# ==========================================================
# Realtime matplotlib plot
# ==========================================================
class LivePlot(FigureCanvasQTAgg):
    def __init__(self, num_devices, title, key):
        fig = Figure(figsize=(6, 4))
        super().__init__(fig)
        self.ax = self.figure.add_subplot(111)

        self.num_devices = num_devices
        self.title = title
        self.key = key

        self.t = []
        self.data = {f"dev{i}": [] for i in range(num_devices)}
        self.max_points = 200

    def update_plot(self, logs, now):
        self.t.append(now)
        for i in range(self.num_devices):
            val = logs.get(f"dev{i}_{self.key}", 0)
            self.data[f"dev{i}"].append(val)

        if len(self.t) > self.max_points:
            self.t = self.t[-self.max_points:]
            for k in self.data:
                self.data[k] = self.data[k][-self.max_points:]

        self.ax.cla()
        for i in range(self.num_devices):
            self.ax.plot(self.t, self.data[f"dev{i}"], label=f"dev{i}")

        self.ax.set_title(self.title)
        self.ax.set_xlabel("Sim Time (s)")
        self.ax.grid(True)
        self.ax.legend()
        self.draw()


# ==========================================================
# Background Simulation Thread
# ==========================================================
class SimulationThread(threading.Thread):
    def __init__(self, num_devices, control, logs, mode="cooperative"):
        super().__init__()
        self.control = control
        self.logs = logs
        self.running = True

        # Create environment
        self.env = simpy.Environment()

        dp = DeviceParams()
        wp = WorkloadProfile()
        gp = GlobalParams(num_devices=num_devices, sim_time=99999)

        rng = np.random.default_rng(42)

        # Create devices
        self.devices = [
            VRDevice(self.env, i, gp, dp, wp, rng, mode == "cooperative", draw_heterogeneity(rng))
            for i in range(num_devices)
        ]

        # Scheduler
        if mode == "cooperative":
            CooperativeScheduler(self.env, self.devices, gp, dp)
        else:
            IndependentScheduler(self.env, self.devices, gp, dp)

    def run(self):
        while self.running:
            # Apply GUI control
            for i, dev in enumerate(self.devices):
                dev.external_load_factor = self.control[f"dev{i}_load"]
                dev.offline = self.control[f"dev{i}_offline"]

            # Step simulation
            self.env.step()

            # Collect logs
            for i, dev in enumerate(self.devices):
                self.logs[f"dev{i}_lat"] = dev.logs["latency"][-1] if dev.logs["latency"] else 0
                self.logs[f"dev{i}_power"] = dev.logs["power"][-1] if dev.logs["power"] else 0
                self.logs[f"dev{i}_temp"] = dev.logs["temp"][-1] if dev.logs["temp"] else 0

            time.sleep(0.03)


# ==========================================================
# GUI Control Panel
# ==========================================================
class ControlGUI(QWidget):
    def __init__(self, control, logs, num_devices, sim_thread):
        super().__init__()
        self.control = control
        self.logs = logs
        self.sim_thread = sim_thread
        self.num_devices = num_devices

        layout = QHBoxLayout()

        # Left Control
        left = QVBoxLayout()
        self.labels = []

        for i in range(num_devices):
            lbl = QLabel(f"Device {i}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(10, 300)
            slider.setValue(100)
            slider.valueChanged.connect(lambda v, d=i: self.update_load(d, v))

            chk = QCheckBox("Offline")
            chk.stateChanged.connect(lambda s, d=i: self.update_offline(d, s))

            left.addWidget(lbl)
            left.addWidget(slider)
            left.addWidget(chk)
            self.labels.append(lbl)

        layout.addLayout(left)

        # Right plots
        right = QVBoxLayout()
        self.plot_lat = LivePlot(num_devices, "Latency", "lat")
        self.plot_pwr = LivePlot(num_devices, "Power", "power")
        self.plot_tmp = LivePlot(num_devices, "Temperature", "temp")

        right.addWidget(self.plot_lat)
        right.addWidget(self.plot_pwr)
        right.addWidget(self.plot_tmp)

        layout.addLayout(right)
        self.setLayout(layout)

        # Timer to refresh
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(150)

    def update_load(self, dev, value):
        self.control[f"dev{dev}_load"] = value / 100.0

    def update_offline(self, dev, state):
        self.control[f"dev{dev}_offline"] = (state == Qt.Checked)

    def refresh(self):
        now = self.sim_thread.env.now
        self.plot_lat.update_plot(self.logs, now)
        self.plot_pwr.update_plot(self.logs, now)
        self.plot_tmp.update_plot(self.logs, now)

        for i in range(self.num_devices):
            lat = self.logs.get(f"dev{i}_lat", 0)
            pwr = self.logs.get(f"dev{i}_power", 0)
            tmp = self.logs.get(f"dev{i}_temp", 0)
            self.labels[i].setText(
                f"Device {i}: lat={lat:.1f} | pwr={pwr:.1f} | temp={tmp:.1f}"
            )


# ==========================================================
# Main
# ==========================================================
def main():
    num_devices = 5

    control = {}
    for i in range(num_devices):
        control[f"dev{i}_load"] = 1.0
        control[f"dev{i}_offline"] = False

    logs = {}

    sim_thread = SimulationThread(num_devices, control, logs, mode="cooperative")
    sim_thread.start()

    app = QApplication(sys.argv)
    gui = ControlGUI(control, logs, num_devices, sim_thread)
    gui.show()
    app.exec_()

    sim_thread.running = False


if __name__ == "__main__":
    main()
