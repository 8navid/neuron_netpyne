# Simple Excitatory-Inhibitory Neural Network Demo

A beginner-friendly computational neuroscience project using NetPyNE and NEURON.

This project simulates a small excitatory-inhibitory (E/I) neural network and demonstrates how interactions between neurons can generate collective activity, spike patterns, and simple brain-like rhythms.

---

# Features

- Hodgkin-Huxley neuron model
- Excitatory and inhibitory populations
- Spike raster visualization
- Population firing rate analysis
- Power spectral density (PSD) analysis
- Automatic text report generation
- Animated spike raster GIF
- Simple and educational code structure

---

# Technologies

- Python
- NetPyNE
- NEURON
- NumPy
- SciPy
- Matplotlib

---

# Network Structure

| Population | Number of Cells | Function |
|---|---|---|
| E | 40 | Excitatory neurons |
| I | 10 | Inhibitory neurons |

Connections:
- E → E
- E → I
- I → E

---

# Generated Outputs

The simulation automatically creates:

| File | Description |
|---|---|
| `simulation_report.txt` | Simulation statistics and analysis |
| `network_summary.png` | Raster plot, population activity, PSD |
| `network_animation.gif` | Animated spike raster |

---

# Example Concepts Demonstrated

- Neural spiking
- Excitation vs inhibition
- Emergent network activity
- Oscillatory dynamics
- Brain rhythm generation
- Spectral analysis in neuroscience

---

# Installation

## 1. Install Python packages

```bash
pip install netpyne neuron matplotlib scipy numpy pillow
```

---

# Run the Simulation

```bash
python comp.py
```

---

# Example Visualization

The project generates:
- Spike raster plots
- Population activity plots
- Frequency spectrum analysis
- Animated network activity

---

# Educational Purpose

This project was created as a simple educational demo for computational neuroscience courses and beginner neural network simulations.

---

# License

MIT License


#
Provided by navid danaee 8navid@gmail.com
