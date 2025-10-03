# Experimental Framework for Non-Linear Material Discovery

This document outlines a complete workflow for using a Physics-Informed Neural Network (PINN) to discover non-linear material properties from experimental data.

## 1. The "Test" for Non-Linearity (The Theory)

The core idea is to hypothesize a more complex physical model than simple linear elasticity and let the experimental data validate or parameterize it.

### The Hypothesis
We assume the material's stress-strain relationship might be described by a simple non-linear model, such as the [Duffing oscillator's](https://en.wikipedia.org/wiki/Duffing_equation) restoring force:

**σ = E * ε + k * ε³**

Where:
- **σ** is stress.
- **ε** is strain.
- **E** is the linear Young's Modulus.
- **k** is the non-linear stiffness parameter.

### The Test
The "test" is to build a PINN where `E` and `k` are both **trainable variables**. We then train this PINN against real-world experimental data.

After training, we inspect the discovered value of `k`:
- If `k` converges to a value very close to zero, it implies the non-linear term is insignificant, and the material behaves linearly under the tested conditions.
- If `k` converges to a significant, stable, non-zero value, we have strong evidence of non-linear behavior and have successfully quantified it with the `E` and `k` parameters.

## 2. The Physical Rig Design

To generate the necessary data, a carefully designed physical experiment is required.

![Test Rig Diagram](https://i.imgur.com/k8d2Z1g.png)

### Components
1.  **Specimen:** A flat, rectangular "coupon" of the material being tested. Its surface must be prepared with a **speckle pattern** (a random spray of black dots on a white background). This is essential for the imaging system to track deformation.
2.  **Clamping Mechanism:** A rigid, heavy-duty vise to clamp one end of the specimen, creating a cantilever beam. This enforces a known "zero displacement" and "zero rotation" boundary condition.
3.  **Loading System:** A precise actuator or a hanger for calibrated weights to apply a known force (`P`) at the free end of the specimen. A load cell should be used to get an exact reading of the applied force.
4.  **Imaging System (Digital Image Correlation - DIC):**
    *   **Cameras:** One or (ideally) two high-resolution machine vision cameras mounted on a stable, vibration-free tripod. A stereo-camera (two-camera) setup allows for the measurement of 3D deformation (including out-of-plane motion).
    *   **Lighting:** Stable, diffuse, and consistent lighting is critical. Two angled softbox lights are recommended to eliminate shadows and glare on the specimen.
5.  **Control & Acquisition:** A central computer synchronizes the loading, image capture, and data recording from the load cell.

## 3. The Data Pipeline

This describes the flow of data from the physical rig to the PINN.

1.  **Image Acquisition:** The rig captures a sequence of high-resolution images of the speckled specimen at various load increments (e.g., 0N, 5N, 10N, ...).
2.  **DIC Processing:** The sequence of images is processed by DIC software (e.g., open-source `Ncorr` in MATLAB or commercial software like GOM Correlate).
    *   The software tracks the displacement of thousands of virtual "subsets" on the speckle pattern between the unloaded and loaded images.
    *   The output of this stage is a structured data file (e.g., `experimental_data.csv`).
3.  **Data Formatting:** The output data file must contain the following information for each measured point:
    *   `X`: The initial x-coordinate of the point (in meters).
    *   `Y`: The initial y-coordinate of the point (in meters).
    *   `u_x`: The measured displacement of that point in the x-direction (in meters).
    *   `u_y`: The measured displacement of that point in the y-direction (in meters).
4.  **PINN Ingestion:** The Python script for the PINN reads this `.csv` file, which provides the ground-truth data for the `Loss_data` term in the total loss function.
