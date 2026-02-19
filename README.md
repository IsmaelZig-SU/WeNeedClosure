📊 Dataset Description

The dataset consists of inputs and targets, both representing the dynamical states of a Couette flow at Reynolds number 1000.
The inputs are generated using a Galerkin projection with 375 modes, referred to in the code as ROM (Reduced-Order Model).
The targets are generated using a Galerkin projection with 900 modes, referred to in the code as FOM (Full-Order Model).
The goal of the diffusion model is to learn a mapping from the ROM distribution to the FOM distribution in a:

  - purely data-driven manner
  - model-agnostic way
  - and time-independent fashion.

📐 Data Shapes

The data tensors have the following shapes:

ROM  : [1, T, D_rom]
FOM  : [1, T, D_fom]

where:

T is the number of time snapshots,

D_rom is the ROM modal dimension (375 in the provided example),

D_fom is the FOM modal dimension (900 in the provided example).

📁 Data Location

Both datasets are stored in the Data/ folder.
If the dataset files are renamed or moved, please update the corresponding command-line arguments:

--input_path for the ROM dataset
--target_path for the FOM dataset

accordingly.


## 🌫 Diffusion Model Overview

The objective of the diffusion model is to learn a stochastic mapping from the **ROM distribution** (low-dimensional Galerkin projection) to the **FOM distribution** (high-dimensional Galerkin projection) using a conditional generative process.

---

### 🔁 Forward (Noising) Process — DDPM

We follow the Denoising Diffusion Probabilistic Model (DDPM) framework.  
Starting from a clean FOM sample \( x_0 \), Gaussian noise is progressively added according to the forward process:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1-\beta_t}\x_{t-1}, \beta_t I\right)
$$

which admits the closed-form expression:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t}\x_0, (1-\bar{\alpha}_t)I\right)
$$

with

$$
\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s.
$$

This defines the **noising schedule**, which gradually transforms structured FOM data into pure Gaussian noise.

---

### 🔄 Reverse (Denoising) Process — SDE Formulation

The forward process can be interpreted as a stochastic differential equation (SDE):

$$
dx = f(x,t)\,dt + g(t)\,dW_t.
$$

The goal of diffusion modeling is to learn the corresponding reverse-time SDE:

$$
dx = \left[ f(x,t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t)\ d\bar{W}_t.
$$

Here, the score function (gradient of the log-density), and \( d\bar{W}_t \) denotes a reverse-time Wiener process.

The reverse-time dynamics are derived from the **Fokker–Planck equation**, which governs the evolution of the probability density \( p_t(x) \).  
Learning the score function allows the diffusion process to be inverted and samples from the target FOM distribution to be generated.

---

### 🎯 Conditional Generation with ROM Prior

In this work, the reverse diffusion process is conditioned on the ROM state, which is treated as a prior.  
The model therefore learns the conditional score:

$$
\nabla_x \log p_t(x \mid \mathrm{ROM}).
$$

This enables sampling from the FOM distribution given a ROM input, yielding a **data-driven, model-agnostic, and time-independent closure mapping** from ROM to FOM.

---

### 🤖 Transformer Denoiser with Attention

The score function (denoiser) is parameterized by a Transformer network.

- Noisy FOM states are embedded as **noise tokens**.
- ROM states are embedded as **conditioning tokens**.

The architecture employs:
- **Self-attention** on the noisy FOM tokens to capture global correlations in the FOM state.
- **Cross-attention** between noisy FOM tokens (queries) and ROM tokens (keys and values) to inject ROM information into the denoising process.

---

### 🧠 Attention Mechanism

Attention computes weighted combinations of tokens based on similarity:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

- **Self-attention**: \(Q,K,V\) come from the same sequence (FOM tokens).
- **Cross-attention**: \(Q\) comes from FOM tokens, while \(K,V\) come from ROM tokens.

This allows the model to dynamically focus on the most relevant ROM modes when reconstructing FOM states.

---

### 📚 References

- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020.  
- Vaswani et al., *Attention Is All You Need*, 2017.
