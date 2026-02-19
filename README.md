📊 Dataset Description

The dataset consists of inputs and targets, both representing the dynamical states of a Couette flow at Reynolds number 1000.

The inputs are generated using a Galerkin projection with 375 modes, referred to in the code as ROM (Reduced-Order Model).

The targets are generated using a Galerkin projection with 900 modes, referred to in the code as FOM (Full-Order Model).

The goal of the diffusion model is to learn a mapping from the ROM distribution to the FOM distribution in a:

purely data-driven manner,

model-agnostic way,

and time-independent fashion.

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
