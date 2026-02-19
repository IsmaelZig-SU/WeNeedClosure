import argparse
from src.diffusion_experiment import DiffExp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Diffusion')

    #Params

    parser.add_argument('--input_path', type = str, default = 'Data/Sol_small_ROM_Re1000.h5' , help = "Path to ROM data")
    parser.add_argument('--target_path', type = str, default = 'Data/Sol_ROM_Re1000.h5' , help = "Path to ROM data")
    parser.add_argument('--exp_dir', type = str, default = 'Trained_models/' , help = "Saving directory")
    parser.add_argument('--info', type = str, default = '' , help = "Additional informations")

    #----------------Diffusion model--------------------------#

    parser.add_argument('--denoise_steps', type = int, default = 1000 , help = "Number of noising/denoising diffusion steps")
    parser.add_argument('--d_model', type = int, default = 128 , help = "Embedding dimension : Inputs are projected to an embedding space, become tokens for attention to be applied upon")
    parser.add_argument('--n_heads', type = int, default = 4 , help = "Number of attention heads. Must be 2^n and must divide d_model")
    parser.add_argument('--n_layers', type = int, default = 4 , help = "Number of attention blocks in series")

    #----------------Training routine--------------------------#

    parser.add_argument('--test_size', type = float, default = 0.2 , help = "Fraction of data kept for test")
    parser.add_argument('--batch_size', type = int, default = 16 , help = "Batch size : increasing favors faster training, reducing favors lighter memory")
    parser.add_argument('--epochs', type = int, default = 100 , help = "Training epochs")


    args = parser.parse_args()
    #############################################################################

    diff = DiffExp(args)
    diff.main_train()