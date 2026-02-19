import torch
from src.load_data import Load_Data
from src.model import Diffusion_Transformer
from src.training_loop import Train_Loop

class DiffExp():

	def __init__(self,args):

   
		if torch.cuda.is_available():
			self.device = torch.device("cuda")

		elif torch.backends.mps.is_available():
			self.device = torch.device("mps") 

		else:
			
			self.device = torch.device("cpu")

		print(f'Running experiment on : {self.device}')

		self.input_path   = args.input_path
		self.target_path  = args.target_path
		self.exp_dir      = args.exp_dir
		self.info         = args.info

		# Diffusion Model Hyperparameters
		self.denoise_steps = args.denoise_steps
		self.d_model       = args.d_model
		self.n_heads       = args.n_heads
		self.n_layers      = args.n_layers

		# Training Routine
		self.test_size     = args.test_size
		self.batch_size    = args.batch_size
		self.epochs        = args.epochs	
		self.exp_name      = f"Diffusion_exp_diffsteps{self.denoise_steps}_emb_dim{self.d_model}_nblocks{self.n_layers}_{self.info}"

		torch.cuda.empty_cache()


	def main_train(self) : 

		#Loading and visualising data
		print("########## LOADING DATASET ##########")
		data = Load_Data(self)
		self.rom, self.fom, train_loader, test_loader, self.D_rom, self.D_fom = data.load_data()
		snapshots = self.rom.shape[-1]
		test_snapshot = int((1-self.test_size)* snapshots)
		trajs = self.rom.shape[0]
		test_snaps = snapshots - test_snapshot

		print("\n" + "="*50)
		print(f"{'DATASET SUMMARY':^50}")
		print("="*50)
		print(f"• Trajectories:    {trajs}")
		print(f"• Total Snapshots: {snapshots}")
		print(f"  └─ Training:     {test_snapshot}")
		print(f"  └─ Testing:      {test_snaps}")
		print("-"*50)
		print(f"• Architecture:    {self.D_rom} (ROM) → {self.D_fom} (FOM)")
		print("="*50 + "\n")

		model = Diffusion_Transformer(
		    fom_dim= self.D_fom,
		    rom_dim= self.D_rom,
		    d_model=self.d_model,
		    nheads=self.n_heads,
		    nlayers=self.n_layers
		).to(self.device)

		print("Total learnable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

		trainer = Train_Loop(self, model, train_loader, test_loader)
		best_model, alphas, betas, alphas_cumprod = trainer.run()

		save_dict = {
    		'model_state_dict': best_model.state_dict(),
	    	'model_config': {
		        'fom_dim': self.D_fom,
		        'rom_dim': self.D_rom,
		        'd_model': self.d_model,
		        'nheads': self.n_heads,
		        'nlayers': self.n_layers
		    	},
		    'scheduler_config': {
		        'betas': betas,
		        'alphas' : alphas,
		        'alphas_cumprod': alphas_cumprod, 
		        'T' : self.denoise_steps
		    	},
		    'test_data' : {
			    'rom' : self.rom[:, :, test_snapshot:], 
			    'fom' : self.fom[:, :, test_snapshot:]
			    }
			}

		torch.save(save_dict, f"{self.exp_name}.pt")
		print(f"✅ Best model saved in : {self.exp_name}")



