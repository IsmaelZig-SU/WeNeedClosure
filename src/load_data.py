import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class TimeSplitDataset(Dataset):

    def __init__(self, rom, fom):

        self.rom = torch.tensor(rom, dtype=torch.float32)
        self.fom = torch.tensor(fom, dtype=torch.float32)
        self.T = self.rom.shape[0]

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return self.rom[idx, :], self.fom[idx, :]


class Load_Data():

	def __init__(self, args):
      
		self.input_path = args.input_path
		self.target_path = args.target_path
		self.batch_size = args.batch_size
		self.test_size = args.test_size

	def extract_dataset(self, filename, target_key):

		"""
		Search for a dataset named target_key in an HDF5 file
		and return it as a NumPy array.
		"""

		with h5py.File(filename, "r") as f:
			
			found = []

			def visitor(name, obj):
				if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == target_key:
					found.append(obj[()])

			f.visititems(visitor)

			if not found:

				raise KeyError(f"Dataset '{target_key}' not found in {filename}")

			return found[0]


	def load_data(self) :

		self.rom = self.extract_dataset(self.input_path, "q")
		self.fom = self.extract_dataset(self.target_path, "q_ref")

		N, D_rom, T = self.rom.shape
		_, D_fom, _ = self.fom.shape

		print(f"ROM shape: {self.rom.shape}")
		print(f"FOM shape: {self.fom.shape}")

		rom = self.rom.transpose(0, 2, 1).reshape(-1, D_rom)
		fom = self.fom.transpose(0, 2, 1).reshape(-1, D_fom)

		samples = rom.shape[0]
		split_idx = int(samples * (1-self.test_size))

		rom_train = rom[:split_idx, :]   # shape (T x N, D)
		rom_test = rom[split_idx:, :]    # shape (T x N, D)

		fom_train = fom[:split_idx, :]
		fom_test = fom[split_idx:, :]

		train_dataset = TimeSplitDataset(rom_train, fom_train)
		test_dataset = TimeSplitDataset(rom_test, fom_test)

		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

		return self.rom, self.fom, train_loader, test_loader, D_rom, D_fom
