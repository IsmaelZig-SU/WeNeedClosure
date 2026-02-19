import torch
from tqdm import tqdm
import torch.nn as nn

class Train_Loop():

    def __init__(self, exp_parent, model, train_loader, test_loader):
    
        self.exp = exp_parent
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.mse = nn.MSELoss()
        
        self.best_test_loss = float("inf")

    def noising_schedule(self, T, device, beta_start=0.0001, beta_end=0.02):
        
        scale = 1000 / T
        betas = torch.linspace(beta_start, beta_end, T, device=device) * scale
        
        return torch.clamp(betas, max=0.999)


    def get_lr(self, epoch):

        if epoch < 100 :
            return 1e-3

        elif epoch < 200:
            return 1e-4

        elif epoch < 300:
            return 5e-5

        else :
            return 1e-5

    def q_sample(self, x0, t, noise=None):

        """
        Sample noisy image x_t given x_0 and timestep t using:
        x_t = sqrt(alphas_cumprod[t]) * x_0 + sqrt(1 - alphas_cumprod[t]) * noise
        """

        if noise is None:
            noise = torch.randn_like(x0)

        alpha_t = self.alphas_cumprod[t].sqrt()
        one_minus_alpha_t = (1 - self.alphas_cumprod[t]).sqrt()

        while alpha_t.ndim < x0.ndim:

            alpha_t = alpha_t.unsqueeze(-1)
            one_minus_alpha_t = one_minus_alpha_t.unsqueeze(-1)

        return alpha_t * x0 + one_minus_alpha_t * noise

    def run(self):

        self.betas = self.noising_schedule(self.exp.denoise_steps, self.exp.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        for epoch in range(self.exp.epochs):
            lr = self.get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # --- TRAIN PHASE ---
            avg_train_loss = self.train_epoch(epoch, lr)

            # --- TEST PHASE ---
            avg_test_loss = self.test_epoch(epoch)

            # --- SAVE LOGIC ---
            if avg_test_loss < self.best_test_loss:
                self.best_test_loss = avg_test_loss
                best_state_dict = self.model.state_dict()  # <-- copy best weights
                print(f"✅ New best model : {self.best_test_loss:.6f}")

        self.model.load_state_dict(best_state_dict)

        return self.model, self.alphas.cpu(), self.betas.cpu(), self.alphas_cumprod.cpu()


    def train_epoch(self, epoch, lr):

        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        
        for rom, fom in pbar:
            rom, fom = rom.to(self.exp.device), fom.to(self.exp.device)
            
            t = torch.randint(0, self.exp.denoise_steps, (rom.shape[0],), device=self.exp.device)
            noise = torch.randn_like(fom)
            
            x_t = self.q_sample(fom, t, noise=noise) 
            
            eps_pred = self.model(x_t, rom)
            loss = self.mse(eps_pred, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
            
        return total_loss / len(self.train_loader)

    def test_epoch(self, epoch):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            pbar = tqdm(self.test_loader, desc=f"Epoch {epoch+1} [TEST]")

            for rom, fom in pbar:

                rom, fom = rom.to(self.exp.device), fom.to(self.exp.device)
                t = torch.randint(0, self.exp.denoise_steps, (rom.shape[0],), device=self.exp.device)
                noise = torch.randn_like(fom)
                
                x_t = self.q_sample(fom, t, noise=noise)
                eps_pred = self.model(x_t, rom)
                loss = self.mse(eps_pred, noise)

                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))
                
        return total_loss / len(self.test_loader)