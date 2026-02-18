import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            Y_mag = batch['Y_mag'].to(self.device)
            X_mag = batch['X_mag'].to(self.device)
            T_mag = batch['T_mag'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(Y_mag, X_mag)
            
            targets = {'target_mag': T_mag}
            loss = self.model.compute_loss(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                Y_mag = batch['Y_mag'].to(self.device)
                X_mag = batch['X_mag'].to(self.device)
                T_mag = batch['T_mag'].to(self.device)
                
                outputs = self.model(Y_mag, X_mag)
                
                targets = {'target_mag': T_mag}
                loss = self.model.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs=50):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            self.save_checkpoint(epoch, val_loss, is_best=False)
        
        print(f'\nTraining complete! Best Val Loss: {best_val_loss:.6f}')
        
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint['epoch'], checkpoint['loss']