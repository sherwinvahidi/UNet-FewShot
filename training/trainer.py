# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss functions
        self.seg_loss_fn = smp.losses.DiceLoss(mode='multiclass')
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_seg_loss': [],
            'val_seg_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
        }
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_seg_loss = 0
        total_cls_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.config.DEVICE)
            masks = batch['mask'].to(self.config.DEVICE)
            
            # Forward
            self.optimizer.zero_grad()
            seg_output, cls_output = self.model(images)
            
            # Segmentation loss
            seg_loss = self.seg_loss_fn(seg_output, masks)
            
            # Classification loss (dummy labels for now - just for testing)
            # In practice, you'd need actual tumor type labels
            batch_size = images.shape[0]
            dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(self.config.DEVICE)
            cls_loss = self.cls_loss_fn(cls_output, dummy_labels)
            
            # Total loss
            loss = seg_loss + self.config.CLS_LOSS_WEIGHT * cls_loss
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        
        return avg_loss, avg_seg_loss, avg_cls_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_seg_loss = 0
        total_cls_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.config.DEVICE)
                masks = batch['mask'].to(self.config.DEVICE)
                
                seg_output, cls_output = self.model(images)
                
                seg_loss = self.seg_loss_fn(seg_output, masks)
                
                # Dummy classification loss
                batch_size = images.shape[0]
                dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(self.config.DEVICE)
                cls_loss = self.cls_loss_fn(cls_output, dummy_labels)
                
                total_loss += (seg_loss + self.config.CLS_LOSS_WEIGHT * cls_loss).item()
                total_seg_loss += seg_loss.item()
                total_cls_loss += cls_loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = total_seg_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)
        
        return avg_loss, avg_seg_loss, avg_cls_loss
    
    def train(self):
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_seg_loss, train_cls_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_seg_loss, val_cls_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_seg_loss'].append(train_seg_loss)
            self.history['val_seg_loss'].append(val_seg_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Seg: {train_seg_loss:.4f} | Val Seg: {val_seg_loss:.4f}")
            print(f"Train Cls: {train_cls_loss:.4f} | Val Cls: {val_cls_loss:.4f}")
            print(f"{'='*60}\n")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("✓ Training complete!")
        self.plot_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {filename}")
    
    def plot_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Segmentation Loss
        axes[0, 1].plot(self.history['train_seg_loss'], label='Train Seg Loss')
        axes[0, 1].plot(self.history['val_seg_loss'], label='Val Seg Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Loss')
        axes[0, 1].set_title('Segmentation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Classification Loss
        axes[1, 0].plot(self.history['train_cls_loss'], label='Train Cls Loss')
        axes[1, 0].plot(self.history['val_cls_loss'], label='Val Cls Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cross Entropy Loss')
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Hide empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'training_history.png'))
        plt.show()
