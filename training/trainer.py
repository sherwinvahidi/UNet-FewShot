# training/trainer.py
import torch
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
 
        # Loss — Dice loss for multiclass segmentation
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
 
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE
        )
 
        # Scheduler — halve LR after 3 epochs without improvement
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
 
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
 
    # ── Training ─────────────────────────────────────────────────
 
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
 
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}"
        )
 
        for batch in pbar:
            images = batch['image'].to(self.config.DEVICE)
            masks = batch['mask'].to(self.config.DEVICE)
 
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, masks)
            loss.backward()
            self.optimizer.step()
 
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
 
        return total_loss / len(self.train_loader)
 
    # ── Validation ───────────────────────────────────────────────
 
    def validate(self):
        self.model.eval()
        total_loss = 0
 
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.config.DEVICE)
                masks = batch['mask'].to(self.config.DEVICE)
 
                output = self.model(images)
                loss = self.loss_fn(output, masks)
                total_loss += loss.item()
 
        return total_loss / len(self.val_loader)
 
    # ── Main loop ────────────────────────────────────────────────
 
    def train(self):
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Train batches: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader)}")
 
        best_val_loss = float('inf')
 
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
 
            self.scheduler.step(val_loss)
 
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
 
            # Epoch summary
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'=' * 50}\n")
 
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
 
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
 
        print("✓ Training complete!")
        return self.history
 
    # ── Checkpointing ────────────────────────────────────────────
 
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
 
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch + 1}.pth'
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)
        print(f"✓ Saved: {filename}")