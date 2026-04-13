# training/prototypical_trainer.py (COMPLETE FIXED VERSION)
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import random
import os

class PrototypicalTrainer:
    """
    Trainer for Prototypical Networks with episodic training
    """
    
    def __init__(self, model, config, train_dataset, val_dataset, test_loader=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_loader = test_loader
        
        # Loss
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'k_shot_performance': {}
        }

    def _freeze_batchnorm(self):
        """Keep BN in eval mode — batch stats unreliable with small episodes."""
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()
    
    def sample_episode(self, dataset, k_shot=5, n_query=10):
        """Sample one episode from dataset"""
        n = len(dataset)
        
        # Sample k_shot + n_query unique samples
        total_needed = k_shot + n_query
        if total_needed > n:
            # If not enough samples, sample with replacement
            indices = random.choices(range(n), k=total_needed)
        else:
            indices = random.sample(range(n), total_needed)
        
        support_indices = indices[:k_shot]
        query_indices = indices[k_shot:k_shot+n_query]
        
        # Load support set
        support_images, support_masks = [], []
        for idx in support_indices:
            sample = dataset[idx]
            support_images.append(sample['image'])
            support_masks.append(sample['mask'])
        
        # Load query set
        query_images, query_masks = [], []
        for idx in query_indices:
            sample = dataset[idx]
            query_images.append(sample['image'])
            query_masks.append(sample['mask'])
        
        return {
            'support_images': torch.stack(support_images),
            'support_masks': torch.stack(support_masks),
            'query_images': torch.stack(query_images),
            'query_masks': torch.stack(query_masks)
        }
    
    def train_episode(self, k_shot=5, n_query=10):
        """Train on one episode with prototype-attention"""
        self.model.train()
        self._freeze_batchnorm()
        
        # Sample episode
        episode = self.sample_episode(self.train_dataset, k_shot, n_query)
        
        support_images = episode['support_images'].to(self.config.DEVICE)
        support_masks = episode['support_masks'].to(self.config.DEVICE)
        query_images = episode['query_images'].to(self.config.DEVICE)
        query_masks = episode['query_masks'].to(self.config.DEVICE)
        
        # Use prototype-attention forward
        outputs = self.model.forward_with_prototype_attention(
            query_images, support_images, support_masks
        )
        
        # Compute loss (only on query set)
        loss = self.loss_fn(outputs, query_masks)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=1000, k_shot=5, n_query=10):
        """
        Train Prototypical Networks with episodic training
        """
        print(f"Starting Prototypical Networks training")
        print(f"Episodes: {num_episodes}, k-shot: {k_shot}, n-query: {n_query}")
        print(f"Device: {self.config.DEVICE}")
        
        for episode_idx in range(num_episodes):
            loss = self.train_episode(k_shot, n_query)
            
            # Print progress
            if (episode_idx + 1) % 100 == 0:
                print(f"Episode [{episode_idx+1}/{num_episodes}] Loss: {loss:.4f}")
                
                # Validate every 100 episodes
                val_loss = self.validate(k_shot, n_query, num_val_episodes=10)
                print(f"  Validation Loss: {val_loss:.4f}")
                
                self.history['train_loss'].append(loss)
                self.history['val_loss'].append(val_loss)
            
            # Save checkpoint every 500 episodes
            if (episode_idx + 1) % 500 == 0:
                self.save_checkpoint(episode_idx)
        
        print("✓ Training complete!")
        return self.history
    
    def validate(self, k_shot=5, n_query=10, num_val_episodes=10):
        """Validate on validation set using prototype-attention"""
        self.model.eval()
        
        total_loss = 0
        for _ in range(num_val_episodes):
            episode = self.sample_episode(self.val_dataset, k_shot, n_query)
            
            support_images = episode['support_images'].to(self.config.DEVICE)
            support_masks = episode['support_masks'].to(self.config.DEVICE)
            query_images = episode['query_images'].to(self.config.DEVICE)
            query_masks = episode['query_masks'].to(self.config.DEVICE)
            
            with torch.no_grad():
                # Use new prototype-attention method
                query_pred = self.model.forward_with_prototype_attention(
                    query_images, support_images, support_masks
                )
                loss = self.loss_fn(query_pred, query_masks)
                total_loss += loss.item()
        
        return total_loss / num_val_episodes
    
    def evaluate_k_shot(self, k_values=[1, 5, 10, 20], num_episodes=20):
        """
        Evaluate few-shot performance at different k values
        """
        self.model.eval()
        results = {}
        
        for k in k_values:
            dice_scores = []
            
            for _ in tqdm(range(num_episodes), desc=f"Evaluating k={k}"):
                episode = self.sample_episode(self.val_dataset, k_shot=k, n_query=10)
                
                support_images = episode['support_images'].to(self.config.DEVICE)
                support_masks = episode['support_masks'].to(self.config.DEVICE)
                query_images = episode['query_images'].to(self.config.DEVICE)
                query_masks = episode['query_masks'].to(self.config.DEVICE)
                
                with torch.no_grad():
                    # Use prototype-attention method
                    query_pred = self.model.forward_with_prototype_attention(
                        query_images, support_images, support_masks
                    )
                    
                    # Dice = 1 - Dice loss
                    loss = self.loss_fn(query_pred, query_masks)
                    dice = 1 - loss.item()
                    dice_scores.append(dice)
            
            mean_dice = np.mean(dice_scores)
            std_dice = np.std(dice_scores)
            results[k] = {'mean': mean_dice, 'std': std_dice}
            
            print(f"k={k}: DICE = {mean_dice:.4f} ± {std_dice:.4f}")
        
        return results
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        path = os.path.join(self.config.CHECKPOINT_DIR, 
                           f'prototypical_ep{episode+1}.pth')
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")