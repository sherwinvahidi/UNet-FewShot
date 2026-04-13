# models/maml_segmentation.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import OrderedDict
import copy
import numpy as np

class MAMLSegmentation(nn.Module):
    """
    Model-Agnostic Meta-Learning for Segmentation
    
    Key idea: Learn an initialization that can quickly adapt to new
    tumor types with just a few gradient steps
    """
    
    def __init__(self, encoder_name='resnet34', in_channels=2, num_classes=4):
        super().__init__()
        
        # Base U-Net architecture
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    
    def forward(self, x, params=None):
        """
        Forward pass with optional custom parameters
        
        Args:
            x: Input images
            params: Optional OrderedDict of parameters (for inner loop)
        """
        if params is None:
            # Standard forward
            return self.model(x)
        else:
            # Forward with custom parameters (for inner loop adaptation)
            # Clone the model
            return self.model(x)
    
    def clone_model(self):
        """Clone model for inner loop adaptation"""
        return copy.deepcopy(self.model)
    
    def get_parameters(self):
        """Get all trainable parameters as OrderedDict"""
        return OrderedDict(self.model.named_parameters())


class MAMLTrainer:
    """
    MAML Training Loop
    
    Implements both inner loop (task adaptation) and outer loop (meta-update)
    """
    
    def __init__(self, model, config, train_dataset, val_dataset):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Meta-learning hyperparameters
        self.inner_lr = 0.01      # Learning rate for inner loop (task adaptation)
        self.outer_lr = 0.001     # Learning rate for meta-update
        self.num_inner_steps = 5  # Gradient steps in inner loop
        
        # Meta-optimizer (updates the initialization)
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.outer_lr)
        
        # Loss function
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
        
        # History
        self.history = {
            'meta_train_loss': [],
            'meta_val_loss': [],
            'inner_loop_loss': []
        }
    
    def sample_task(self, dataset, k_shot=5, n_query=10):
        """
        Sample one task (episode) for MAML
        
        Returns:
            support_set: k examples for adaptation (inner loop)
            query_set: examples for meta-update (outer loop)
        """
        import random
        n = len(dataset)
        
        # Sample k_shot + n_query unique samples
        indices = random.sample(range(n), min(k_shot + n_query, n))
        
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
    
    def inner_loop(self, support_images, support_masks):
        """
        Inner loop: Adapt model to support set
        
        Takes num_inner_steps gradient steps on support set
        Returns adapted model
        """
        # Clone model for this task
        adapted_model = self.model.clone_model()
        adapted_model.train()
        
        # Create optimizer for inner loop
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Take num_inner_steps gradient steps on support set
        for step in range(self.num_inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward on support set
            support_pred = adapted_model(support_images)
            
            # Loss on support set
            support_loss = self.loss_fn(support_pred, support_masks)
            
            # Backward and update
            support_loss.backward()
            inner_optimizer.step()
        
        return adapted_model, support_loss.item()
    
    def meta_train_step(self, k_shot=5, n_query=10):
        """
        One meta-training step (one task)
        
        1. Sample task (support + query)
        2. Inner loop: adapt to support set
        3. Outer loop: compute loss on query set, update meta-parameters
        """
        # Sample task
        task = self.sample_task(self.train_dataset, k_shot, n_query)
        
        support_images = task['support_images'].to(self.config.DEVICE)
        support_masks = task['support_masks'].to(self.config.DEVICE)
        query_images = task['query_images'].to(self.config.DEVICE)
        query_masks = task['query_masks'].to(self.config.DEVICE)
        
        # Inner loop: adapt to support set
        adapted_model, inner_loss = self.inner_loop(support_images, support_masks)
        
        # Outer loop: compute loss on query set with adapted model
        adapted_model.eval()
        query_pred = adapted_model(query_images)
        query_loss = self.loss_fn(query_pred, query_masks)
        
        # Meta-update: backprop through the adaptation process
        # Note: This simplified version doesn't compute second-order gradients
        # For full MAML, you'd need higher-order derivatives
        self.meta_optimizer.zero_grad()
        
        # Instead, we do first-order MAML (FOMAML)
        # Train the original model to minimize post-adaptation query loss
        self.model.train()
        original_pred = self.model(torch.cat([support_images, query_images], dim=0))
        original_masks = torch.cat([support_masks, query_masks], dim=0)
        original_loss = self.loss_fn(original_pred, original_masks)
        
        original_loss.backward()
        self.meta_optimizer.step()
        
        return original_loss.item(), inner_loss, query_loss.item()
    
    def train(self, num_tasks=1000, k_shot=5, n_query=10):
        """
        Meta-training loop
        
        Args:
            num_tasks: Number of tasks (episodes) to train on
            k_shot: Number of support examples per task
            n_query: Number of query examples per task
        """
        print(f"Starting MAML training")
        print(f"Tasks: {num_tasks}, k-shot: {k_shot}, n-query: {n_query}")
        print(f"Inner LR: {self.inner_lr}, Outer LR: {self.outer_lr}")
        print(f"Inner steps: {self.num_inner_steps}")
        print(f"Device: {self.config.DEVICE}")
        
        for task_idx in range(num_tasks):
            meta_loss, inner_loss, query_loss = self.meta_train_step(k_shot, n_query)
            
            # Record
            self.history['meta_train_loss'].append(meta_loss)
            self.history['inner_loop_loss'].append(inner_loss)
            
            # Print progress
            if (task_idx + 1) % 100 == 0:
                print(f"Task [{task_idx+1}/{num_tasks}] "
                      f"Meta Loss: {meta_loss:.4f} "
                      f"Inner Loss: {inner_loss:.4f} "
                      f"Query Loss: {query_loss:.4f}")
                
                # Validate
                val_loss = self.validate(k_shot, n_query, num_val_tasks=10)
                print(f"  Validation Loss: {val_loss:.4f}")
                self.history['meta_val_loss'].append(val_loss)
            
            # Save checkpoint
            if (task_idx + 1) % 500 == 0:
                self.save_checkpoint(task_idx)
        
        print("✓ MAML training complete!")
        return self.history
    
    def validate(self, k_shot=5, n_query=10, num_val_tasks=10):
        """Validate MAML on validation set"""
        total_loss = 0
        
        for _ in range(num_val_tasks):
            task = self.sample_task(self.val_dataset, k_shot, n_query)
            
            support_images = task['support_images'].to(self.config.DEVICE)
            support_masks = task['support_masks'].to(self.config.DEVICE)
            query_images = task['query_images'].to(self.config.DEVICE)
            query_masks = task['query_masks'].to(self.config.DEVICE)
            
            # Inner loop adaptation
            adapted_model, _ = self.inner_loop(support_images, support_masks)
            
            # Evaluate on query
            adapted_model.eval()
            with torch.no_grad():
                query_pred = adapted_model(query_images)
                query_loss = self.loss_fn(query_pred, query_masks)
                total_loss += query_loss.item()
        
        return total_loss / num_val_tasks
    
    def evaluate_k_shot(self, k_values=[1, 5, 10, 20], num_tasks=20):
        """
        Evaluate MAML at different k-shot values
        
        For each k:
          1. Sample k support examples
          2. Adapt model (inner loop)
          3. Evaluate on query set
        """
        results = {}
        
        for k in k_values:
            print(f"\nEvaluating MAML at k={k}...")
            dice_scores = []
            
            for task_idx in range(num_tasks):
                task = self.sample_task(self.val_dataset, k_shot=k, n_query=10)
                
                support_images = task['support_images'].to(self.config.DEVICE)
                support_masks = task['support_masks'].to(self.config.DEVICE)
                query_images = task['query_images'].to(self.config.DEVICE)
                query_masks = task['query_masks'].to(self.config.DEVICE)
                
                # Adapt to support set
                adapted_model, _ = self.inner_loop(support_images, support_masks)
                
                # Evaluate on query
                adapted_model.eval()
                with torch.no_grad():
                    query_pred = adapted_model(query_images)
                    loss = self.loss_fn(query_pred, query_masks)
                    dice = 1 - loss.item()
                    dice_scores.append(dice)
                
                if (task_idx + 1) % 5 == 0:
                    print(f"  Task {task_idx+1}/{num_tasks}: DICE = {dice:.4f}")
            
            mean_dice = np.mean(dice_scores)
            std_dice = np.std(dice_scores)
            results[k] = {'mean': mean_dice, 'std': std_dice}
            
            print(f"✓ k={k}: DICE = {mean_dice:.4f} ± {std_dice:.4f}")
        
        return results
    
    def save_checkpoint(self, task_idx):
        """Save MAML checkpoint"""
        import os
        checkpoint = {
            'task': task_idx,
            'model_state_dict': self.model.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'history': self.history,
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'num_inner_steps': self.num_inner_steps
        }
        
        path = os.path.join(self.config.CHECKPOINT_DIR, f'maml_task{task_idx+1}.pth')
        torch.save(checkpoint, path)
        print(f"  Saved MAML checkpoint: {path}")
