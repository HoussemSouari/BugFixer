"""
Optimized training pipeline with fp16, gradient accumulation, early stopping,
and comprehensive monitoring for Java bug-fixing model.
"""

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

from transformers import get_linear_schedule_with_warmup
try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str = "Salesforce/codet5-base"
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_epochs: int = 15
    max_source_length: int = 512
    max_target_length: int = 512
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Advanced training
    use_fp16: bool = True
    num_beams: int = 5
    early_stopping: bool = True
    patience: int = 3
    
    # Scheduling
    lr_scheduler: str = 'linear'  # 'linear' or 'cosine'
    
    # Output
    output_dir: str = "./output"
    save_total_limit: int = 3
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Curriculum learning
    use_curriculum: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BugFixDataset(Dataset):
    """PyTorch Dataset for bug-fix pairs"""
    
    def __init__(
        self,
        buggy_codes: List[str],
        fixed_codes: List[str],
        bug_types: List[str],
        tokenizer,
        config: TrainingConfig,
        bug_type_to_id: Optional[Dict[str, int]] = None,
    ):
        self.buggy_codes = buggy_codes
        self.fixed_codes = fixed_codes
        self.bug_types = bug_types
        self.tokenizer = tokenizer
        self.config = config
        
        # Create bug type mapping
        if bug_type_to_id is None:
            unique_types = list(set(bug_types))
            self.bug_type_to_id = {t: i for i, t in enumerate(unique_types)}
        else:
            self.bug_type_to_id = bug_type_to_id
    
    def __len__(self) -> int:
        return len(self.buggy_codes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buggy = self.buggy_codes[idx]
        fixed = self.fixed_codes[idx]
        bug_type = self.bug_types[idx]
        
        # Tokenize input (buggy code)
        input_encodings = self.tokenizer(
            buggy,
            max_length=self.config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize target (fixed code)
        target_encodings = self.tokenizer(
            fixed,
            max_length=self.config.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(0),
            'attention_mask': input_encodings['attention_mask'].squeeze(0),
            'labels': target_encodings['input_ids'].squeeze(0),
            'bug_type_id': torch.tensor(self.bug_type_to_id[bug_type], dtype=torch.long),
        }


class OptimizedTrainer:
    """
    Optimized trainer with fp16, gradient accumulation, early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Training state
        self.global_step = 0
        self.best_eval_metric = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []
        self.eval_exact_matches = []
        self.eval_bleus = []
        self.learning_rates = []
        
        # Setup mixed precision if available
        self.use_fp16 = config.use_fp16 and HAS_AMP and torch.cuda.is_available()
        if self.use_fp16:
            self.scaler = GradScaler()
            logger.info("Using mixed precision (fp16)")
        else:
            self.scaler = None
            logger.info("Using full precision (fp32)")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized with device: {device}")
    
    def setup_optimizer_and_scheduler(
        self,
        train_dataset: Dataset,
    ):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer setup with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Calculate total training steps
        num_batches_per_epoch = len(train_dataset) // self.config.batch_size
        total_training_steps = num_batches_per_epoch * self.config.num_epochs
        
        # Setup scheduler
        if self.config.lr_scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_training_steps,
            )
        elif self.config.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_training_steps - self.config.warmup_steps,
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer setup: LR={self.config.learning_rate}, warmup={self.config.warmup_steps}")
        logger.info(f"Total training steps: {total_training_steps}")
        
        return self.optimizer, self.scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Train for one epoch with gradient accumulation and mixed precision.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            try:
                # Forward pass with optional mixed precision
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            bug_type_labels=batch.get('bug_type_id'),
                            task='both',
                        )
                        loss = outputs['loss']
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        bug_type_labels=batch.get('bug_type_id'),
                        task='both',
                    )
                    loss = outputs['loss']
                
                # Gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights every N steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    self.train_losses.append(avg_loss)
                    self.learning_rates.append(current_lr)
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}"
                    )
                    
            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                if self.use_fp16:
                    self.scaler.update()
                self.optimizer.zero_grad()
                continue
        
        epoch_loss = total_loss / num_batches
        return epoch_loss
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        Returns validation loss and metrics.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    bug_type_labels=batch.get('bug_type_id'),
                    task='both',
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.eval_losses.append(avg_loss)
        
        return {'loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float] = None):
        """Save model checkpoint"""
        checkpoint_dir = self.config.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'metrics': metrics or {},
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if metrics and 'loss' in metrics:
            if metrics['loss'] < self.best_eval_metric:
                self.best_eval_metric = metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                
                best_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved with loss: {metrics['loss']:.4f}")
                
                self.patience_counter = 0
            else:
                self.patience_counter += 1
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early"""
        if not self.config.early_stopping:
            return False
        
        return self.patience_counter >= self.config.patience
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        """
        Main training loop.
        """
        logger.info("Starting training...")
        
        # Setup data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_dataset)
        
        # Save config
        self.config.save(self.config.output_dir)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)
            
            # Log epoch results
            logger.info(
                f"\nEpoch {epoch+1}/{self.config.num_epochs}:"
                f"\n  Train Loss: {train_loss:.4f}"
                f"\n  Val Loss: {val_metrics['loss']:.4f}"
                f"\n  Best Val Loss: {self.best_eval_metric:.4f}"
            )
            
            # Early stopping check
            if self.should_stop_early():
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training completed!")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Best model loaded for inference")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'total_steps': self.global_step,
            'best_eval_loss': self.best_eval_metric,
            'final_patience_counter': self.patience_counter,
            'num_train_losses': len(self.train_losses),
            'num_eval_losses': len(self.eval_losses),
            'use_fp16': self.use_fp16,
        }
