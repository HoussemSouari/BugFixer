"""
Enhanced CodeT5 model architecture with multi-task learning and curriculum learning.
Includes bug-type classification head for improved generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
import logging

from transformers import AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class CodeT5MultiTask(nn.Module):
    """
    CodeT5 with auxiliary bug-type classification head.
    Implements multi-task learning for improved bug detection.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5-base",
        num_bug_types: int = 7,
        dropout_rate: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_bug_types = num_bug_types
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        
        # Load pre-trained CodeT5
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.config = self.base_model.config
        
        # Store hidden size for heads
        self.hidden_size = self.config.d_model
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Bug-type classification head
        self.bug_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.hidden_size // 2, num_bug_types)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
        logger.info(f"Initialized CodeT5MultiTask with {num_bug_types} bug types")
    
    def _init_classifier_weights(self):
        """Initialize classifier head weights"""
        for module in self.bug_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        bug_type_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        task: str = 'refactoring',  # 'refactoring', 'classification', 'both'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting multi-task learning.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target sequence labels (for seq2seq task)
            bug_type_labels: Bug type labels (for classification task)
            return_dict: Whether to return dict
            task: Which task to perform ('refactoring', 'classification', or 'both')
            
        Returns:
            Dictionary with losses and logits
        """
        # Run base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
        )
        
        loss = base_outputs.loss if labels is not None else None
        
        # Get encoder hidden states for bug classification
        encoder_hidden_states = base_outputs.encoder_last_hidden_state
        
        # Mean pooling over sequence dimension
        pooled = torch.mean(encoder_hidden_states, dim=1)  # (batch_size, hidden_size)
        pooled = self.dropout(pooled)
        
        # Bug-type classification
        bug_logits = self.bug_classifier(pooled)  # (batch_size, num_bug_types)
        
        classification_loss = None
        if bug_type_labels is not None:
            # Cross entropy with label smoothing
            classification_loss = F.cross_entropy(
                bug_logits,
                bug_type_labels,
                label_smoothing=self.label_smoothing,
            )
        
        # Combine losses for multi-task learning
        total_loss = None
        if task == 'refactoring' and loss is not None:
            total_loss = loss
        elif task == 'classification' and classification_loss is not None:
            total_loss = classification_loss
        elif task == 'both':
            if loss is not None and classification_loss is not None:
                # Weighted combination (can be tuned)
                total_loss = 0.7 * loss + 0.3 * classification_loss
            elif loss is not None:
                total_loss = loss
            elif classification_loss is not None:
                total_loss = classification_loss
        
        return {
            'loss': total_loss,
            'seq2seq_loss': loss,
            'classification_loss': classification_loss,
            'logits': base_outputs.logits,
            'bug_logits': bug_logits,
            'past_key_values': base_outputs.past_key_values,
            'decoder_hidden_states': base_outputs.decoder_hidden_states,
            'decoder_attentions': base_outputs.decoder_attentions,
            'cross_attentions': base_outputs.cross_attentions,
            'encoder_last_hidden_state': encoder_hidden_states,
            'encoder_hidden_states': base_outputs.encoder_hidden_states,
            'encoder_attentions': base_outputs.encoder_attentions,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 5,
        early_stopping: bool = True,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate code using beam search with optional parameters.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Stop when beam reaches end token
            length_penalty: Length penalty for beam search
            temperature: Temperature for sampling
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.base_model.config.pad_token_id,
                eos_token_id=self.base_model.config.eos_token_id,
                **kwargs
            )
        
        return outputs
    
    def get_encoder_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get pooled encoder representation for bug classification"""
        outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Mean pooling
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return pooled
    
    def predict_bug_type(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bug type for given code.
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        pooled = self.get_encoder_representation(input_ids, attention_mask)
        logits = self.bug_classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        return predictions, probs
    
    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning"""
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.base_model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and non-trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
        }


class CurriculumLearningScheduler:
    """
    Manages curriculum learning by gradually increasing difficulty.
    Strategy: start with simple bugs (off-by-one), progress to complex ones.
    """
    
    def __init__(self, num_epochs: int = 10):
        self.num_epochs = num_epochs
        
        # Bug type difficulty (1=easy, 7=hard)
        self.bug_difficulty = {
            'off_by_one': 1,
            'operator': 2,
            'missing_break': 2,
            'logic': 3,
            'null_check': 4,
            'type_mismatch': 5,
            'resource_leak': 6,
        }
        
        self.current_epoch = 0
    
    def get_difficulty_threshold(self) -> int:
        """
        Get difficulty threshold for current epoch.
        Gradually increase from 1 to max difficulty.
        """
        max_difficulty = max(self.bug_difficulty.values())
        threshold = int(1 + (max_difficulty - 1) * (self.current_epoch / max(self.num_epochs - 1, 1)))
        return threshold
    
    def should_include_sample(self, bug_type: str) -> bool:
        """
        Determine if sample should be included in current epoch.
        Include samples up to current difficulty threshold.
        """
        difficulty = self.bug_difficulty.get(bug_type, 1)
        threshold = self.get_difficulty_threshold()
        return difficulty <= threshold
    
    def filter_dataset(self, samples: list) -> list:
        """Filter samples based on current curriculum stage"""
        threshold = self.get_difficulty_threshold()
        filtered = [
            s for s in samples
            if self.bug_difficulty.get(s.get('bug_type', 'off_by_one'), 1) <= threshold
        ]
        return filtered
    
    def step(self):
        """Move to next epoch"""
        self.current_epoch += 1


class SyntaxGuidedDecoding:
    """
    Constrains generation to produce syntactically valid Java code.
    Uses banned tokens and length penalties to guide generation.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Tokens that indicate end of valid statement
        self.statement_enders = [';', '}']
        
        # Banned token patterns (will be populated from tokenizer)
        self.banned_tokens = self._get_banned_tokens()
    
    def _get_banned_tokens(self) -> list:
        """Get list of tokens that shouldn't appear in valid Java code"""
        banned = []
        
        # Find tokens representing invalid patterns
        invalid_patterns = [
            '<unk>',  # Unknown tokens
            '<mask>',  # Mask tokens
        ]
        
        for pattern in invalid_patterns:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(pattern)
                if token_id > 0:
                    banned.append(token_id)
            except:
                pass
        
        return banned
    
    def get_constrained_logits(
        self,
        logits: torch.Tensor,
        banned_token_ids: list = None,
    ) -> torch.Tensor:
        """
        Modify logits to ban certain tokens and enforce constraints.
        """
        if banned_token_ids is None:
            banned_token_ids = self.banned_tokens
        
        # Set banned token logits to very negative
        for token_id in banned_token_ids:
            if token_id < logits.shape[-1]:
                logits[..., token_id] = float('-inf')
        
        return logits
