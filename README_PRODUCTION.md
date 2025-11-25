# Java Bug-Fixing Model - Production Implementation Guide

## Overview

This repository contains a complete, production-grade Java bug-fixing model based on CodeT5 with:

- **Real-world focus**: Fixes actual Java bugs (not abstracted variable names)
- **Advanced architecture**: Multi-task learning with bug-type classification
- **Optimized training**: Mixed precision (fp16), gradient accumulation, curriculum learning
- **Comprehensive evaluation**: BLEU, CodeBLEU, exact match, syntax validity
- **Production-ready inference**: Beam search, confidence scoring, fallback mechanisms

## Critical Issues Addressed

### Problem 1: Abstracted Variable Names
**Old Issue**: Model trained on VAR_1, TYPE_1, outputs gibberish on real code  
**Solution**: Removed abstraction layer entirely, keep real variable/method names

### Problem 2: Domain Mismatch
**Old Issue**: 77.8% BLEU on CodeXGLUE but 0% exact match on real code  
**Solution**: Generate synthetic bugs from real Java code with proper training

### Problem 3: Training Configuration
**Old Issue**: Batch size 2, only 3 epochs, no mixed precision  
**Solution**: Batch size 16, 15 epochs, fp16 mixed precision training

## Installation

```bash
cd /home/houssem/BugFixer
pip install -r requirements.txt
python verify_setup.py
```

## Quick Start

### Training
```bash
jupyter notebook training_pipeline.ipynb
```

### Inference
```python
from src.model.production_inference import ProductionInference
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('./output/final_model')
tokenizer = AutoTokenizer.from_pretrained('./output/final_model')
inference = ProductionInference(model, tokenizer, device)

result = inference.predict_single("for(int i=0; i<=arr.length; i++) { }")
print(result.predicted_code)
```

## Project Structure

```
src/
  ├── data_processing/         Data generation & preprocessing
  ├── model/                   Architecture & training
  ├── evaluation/              Metrics computation
  └── utils/                   Config & logging

training_pipeline.ipynb        End-to-end training
test_cases.py                  20 test cases
requirements.txt               Dependencies
```

## Key Features

✅ Real data - Actual Java code (not abstracted)  
✅ Advanced training - Mixed precision, gradient accumulation, early stopping  
✅ Multi-task learning - Bug classification + code generation  
✅ Comprehensive evaluation - 7 metrics  
✅ Production-ready - Beam search, confidence scoring  
✅ Well-documented - 3500+ lines

## Data Preparation

### 7 Bug Types Supported
1. off_by_one - Loop bounds errors
2. null_check - Missing null checks
3. missing_break - Missing switch breaks
4. operator - Wrong operators
5. logic - Logic errors
6. resource_leak - Missing .close()
7. type_mismatch - Wrong type casts

```python
from src.data_processing.real_bug_loader import RealBugGenerator

generator = RealBugGenerator(seed=42)
buggy, original, bug_type = generator.inject_bug(code, 'off_by_one')
```

## Preprocessing

```python
from src.data_processing.advanced_preprocessor import AdvancedJavaPreprocessor

preprocessor = AdvancedJavaPreprocessor(max_length=512)
pair = preprocessor.preprocess_pair(
    buggy_code, fixed_code, bug_type,
    validate_syntax=True,
    add_prefix=True,
)
```

## Training Configuration

```python
from src.model.optimized_trainer import TrainingConfig

config = TrainingConfig(
    batch_size=16,
    gradient_accumulation_steps=4,        # Effective batch: 64
    learning_rate=3e-5,
    num_epochs=15,
    use_fp16=True,                        # Mixed precision
    early_stopping=True,
    patience=3,
)
```

## Expected Performance

| Metric | Target |
|--------|--------|
| Exact Match | >20% |
| BLEU-4 | >70% |
| CodeBLEU | >60% |
| Syntax Valid | >95% |

## Training Time

- Device: Tesla T4 (16GB VRAM)
- Effective batch: 64
- Time per epoch: ~25 minutes
- Total training: ~6 hours

## Evaluation

```python
from src.evaluation.comprehensive_evaluator import BugFixEvaluator

evaluator = BugFixEvaluator(tokenizer)
result = evaluator.evaluate_single(
    buggy='for(int i=0; i<=n; i++) { }',
    fixed='for(int i=0; i<n; i++) { }',
    predicted='for(int i=0; i<n; i++) { }',
    bug_type='off_by_one',
)
```

## Troubleshooting

### CUDA Out of Memory
```python
config.batch_size = 8
config.gradient_accumulation_steps = 8
```

### Low Exact Match
- Increase training data: `num_synthetic_per_sample=10`
- Enable early stopping: `early_stopping=True`
- Reduce learning rate: `learning_rate=1e-5`

### Model Outputs Gibberish
- Verify preprocessing is working
- Check training loss is decreasing
- Reduce beam size: `num_beams=3`

### Inference Too Slow
```python
result = inference.predict_single(
    buggy_code,
    num_beams=3,
)
```

## Testing

```bash
# Run test cases
python test_cases.py

# Evaluate on tests
from test_cases import RealBugTestCases
test_cases = RealBugTestCases.get_all_cases()
```

## References

- [CodeT5: Pre-trained Encoder-Decoder Models for Code](https://arxiv.org/abs/2109.13666)
- [Defects4J: Database of Bugs](https://arxiv.org/abs/1601.02540)
- [CodeBLEU: Evaluation of Code Synthesis](https://arxiv.org/abs/2009.10297)

## License

MIT License
