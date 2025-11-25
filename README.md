# BugFixer - Java Bug-Fixing Model

A production-grade Java bug-fixing model based on CodeT5 that fixes real bugs in actual Java code.

## Quick Links

- **[README_PRODUCTION.md](README_PRODUCTION.md)** - Complete production guide (2000+ lines)
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was implemented (1500+ lines)
- **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** - Quick delivery overview

## What This Does

This system fixes common Java bugs including:
- Off-by-one errors in loops
- Missing null checks
- Missing break statements
- Operator errors (= vs ==, & vs &&)
- Logic errors and inversions
- Resource leaks
- Type mismatches

## Quick Start

### 1. Install Dependencies
```bash
cd /home/houssem/BugFixer
pip install -r requirements.txt
```

### 2. Run Training
```bash
jupyter notebook training_pipeline.ipynb
```
Training takes 6-8 hours on Tesla T4.

### 3. Use the Model
```python
from src.model.production_inference import ProductionInference
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('./output/final_model')
tokenizer = AutoTokenizer.from_pretrained('./output/final_model')
inference = ProductionInference(model, tokenizer, device)

result = inference.predict_single("for(int i=0; i<=arr.length; i++) { }")
print(result.predicted_code)  # for(int i=0; i<arr.length; i++) { }
```

## Project Structure

```
src/
  ├── data_processing/         Data preparation and preprocessing
  ├── model/                   Model architecture and training
  ├── evaluation/              Comprehensive evaluation metrics
  └── utils/                   Utilities and configuration

training_pipeline.ipynb        Complete end-to-end training
test_cases.py                  20 real bug test cases
requirements.txt               All dependencies
```

## Key Features

✅ **Real data** - Fixes actual Java code (not abstracted variables)  
✅ **Advanced training** - Mixed precision (fp16), gradient accumulation, early stopping  
✅ **Multi-task learning** - Bug-type classification alongside code generation  
✅ **Comprehensive evaluation** - BLEU, CodeBLEU, exact match, syntax validation  
✅ **Production-ready** - Beam search, confidence scoring, fallback mechanisms  
✅ **Well-documented** - 3500+ lines of guides and code examples  

## Documentation

For complete information, see:
- **[README_PRODUCTION.md](README_PRODUCTION.md)** - Full production guide with troubleshooting
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** - Quick delivery overview

## Performance

Target metrics after training:
- **Exact Match**: >20%
- **BLEU-4**: >70%
- **CodeBLEU**: >60%
- **Syntax Valid**: >95%

## License

MIT License