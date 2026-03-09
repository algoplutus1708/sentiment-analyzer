# Sentiment Analyzer Model Improvement Plan

## Goal: Increase accuracy from 75% to 90%

## Tasks:
- [x] 1. Analyze current model and training configuration
- [x] 2. Rewrite LLM.ipynb with improved model architecture
- [x] 3. Increase model capacity (d_model, num_layers, num_heads)
- [x] 4. Use full IMDB dataset (25K train, 25K test)
- [x] 5. Increase max_length to 256
- [x] 6. Add advanced training techniques (warmup, label smoothing, etc.)
- [ ] 7. Train model with improved configuration
- [ ] 8. Evaluate and verify 90%+ accuracy
- [ ] 9. Save new model checkpoint
- [x] 10. Update app/model.py if needed

## Model Configuration Changes:
- vocab_size: 12000 → 15000
- d_model: 128 → 384
- num_layers: 2 → 6
- num_heads: 4 → 8
- d_ff: 384 → 1536
- max_len: 128 → 256
- dropout: 0.2 → 0.15
- epochs: 10 → 15
- batch_size: 32 → 48
- learning_rate: 4e-4 → 2e-4

## New Features:
- Pre-LN transformer architecture for better stability
- Multi-pooling (CLS + mean + max + attention-weighted)
- Learning rate warmup + cosine annealing
- Label smoothing (0.1)
- Gradient clipping
- Full IMDB dataset (25K train samples)
- Total parameters: ~14M (vs ~600K original)

## Next Steps:
1. Open LLM.ipynb in Jupyter/Colab
2. Run all cells to train the model
3. Verify 90%+ accuracy on test set
4. Model will be saved as tinyllm_complete.pt

