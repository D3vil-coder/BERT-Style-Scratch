# README: BERT From Scratch 
## Overview

This repository addresses the problem: **Implement a simplified BERT, trained from scratch using PyTorch**, without using any pre-trained BERT code. The solution covers:  
- A compact BERT-like Transformer model architecture
- Pretraining on WikiText-2-v1 (raw, full size: ~72,000 sentences; ~2M tokens)
- Joint Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) training from scratch
- Demonstrating qualitative and quantitative learning with observable statistics and predictions


## Directory Structure & File Role

- `preprocess_final.ipynb` — Cleanly prepares data for BERT pretraining: NSP balancing, tokenization, segment/attention mask, and masking for MLM.    
- `main.ipynb` — Complete model definition, objective wiring, training loop with progress reporting, accuracy statistics, and output demo.  
- `README.md` — This explanation

To view the preprocessed Json Data and Model file- 
```
https://drive.google.com/drive/folders/1QGRP1lELUIsbk6L2k2WDvYmKDqNOpf-Q?usp=sharing
```

- `preprocessed_wikitext2_full.json` — Final data: 100,000 balanced [Sentence A, Sentence B] pairs with all BERT inputs/targets, ~200MB.

- `bert_wikitext2_full.pth` — Final trained weights, for audit/reuse  

---

## Data Preparation Details 

**Dataset:**  
- `wikitext-2-raw-v1` (Hugging Face): ~72k sentences, ~2 million tokens, processed from Wiki articles

**Steps/Quantities:**  
1. Concatenate all non-empty texts, split into sentences with NLTK (`sent_tokenize`), filter out sentences <10 chars.
2. **Create 100,000 balanced pairs:**  
    - **50% positive:** Sentence B is the true next sentence after Sentence A  
    - **50% negative:** Sentence B is a random, clearly non-consecutive sentence  
    - Ensures no NSP degenerate solutions, improves representation learning
3. **Tokenization:**  
    - For each pair, `[CLS] sentence_A [SEP] sentence_B [SEP]`, max length 128  
    - BERT-uncased vocab (~30,522 tokens)
    - Auto-padding/truncation for all fields
4. **MLM Masking:**  
    - Randomly select **15%** of non-special (not [CLS]/[SEP]/[PAD]) positions per input  
    - Of those, **80%** replaced with `[MASK]`, **10%** with random token, **10%** left unchanged
    - Label is original token at masked location, else -100 (for PyTorch loss ignore)
5. **Meta:**  
    - Save token_ids, type_ids, attention_mask, mlm_labels, nsp_label
    - Output: ~100,000 pairs (JSON), NSP ratio checked to be ~0.50



---

## Model, Training Loop & Parameters 

**Model structure:**  
- **Embedding:**  
    - Token embedding: vocab_size x 256  
    - Positional embedding: 128 x 256  
    - Segment embedding: 2 x 256  
    - All summed, then LayerNorm and dropout (0.1)
- **Encoder:**  
    - 4 layers, 256-dim, 8-head TransformerEncoder (PyTorch, batch_first=True)
    - Attention mask to ignore padding
    - Total params: ~5-6 million 
- **MLM prediction head:**  
    - Linear(256→256) + GELU + LayerNorm + Linear(256→vocab_size)
    - Predicts masked tokens across all positions
- **NSP head:**  
    - Same MLP on first position ([CLS]); binary output (is_next/ not_next)

**Training setup:**  
- **Batch size:** 16 
- **Epochs:** 10 (results below, longer possible for >30% MLM)
- **Batches per epoch:** 6250 (100,000//16)
- **Loss:** `total = MLM loss (ignore_index=-100) + NSP loss` (equal weight)  
- **Optimizer:** AdamW, lr=1e-5, weight_decay=0.01  
- **Scheduler:** OneCycleLR (max_lr=1e-4, warmup/decay over total steps=epochs*batches)
- **Regularization:** gradient clip (max norm=1.0)
- **Reporting:** tqdm progress bar by batch, end-of-epoch average metrics

## 📈 Performance Summary
The model shows steady convergence over 10 epochs, effectively learning syntax and sentence relationships.

| Metric | Epoch 1 | Epoch 5 | Epoch 10 |
| :--- | :--- | :--- | :--- |
| **MLM Accuracy** | 6.7% | 25.0% | **29.5%** |
| **NSP Accuracy** | 50.1% | 63.1% | **75.8%** |
| **Total Loss** | 7.83 | 6.16 | **5.27** |


---

## Training Results & Demonstration

### Key metrics (out of 10 epochs):

- **MLM Accuracy (masked tokens only):** Climbed from **6.7%** to **29.5%**
- **NSP Accuracy:** Climbed from **50%** (random) to **75.8%**
- **Losses:** MLM loss dropped from ~7.8 to ~4.8, NSP loss from ~0.7 to ~0.48 across 10 epochs

**Sample output, run summary:**
```
Epoch 10/10
Epoch 10: 100%|██████████| 6250/6250 [08:42<00:00, 11.96it/s, Total Loss=5.5070, MLM Acc=0.295, NSP Acc=0.758]
Epoch 10 Summary: MLM Loss=4.7934, NSP Loss=0.4860
MLM Accuracy (masked only)=0.295, NSP Accuracy=0.758
```


**Qualitative demo — masked prediction examples:**  
```
Sample 1 (NSP Label: 0, Predicted: 1):
Input (with masks): [CLS] he made his debut in [MASK] 1 – 0 win at cardiff [MASK] on 11 august 2007, before scoring in h...
Masked positions originals: ['a', '0', 'city', 'second', 'in']...
Model predictions for masks: ['a', '0', 'season', 'first', 'in']...
```

Model output reasonably matches context. NSP predicts sentence pair continuity well above chance (0.75+).

---

## Takeaways & Observations

- **The major assignment objectives are met:**
    - From-scratch BERT architecture and heads
    - Preprocessing that matches assignment and BERT design
    - Demonstration of learning for both MLM and NSP with clear, interpretable metrics and examples
- **Validation:** Confirmed that training improvements reflect learned representations (mask filling, NSP task).
- **Bottlenecks:** For larger accuracies, more epochs, layers, or a larger dataset would be needed (as pretraining BERT is inherently data- and compute-intensive).
- **Strengths:** Clear modular pipeline, full code reproducibility, extensive inline comments.
- **Weaknesses/Improvements:** No validation set (single split as per spec), compact model so still underfits ("full BERT" would need 10-20x more capacity/data).

---

## How to Run

### Dependencies and Setup

**Required (tested: Python 3.12 + CUDA 11.8):**  
- torch, torchvision, torchaudio (for PyTorch and GPU)
- transformers (for tokenizer use only)
- datasets (Hugging Face, for WikiText-2)
- nltk (for sentence splitting, downloads)
- tqdm (progress bars)

**Install:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets nltk tqdm
```

Ensure you are running on a CUDA-capable GPU.


1. **Preprocess:**
    ```
    python preprocess_final.ipynb
    ```
    Outputs: `preprocessed_wikitext2_full.json`
2. **Train:**  
    ```
    python main.ipynb
    ```
    Progress shown by epoch/batch. Results printed at epoch end and demo at finish.


## References

- Devlin, J. et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv:1810.04805 (2018)
- WikiText-2 Dataset: https://huggingface.co/datasets/mikasenghaas/wikitext-2
- Hugging Face Transformers (tokenizer), PyTorch documentation

---
