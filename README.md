
#  Fake News Detection using RoBERTa

This project implements a robust binary text classification pipeline using `roberta-base` to detect fake news articles. Built with HuggingFace Transformers, PyTorch, and scikit-learn, the model is trained and evaluated using rich metrics including Accuracy, Precision, Recall, F1 Score, AUC, and Equal Error Rate (EER).

---

##  Features

-  Preprocessing and tokenization using HuggingFace’s `RobertaTokenizer`
-  Fine-tuning `roberta-base` on labeled fake news dataset
-  Handles class imbalance using class weights and sampling
-  Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUC, and EER
-  Visualizations: ROC Curve, Confusion Matrix, Loss Curves
-  Model checkpointing and gradient clipping

---

##  Dataset

- **Source**: [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
- **Structure**: CSV with columns:
  - `text`: input news content
  - `label`: 0 (real), 1 (fake)
- **Split**:
  - Training: 31,428 samples
  - Validation: 6,735 samples
  - Test: 6,735 samples

---

##  Usage

### 1️⃣ Preprocessing

* Convert raw text to lowercase, remove special characters and whitespace.
* Tokenize using `RobertaTokenizer` (subword-level BPE).
* Convert to input IDs, attention masks, and apply padding.

### 2️⃣ Training

```bash
python roberta_fakenews.py --epochs 3 --batch_size 16 --lr 2e-5
```

### 3️⃣ Evaluation

* Classification report: Accuracy, Precision, Recall, F1 Score
* Confusion Matrix
* ROC Curve + AUC
* Equal Error Rate (EER) and optimal threshold

---

## 📊 Sample Results

| Metric    | Score                     |
| --------- | ------------------------- |
| Accuracy  | 99.97%                    |
| Precision | 99.97%                    |
| Recall    | 99.97%                    |
| F1 Score  | 99.97%                    |
| AUC       | 1.00(0.99+)               |
| EER       | 0.0003 @ Threshold 0.9833 |

- The RoBERTa model outperformed other baseline models including FakeBERT, a BERT variant designed specifically for fake news detection, demonstrating superior generalization and robustness across all evaluation metrics.
---

##  Visualizations

*  Training & Validation Loss Curve
*  Confusion Matrix
*  ROC Curve with AUC
*  EER threshold plot

---

##  Model Architecture

* Backbone: `roberta-base` (12-layer transformer, 768 hidden units, 12 heads)
* Classification Head:

  * Dropout Layer
  * Linear Layer
  * Sigmoid/Softmax Activation

---


##  Notes

1. You may switch to `roberta-large` for higher accuracy (requires more memory).
2. Training is GPU-accelerated; use CUDA if available.
3. Gradient clipping and learning rate scheduling improve convergence.
4. Use `compute_class_weight` from `sklearn` to address class imbalance.

---

##  Citation

If you use this repository, consider citing the following dataset:

```bibtex
@dataset{isot_fake_news_2022,
  title={ISOT Fake News Dataset},
  url={https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/},
  year={2022},
  publisher={University of Victoria}
}
```

---


