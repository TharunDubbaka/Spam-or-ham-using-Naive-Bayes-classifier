# Spam or Ham using Naive Bayes Classifier

## 📝 Project Overview
This project uses a Naive Bayes Classifier to distinguish between **ham** (legitimate) and **spam** messages. The goal was to tune the feature extractor (`max_features`) for optimal performance while checking for overfitting and maintaining generalization.

---

## ⚡️ Feature Selection
- Tested feature counts ranging from **3000** to **7000**.
- Final choice: **3000** — achieving an ideal balance between accuracy and model complexity.
- Increasing to **4000** yielded no performance benefits and risked overfitting.

---

## 🥇 Results

### Training Set

| Class | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Ham (0) | 0.98 | 1.00 | 0.99 | 3859 |
| Spam (1) | 1.00 | 0.87 | 0.93 | 598 |
| **Accuracy** | | | **0.98** | 4457 |

---

### Testing Set

| Class | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Ham (0) | 0.98 | 1.00 | 0.99 | 966 |
| Spam (1) | 1.00 | 0.85 | 0.92 | 149 |
| **Accuracy** | | | **0.98** | 1115 |

---

## ⚠️ Observations
- An unexpected label (`2`) appeared due to a `{"mode":"full"}` entry, which was fixed during preprocessing.
- The model achieved ~**98% accuracy** across both training and test sets.
- It worked well on real-world sentences.
- After inspecting the confusion matrix, I noticed some spam messages were missed. I fine-tuned the threshold and improved recall.

---

## ✅ Conclusion
The classifier is highly effective at detecting spam, with strong precision and recall, and minimal overfitting at `max_features = 3000`.

---

## 🚀 Next Steps
- Deploy the model (e.g., Streamlit app or Flask API).
- Explore more advanced text processing or deep learning models (e.g., LSTM, Transformer).

