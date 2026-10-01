# 🚛 ScaniaGuard — Cost-Sensitive APS Failure Detection

> A classical machine-learning system for detecting APS-related failures in Scania trucks under severe class imbalance and asymmetric misclassification costs.

---
## Live Demo -> https://scaniatruck-aps-detection-app.streamlit.app/
## 🚛 Project Overview

**ScaniaGuard** predicts whether a Scania truck belongs to the **APS-failure class** using anonymized operational data.

The key challenge is that:

- The dataset is highly imbalanced.
- Missing values are extensive.
- A **False Negative (missed failure)** is much more costly than a False Positive.
- Therefore, accuracy alone is not an appropriate optimization target.

### Cost Matrix

| Prediction Error | Cost |
|---|---:|
| False Positive | 10 |
| False Negative | 500 |

The project therefore focuses on **cost-sensitive classification and threshold optimization** rather than simply maximizing accuracy.

---

## 🎯 Objective

Given the operational measurements of a truck:

> **Predict APS-related failure while minimizing the total cost of misclassification.**

The final decision threshold was selected using the validation set based on:

```text
Total Cost = (10 × FP) + (500 × FN)
