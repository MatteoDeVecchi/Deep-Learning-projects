#  Hotel Review Score Regressor – BiLSTM + Tabular Fusion

This project was developed as part of a university exam in Deep Learning, under time and computational constraints. The goal was to design a hybrid model capable of predicting hotel review scores (normalized between 0 and 10) by combining **textual review data** and **structured/tabular data**.

Importantly, this was an educational task where **model performance was not the main evaluation criterion**: the goal was to demonstrate a correct pipeline design, integration of different data types, and the use of deep learning best practices.

---

##  Project Overview

The solution consists of a **dual-branch neural network**:
- A **BiLSTM branch** for processing review text sequences
- A **dense MLP branch** for numerical and categorical structured features (e.g. hotel name, reviewer nationality, number of reviews, etc.)

The two branches are concatenated and followed by a final dense output layer.

---

##  Architecture Summary

-  **Text preprocessing:** tokenization, padding, embedding
-  **Structured data preprocessing:** feature scaling, one-hot encoding, label encoding
-  **Model components:**
  - Embedding layer + BiLSTM for text
  - BatchNorm + Dense layers for structured features
  - Fusion via concatenation + dropout + dense output

---

##  Technologies Used

- Python, Pandas, NumPy
- TensorFlow / Keras
- Scikit-learn for preprocessing and metrics
- Google Colab

---

## Evaluation

- Loss function: Mean Squared Error (MSE)
- Achieved test MSE: **1.81**
- ⚠️ Score not used for grading — the focus was on architecture, explainability, and robustness under realistic constraints

---

##  Project Structure

1. **Data preprocessing**
   - Clean text
   - Encode structured columns
   - Split into train/val/test (no leakage)
2. **Model building**
   - Build BiLSTM and MLP branches
   - Concatenate and add output layer
3. **Training & tuning**
   - KFold cross-validation
   - Batch size, dropout, early stopping
4. **Evaluation**
   - Compare MSE on val/test
   - Plot training history

---

##  Notes

This project was part of my second-year coursework in Artificial Intelligence (Deep Learning module), and reflects my ability to:
- Design mixed-input DL architectures
- Work under limited time/resources (no GPU tuning)
- Apply good practices in modeling, testing and documenting
- mark: 29/30

Feel free to explore the notebook and get in touch if you want to discuss it!
