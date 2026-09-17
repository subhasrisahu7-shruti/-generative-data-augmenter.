# Generative Tabular Data Augmenter for Enterprise Analytics

An end-to-end data science pipeline built to solve data scarcity, privacy regulations, and class imbalance using Conditional Generative Adversarial Networks (CTGAN). This framework analyzes the structural metadata and mathematical correlations of a baseline dataset to generate high-fidelity, privacy-compliant synthetic records.

## 📊 Project Overview
* **The Problem:** Real-world enterprise datasets are often constrained by privacy laws (GDPR/CCPA) or severe class imbalances, leading to heavily biased machine learning models.
* **The Solution:** A generative data architecture utilizing the `SDV` (Synthetic Data Vault) ecosystem and `CTGAN` to upscale tabular data density while fully preserving statistical integrity.
* **The Outcome:** Successfully generated 500 synthetic client records matching the original distribution curves with high mathematical fidelity without replicating or exposing actual consumer identities.

---

## 🚀 Key Features
* **Automated Metadata Handling:** Infers data schemas, structural constraints, and variable types programmatically.
* **Deep Generative Modeling:** Trains a Conditional GAN to learn complex joint distributions between continuous metrics (Age, Income) and categorical fields (Membership Tiers).
* **Fail-Safe Mathematical Validation:** Implements direct custom statistical checks using NumPy, Pandas, and Seaborn to validate data integrity.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.10+
* **Core Machine Learning:** `sdv`, `ctgan`, `pandas`, `numpy`
* **Visualization:** `seaborn`, `matplotlib`
* **Environment:** Google Colab

---

## 📈 Evaluation Matrix & Fidelity Performance

The validation pipeline calculates explicit descriptive similarity shapes to check data drift and profile fidelity:

| Metric Evaluation Type | Alignment Score | Description |
| :--- | :---: | :--- |
| **Data Structure Integrity** | ✅ 100% Passed | Synthetic columns exactly match original shapes and type rules. |
| **Column Shape Quality (Age)** | **High Fidelity** | Distribution curves match original population curves smoothly. |
| **Column Shape Quality (Income)**| **High Fidelity** | Successfully preserved statistical income bracket boundaries. |

---

## 💻 How to Run this Project

1. Open Google Colab and upload the notebook file.
2. Install the core dependencies by executing:
   ```bash
   pip install sdv ctgan pandas numpy matplotlib seaborn
   ```
3. Run the notebook blocks sequentially to train the CTGAN model, generate synthetic rows, and compute verification charts.
