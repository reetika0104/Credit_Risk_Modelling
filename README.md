<p align="center">
  <img src=".pipeline_sum.jpg" alt="Credit Risk Modelling Pipeline" width="700"/>
</p>

<h1 align="center">💰 Credit Risk Modelling</h1>

<p align="center">
  <strong>End-to-end calculation of PD, LGD, EAD & Expected Loss using Machine Learning in Python</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.0.2-orange?logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Basel_II-Compliant-critical" alt="Basel II"/>
</p>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Pipeline](#-pipeline)
- [Notebooks](#-notebooks)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Deliverables](#-deliverables)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Technologies](#-technologies)
- [License](#-license)
- [Author](#-author)

---

## 🧠 About the Project

Credit risk modelling is critical for financial institutions — it quantifies the risk that a borrower will fail to repay a loan, credit card, or other debt obligation. This project provides a **complete, Basel II-compliant pipeline** to model credit risk using machine learning.

The goal is to build production-ready models that output:
- A **credit scorecard** for daily use
- An **expected loss calculation** pipeline

> This project includes revisions and additions built on top of the credit risk course on [365 Data Science](https://365datascience.com/courses/credit-risk-modeling-in-python/).

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Preprocessing** | Fine & coarse classing, dummy variable encoding |
| 📊 **PD Model** | Logistic regression-based Probability of Default |
| 🃏 **Scorecard** | Practical credit scorecard exported as CSV |
| 📉 **LGD Model** | Two-stage beta regression for Loss Given Default |
| 💵 **EAD Model** | Linear regression for Exposure at Default |
| 🧮 **Expected Loss** | Full EL calculation combining PD × LGD × EAD |
| 🔍 **PSI Check** | Population Stability Index for model monitoring |

---

## 🔁 Pipeline

<p align="center">
  <img src=".pipeline_sum.jpg" alt="Pipeline Summary" width="650"/>
</p>

### End-to-End Flow

```mermaid
flowchart LR
    A[📥 Raw Data] --> B[🔄 Preprocessing]
    B --> C[🧮 Feature Engineering]
    C --> D[📊 PD Model]
    C --> E[📉 LGD Model]
    C --> F[💵 EAD Model]
    D --> G[🃏 Scorecard]
    D --> H[🧮 Expected Loss]
    E --> H
    F --> H
    H --> I[📋 Risk Decision]
    G --> I
```

### Model Training Strategy

```mermaid
flowchart TD
    subgraph PD ["Probability of Default"]
        PD1[Logistic Regression] --> PD2[Credit Scorecard]
        PD1 --> PD3[Cutoff Rate Analysis]
    end

    subgraph LGD ["Loss Given Default"]
        LGD1[Stage 1: Logistic Regression] --> LGD2[Stage 2: Linear Regression]
    end

    subgraph EAD ["Exposure at Default"]
        EAD1[Linear Regression]
    end

    PD --> EL[🧮 Expected Loss = PD × LGD × EAD]
    LGD --> EL
    EAD --> EL
```

### Data Split & Validation

```mermaid
gantt
    title Dataset Timeline
    dateFormat YYYY
    axisFormat %Y

    section Training
    Training Data (2007-2014)  :done, 2007, 2014

    section Validation
    PSI Validation (2015)      :active, 2015, 2016
```

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `L01_LoanData_CreditRisk_Preprocessing_PD` | Data preprocessing & feature engineering |
| 02 | `L02_LoanData_CreditRisk_PD_Scorecard_Cutoffs` | PD modelling, scorecard creation & cutoff analysis |
| 03 | `L03_LoanData_CreditRisk_LGD_EAD_and_Expected_Loss` | LGD, EAD modelling & expected loss calculation |
| 04 | `L04_LoanData_CreditRisk_PSI_check` | Population Stability Index validation |

---

## 📂 Dataset

**Source:** [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/wendykan/lending-club-loan-data/version/1)

- **800,000+** consumer loans issued between 2007–2015
- **Training set:** 2007–2014 data
- **Validation set:** 2015 data (used for PSI checks and model stability testing)

---

## 📈 Model Performance

| Model | Algorithm | Accuracy | AUC-ROC |
|-------|-----------|----------|---------|
| **PD** | Logistic Regression | 0.572 | 0.684 |
| **LGD – Stage 1** | Logistic Regression | 0.572 | 0.684 |
| **LGD – Stage 2** | Linear Regression | 0.777 | — |
| **EAD** | Linear Regression | 0.658 | — |

```mermaid
pie title Model Accuracy Comparison
    "PD (57.2%)" : 57.2
    "LGD Stage 2 (77.7%)" : 77.7
    "EAD (65.8%)" : 65.8
```

> ⚠️ These are baseline results. Further optimization (hyperparameter tuning, ensemble methods) can improve performance.

---

## 📦 Deliverables

1. ✅ Credit **Scorecard** — easy-to-interpret, Basel II compliant
2. ✅ **Cutoff analysis** — impact of thresholds on borrower acceptance rates
3. ✅ Trained **LGD, EAD & EL** models
4. ✅ **PSI monitoring** schema for model drift detection

---

## 🗂 Project Structure

```
Credit_Risk_Modelling/
├── 📁 notebooks/          # Jupyter notebooks (L01–L04)
├── 📁 src/                # Helper functions & utilities
├── 📁 data/               # Datasets & scorecard output
├── 📁 models/             # Serialized trained models (.sav)
├── 📄 requirements.txt    # Python dependencies
├── 📄 LICENSE             # MIT License
└── 📄 README.md           # You are here
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/reetika0104/Credit_Risk_Modelling.git
cd Credit_Risk_Modelling

# 2. Install dependencies
pip install -r requirements.txt
```

### Workflow

```mermaid
flowchart TD
    A[Clone Repo] --> B[Install Dependencies]
    B --> C[Open Notebook L01]
    C --> D[Run Preprocessing]
    D --> E[Train PD Model - L02]
    E --> F[Train LGD & EAD - L03]
    F --> G[Check PSI - L04]
    G --> H[✅ Deploy Models]
```

### Alternative: Google Colab

You can run this project directly on [Google Colab](https://colab.research.google.com) — no local setup required. Just load the repo and start running notebooks.

---

## 🛠 Technologies

| Tool | Version |
|------|---------|
| Python | 3.8 |
| Scikit-Learn | 1.0.2 |
| NumPy | latest |
| Pandas | latest |
| Matplotlib | latest |
| Seaborn | latest |
| SciPy | latest |
| Jupyter Notebook | 6.4.12 |

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 👤 Author

<p align="center">
  <img src="https://github.com/reetika0104.png" width="120" style="border-radius: 50%;" alt="Om Singh"/>
</p>

<p align="center">
  <strong>Om Singh</strong><br/>
  <a href="https://github.com/reetika0104">
    <img src="https://img.shields.io/badge/GitHub-reetika0104-181717?logo=github" alt="GitHub"/>
  </a>
</p>

<p align="center">Made with ❤️ in India 🇮🇳</p>
