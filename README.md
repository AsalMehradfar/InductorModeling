<h1 align="center">EM-Aware Physical Synthesis: Neural Inductor Modeling and Intelligent Placement & Routing for RF Circuits</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
  <a href="https://github.com/AsalMehradfar/InductorModeling/stargazers"><img src="https://img.shields.io/github/stars/AsalMehradfar/InductorModeling?style=social" alt="GitHub Stars"/></a>
</p>


<p align="justify">
This repository provides an inductor design and modeling framework for optimizing inductor layout parameters in integrated circuits using machine learning. We train a forward model to predict the quality factor ($Q$) from inductor layout parameters, then perform differentiable backward optimization to infer $(L_v, L_h, L_c)$ values that maximize $Q$. The framework supports both training on your own dataset and running fast, pretrained optimization — each instance takes less than a second on a standard CPU.
</p>

## 📖 Table of Contents

  * [Environment Setup](#%EF%B8%8F-environment-setup)
  * [Usage](#-usage)
  * [Where to Ask for Help](#-where-to-ask-for-help)

## ⚙️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage dependencies.

#### 📦 Install via `inductor.yml`

Clone the repository and create the environment:

```bash
# Clone the repository
git clone https://github.com/AsalMehradfar/InductorModeling.git
cd InductorModeling

# Create the environment from the YAML file
conda env create -f inductor.yml

# Activate the environment
conda activate inductor
```

#### 🔄 Optional: Update Environment

If you make changes to the YAML or add packages later:

```bash
conda env update -f inductor.yml --prune
```

## 🚀 Usage

### 🧠 Train the Forward Model

To train the model on your own data, run:

```bash
python scripts/train.py
```


> 📂 You have two options for providing your dataset:
>
> - Place your raw csv files in the `data/raw/` folder — the code will automatically preprocess them.
>- If you’ve already prepared the full dataset, you can directly place a `processed_data.csv` file in `data/processed/`.

### 🔁 Run Backward Optimization

To use the pretrained model for inductor design (i.e., predicting inductor layout dimensions that maximize $Q$):

```bash
python scripts/backward_eval.py
```

> 📌 Note:
If you only want to use the pretrained model, just place your test data in the `data/processed/` folder. The code will automatically load the trained weights and run the optimization.

## ❓ Where to Ask for Help

<p align="justify" > 
If you have any questions, feel free to open a <a href="https://github.com/AsalMehradfar/InductorModeling/discussions">Discussion</a> and ask your question. You can also email <a href="mailto:mehradfa@usc.edu">mehradfa@usc.edu</a> (Asal Mehradfar).
</p>
