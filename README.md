This repository contains the code, datasets, and experiments for my Master’s thesis at Universität Hildesheim on computer vision for historical art analysis.
The goal of the project is to explore how transformer models classify, interpret, and explain features of historical artworks such as portraits and art styles.

Objective:

Collect and preprocess art portrait and artstyle dataset.

Fine-tune CSWin Model on these datasets.

Evaluate model performance and interpretability.

Use evidence counterfactuals, attention maps and segmentation to analyze model decisions.

Contribution: A pipeline for art dataset preparation, training, and explainability, with case studies on portraits and historical datasets.

🗂 Repository Structure
├── data/                         # Dataset folders (PortraitDataset, HAB, WikiArt, etc.)
├── preprocessing/                # Scripts for data cleaning and cropping
│   ├── DP.py
│   ├── DP_crop.py
│   └── filecount.py
├── training/                     # Model training and fine-tuning
│   ├── finetune.py
│   ├── finetune_portrait.py
│   ├── focal_loss.py
│   └── utils.py
├── interpretability/             # Attention / heatmap analysis
│   ├── attention_coordinates_analysis.py
│   ├── heatmapanal.py
│   └── segmentation_dinoV3.py
├── scraping/                     # Scripts for harvesting images & metadata
│   ├── artscraper.py
│   ├── modified_art_extractor.py
│   └── scrapWikiArt.py
├── notebooks/                    # Jupyter notebooks for experiments
├── requirements.txt              # Dependencies
└── thesis.pdf                    # Final thesis document (if included)


🔬 Research Context

This project builds on research in computational art history, domain adaptation, and explainable AI.
It aims to bridge computer vision with art historical inquiry by making model decisions more transparent and domain-relevant.
