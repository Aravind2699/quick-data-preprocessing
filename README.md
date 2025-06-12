# ML Preprocessing Toolkit

This project provides a toolkit for cleaning, preprocessing, visualizing, and augmenting tabular datasets, with a focus on student data. It is organized for clarity and scalability in machine learning workflows.

## Project Structure

- `dataset/` — Contains raw and processed datasets (e.g., `student_new.csv`).
- `script/` — Python scripts for data processing (e.g., `ml-preprocessing-toolkit.py`).
- `notebooks/` — Jupyter notebooks for exploration and prototyping.
- `models/` — Saved machine learning models.
- `results/` — Generated plots, reports, and evaluation metrics.
- `logs/` — Training and processing logs.
- `configs/` — Configuration files for experiments.
- `tests/` — Unit and integration tests.
- `docs/` — Documentation and references.

## Main Script: `ml-preprocessing-toolkit.py`

### Features
- **Data Cleaning:** Removes duplicates, trims whitespace, handles NaNs and infinite values.
- **Preprocessing:** Imputes missing values, encodes categorical features, and scales numeric features.
- **Visualization:** Plots correlation heatmaps for preprocessed data.
- **Data Augmentation:** Adds synthetic samples by resampling existing data.

### Example Usage
```python
from script.ml-preprocessing-toolkit import preprocess

data = preprocess.load_data('dataset/student_new.csv')
X_processed, feature_names, preprocessor = preprocess.preprocess_data(data)
preprocess.plot_corr_map(X_processed, feature_names)
preprocess.add_data(data, 1000)  # Augment data by 1000 samples
```

## Getting Started
1. Place your dataset in the `dataset/` folder.
2. Run or modify the scripts in `script/` as needed.
3. Use the provided structure to keep your project organized.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- seaborn
- matplotlib

Install dependencies with:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

## License
MIT License
