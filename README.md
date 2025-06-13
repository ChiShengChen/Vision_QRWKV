# Vision QRWKV: Exploring Quantum-Enhanced RWKV Models for Image Classification

[![arXiv](https://img.shields.io/badge/arXiv-2506.06633-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2506.06633)  
![image](https://github.com/user-attachments/assets/dba471bb-2198-48ef-9bd6-04f406d7b76c)

## Introduction

This project explores the application of classical and quantum Receptance Weighted Key Value (RWKV) models for image classification tasks. It includes implementations of both model types and scripts to train and evaluate them on a variety of MedMNIST datasets, as well as standard datasets like MNIST and FashionMNIST. The primary goal is to compare the performance characteristics of quantum RWKV models against their classical counterparts in the context of visual recognition.

## Key Features

*   Implementation of Classical RWKV model (`rwkv.py`).
*   Implementation of Quantum-enhanced RWKV model (`quantum_rwkv.py`).
*   Test scripts for training and evaluation on multiple datasets:
    *   **MedMNIST**: BloodMNIST, TissueMNIST, OCTMNIST, PathMNIST, ChestMNIST, OrganAMNIST, OrganSMNIST, OrganCMNIST, DermaMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST.
    *   **Standard**: MNIST, FashionMNIST.
*   Automated saving of training progress, epoch-wise metrics, confusion matrices, and overall performance summaries in CSV and PNG formats.
*   Performance comparison between classical and quantum models, summarized in `model_comparison_summary.csv`.

## Project Structure

A brief overview of important files and directories:

*   `rwkv.py`: Core implementation of the classical RWKV model.
*   `quantum_rwkv.py`: Core implementation of the quantum-enhanced RWKV model.
*   `test_classical_<dataset_name>.py`: Scripts to run classical model experiments for a given `<dataset_name>`.
*   `test_quantum_<dataset_name>.py`: Scripts to run quantum model experiments for a given `<dataset_name>`.
*   `results_<dataset>_<modeltype>_<timestamp>/`: Directories where all outputs (logs, plots, CSV summaries) for each experiment run are stored.
*   `model_comparison_summary.csv`: A summary CSV comparing the final test accuracies of classical and quantum models across all tested datasets.
*   `data/`: Default directory where datasets are downloaded by `medmnist` library. This directory can be added to `.gitignore`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ChiShengChen/Vision_QRWKV.git
    cd Vision_QuRWKV
    ```

2.  **Create a Python Environment:**
    It's recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    # Using conda
    conda create -n qurwkv python=3.10  # Or your preferred Python version
    conda activate qurwkv
    ```

3.  **Install Dependencies:**
    The main dependencies include PyTorch, MedMNIST, scikit-learn, Matplotlib, and Seaborn. You should create a `requirements.txt` file for your project.
    A basic set of dependencies can be installed via pip:
    ```bash
    pip install torch torchvision torchaudio
    pip install medmnist numpy scikit-learn matplotlib seaborn
    # Add any other specific libraries your RWKV or Quantum RWKV implementations might need
    ```
    *It is highly recommended to generate a `requirements.txt` from your working environment:*
    ```bash
    pip freeze > requirements.txt
    ```

## Usage

To run an experiment for a specific dataset and model type, execute the corresponding Python script from the `Vision_QuRWKV` directory.

For example:

*   To train and evaluate a classical RWKV model on BloodMNIST:
    ```bash
    python test_classical_bloodmnist.py
    ```

*   To train and evaluate a quantum RWKV model on OrganAMNIST:
    ```bash
    python test_quantum_organamnist.py
    ```

Results, including training logs, performance metrics (loss, accuracy), and confusion matrix plots, will be saved in a uniquely timestamped subdirectory (e.g., `results_bloodmnist_classical_YYYYMMDD_HHMMSS/`) within the `Vision_QuRWKV` directory.

## Results Summary

The project includes a comparative analysis of classical and quantum RWKV models across various datasets. The following table summarizes the final test accuracies achieved:

| Dataset        | Classical Accuracy (%) | Quantum Accuracy (%) | Quantum Better? |
|----------------|------------------------|----------------------|-----------------|
| BloodMNIST     | 91.32                  | 92.22                | Yes             |
| TissueMNIST    | 55.43                  | 55.48                | Yes             |
| OCTMNIST       | 55.80                  | 57.00                | Yes             |
| PathMNIST      | 74.00                  | 71.11                | No              |
| ChestMNIST     | 74.44                  | 77.26                | Yes             |
| OrganAMNIST    | 78.24                  | 76.27                | No              |
| OrganSMNIST    | 60.45                  | 59.18                | No              |
| OrganCMNIST    | 75.60                  | 74.66                | No              |
| DermaMNIST     | 71.97                  | 71.97                | Equal           |
| PneumoniaMNIST | 83.81                  | 84.78                | Yes             |
| RetinaMNIST    | 49.25                  | 53.75                | Yes             |
| BreastMNIST    | 77.56                  | 77.56                | Equal           |
| FashionMNIST   | 85.56                  | 86.08                | Yes             |
| MNIST          | 96.49                  | 96.12                | No              |

This summary is also available in the `model_comparison_summary.csv` file.

## Future Work / TODO

*   Finalize and include a `requirements.txt` file.
*   Explore different quantum circuit designs or quantum machine learning techniques within the `quantum_rwkv.py` model.
*   Conduct more extensive hyperparameter tuning for both classical and quantum models.
*   Expand testing to larger-scale datasets or more complex vision tasks.
*   Investigate the impact of different RWKV configurations (e.g., number of layers, embedding dimensions).

## Contributing

Contributions to this project are welcome. Please feel free to open an issue to discuss potential changes or submit a pull request.

## Citation
If this code or paper helps your project, please cite this paper!  
```
@article{chen2025vision,
  title={Vision-QRWKV: Exploring Quantum-Enhanced RWKV Models for Image Classification},
  author={Chen, Chi-Sheng},
  journal={arXiv preprint arXiv:2506.06633},
  year={2025}
}
```
