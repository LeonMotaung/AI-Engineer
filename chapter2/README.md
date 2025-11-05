# 📊 Python Data Preprocessing Techniques

This repository contains a collection of Python scripts demonstrating common data preprocessing techniques using the `scikit-learn` (sklearn) library. Preprocessing is a crucial step in any machine learning pipeline, ensuring data is in a suitable format for algorithms to learn from effectively.

## 🛠️ Techniques Covered

This project provides simple, standalone examples for the following methods:

* **Mean Removal (Standardization):** Centering data by removing the mean, resulting in a distribution with a mean of 0 and a standard deviation of 1.
* **Scaling (Min-Max Scaling):** Rescaling features to a specific range (e.g., 0 to 1) to create a level playing field for all features.
* **Normalization (L1/L2):** Scaling individual samples (rows) to have a unit norm. This is useful when the magnitude of the sample matters less than its direction.
* **Binarization:** Converting numerical data into boolean values (0 or 1) based on a specific threshold.
* **Label Encoding:** Converting categorical (text) labels into a numerical form that machine learning algorithms can understand.

## 🚀 Getting Started

To run these examples, you'll need to have Python and the `scikit-learn` library installed.

### Prerequisites

* Python 3.x
* NumPy
* scikit-learn

### Installation

1.  **Clone this repository (optional):**
    ```bash
    https://github.com/LeonMotaung/AI-Engineer.git
    cd AI-Engineer
    ```

2.  **Install the required packages:**
    If you don't have the libraries, you can install them via pip:
    ```bash
    pip install numpy scikit-learn
    ```

## 🏃‍♂️ Usage

All examples are contained in a single Python file (e.g., `preprocessing_demo.py`). You can run the file directly to see the printed output for each technique.

```bash
python preprocessing_demo.py
