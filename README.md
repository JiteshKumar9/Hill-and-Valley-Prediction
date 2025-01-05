# Hill and Valley Prediction using Logistic Regression

## Project Overview

This project uses a logistic regression model to predict whether a given set of points on a two-dimensional graph represents a **Hill** or a **Valley**. The dataset consists of 100 floating-point values that represent points along the X-axis, and the corresponding **Class** label (0 or 1) indicates whether the points form a **Valley (0)** or a **Hill (1)**.

## Dataset

The dataset contains 1212 records, each with 100 features (V1 to V100) representing the points on the graph. The target variable, **Class**, is a binary label indicating whether the sequence of points forms a Hill (1) or a Valley (0).

### Dataset Columns:
- **V1, V2, ..., V100**: Features representing the points along the X-axis.
- **Class**: Target variable (0 for Valley, 1 for Hill).

### Example:

| V1     | V2     | V3     | ... | V100   | Class |
|--------|--------|--------|-----|--------|-------|
| 39.02  | 36.49  | 38.20  | ... | 39.10  | 0     |
| 1.83   | 1.71   | 1.77   | ... | 1.69   | 1     |
| 68177.69|66138.42|72981.88| ... | 74920.24| 1     |

## Project Steps

### 1. **Data Preprocessing**
   - Import necessary libraries: `pandas`, `numpy`, `matplotlib`, `sklearn`.
   - Load the dataset from a CSV file.
   - Explore the data by checking the first few rows, dataset info, and summary statistics.
   - Define the target variable (`y`) and feature variables (`X`).

### 2. **Data Standardization**
   - Standardize the feature variables to have a mean of 0 and a standard deviation of 1 using `StandardScaler` from `sklearn`.

### 3. **Train-Test Split**
   - Split the dataset into training and testing sets using `train_test_split` from `sklearn`.

### 4. **Model Building**
   - Build a logistic regression model using the `LogisticRegression` class from `sklearn`.
   - Train the model on the training dataset.

### 5. **Model Evaluation**
   - Evaluate the model's performance using metrics like accuracy, precision, recall, and F1 score.

### 6. **Visualization**
   - Plot the first two rows of the dataset to visualize a "Hill" and a "Valley".
   - Display the results and predictions.

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

### Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

**Usage**
Clone the repository or download the project files.
Open the Jupyter notebook or Python script.
Run the cells sequentially to load the dataset, preprocess the data, train the model, and evaluate its performance.

# Clone the repository
git clone https://github.com/your-username/hill-valley-prediction.git

# Navigate to the project directory
cd hill-valley-prediction

# Run the Python script or Jupyter notebook
python hill_valley_prediction.py
Results
The logistic regression model will predict whether a given set of 100 points forms a Hill or a Valley. The evaluation metrics will provide an insight into the model's performance.
