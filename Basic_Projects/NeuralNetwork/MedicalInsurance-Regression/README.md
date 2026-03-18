# Insurance Charges Prediction using Deep Learning

This project predicts insurance charges using a neural network built with TensorFlow and Keras. The model is trained on the popular **insurance dataset**, which includes features such as age, BMI, number of children, smoking status, and region. The improved neural network model achieves an **R² score of 0.86** on the test set, demonstrating strong predictive performance.

---

## Dataset

- The dataset contains columns like:
  - `age` – Age of the policyholder
  - `sex` – Gender (male/female)
  - `bmi` – Body Mass Index
  - `children` – Number of children
  - `smoker` – Whether the person smokes
  - `region` – Geographical region
  - `charges` – Insurance charges (target variable)

- Missing values were checked and there were **no missing entries**.

- Categorical features were converted to numerical values using **one-hot encoding**.

---

## Project Workflow

1. **Data Loading & Exploration**
   - Load CSV data using `pandas`
   - Inspect data using `.head()`, `.info()`, and `.describe()`
   - Check for missing values

2. **Data Preprocessing**
   - Convert categorical variables using `pd.get_dummies()`
   - Split data into features `X` and target `y`
   - Train-test split using `sklearn.model_selection.train_test_split` (80%-20%)
   - Scale features using `MinMaxScaler`
   - Reshape `y` to 2D arrays for TensorFlow

3. **Modeling**
   - **Baseline Model**: Simple 1→1 dense network
     - Loss: Mean Absolute Error (MAE)
     - Optimizer: SGD
   - **Improved Model**:
     - Architecture: 128 → 64 → 32 → 1 dense layers
     - Activations: ReLU for hidden layers, linear for output
     - Loss: Huber (robust to outliers)
     - Optimizer: Adam (learning rate = 0.001)
     - Early stopping with validation split (20%) to prevent overfitting

4. **Training**
   - Trained up to 500 epochs (stopped early if no improvement)
   - Tracked MAE and loss on both training and validation sets

5. **Evaluation**
   - Model evaluated on the test set
   - **R² Score**: 0.86
   - Loss curve plotted to visualize convergence

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/AreejFatiima/Deep-Learning.git