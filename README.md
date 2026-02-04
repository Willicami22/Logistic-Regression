# Heart Disease Risk Prediction: Logistic Regression

A machine learning project implementing logistic regression from scratch for heart disease prediction, featuring exploratory data analysis, model training with visualization, regularization techniques, and AWS SageMaker deployment preparation.

## Table of Contents
- [Exercise Summary](#exercise-summary)
- [Dataset Description](#dataset-description)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installing](#installing)
- [Deployment](#deployment)
- [Built With](#built-with)
- [Authors](#authors)
- [License](#license)

## Exercise Summary

This project implements a comprehensive logistic regression model for heart disease prediction, including:

- **Exploratory Data Analysis (EDA)**: Statistical analysis and visualization of patient health metrics
- **Model Training**: Custom implementation of logistic regression with gradient descent optimization
- **Visualization**: Decision boundaries, probability distributions, and performance metrics
- **Regularization**: L2 regularization to prevent overfitting and improve model generalization
- **AWS SageMaker Deployment**: Preparation for cloud-based model deployment (deployment ready)

The model achieves **82.72% accuracy** and **0.80 F1-score** on the test set with optimal regularization (λ=1.0), demonstrating reliable heart disease risk prediction capabilities.

## Dataset Description

**Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)

**Dataset Characteristics**:
- **Size**: 303 patients
- **Features**: 13 clinical and diagnostic attributes
- **Target**: Binary classification (Presence/Absence of heart disease)
- **Disease Presence Rate**: ~55% (165 positive cases)

**Key Features**:
- **Age**: 29-77 years
- **Cholesterol**: 112-564 mg/dL
- **Blood Pressure (BP)**: Resting blood pressure
- **Max HR**: Maximum heart rate achieved
- **ST Depression**: Exercise-induced ST segment depression
- **Sex**: Gender (0=Female, 1=Male)
- **Chest Pain Type**: 4 categories of chest pain
- **FBS over 120**: Fasting blood sugar > 120 mg/dL
- **EKG Results**: Resting electrocardiographic results
- **Exercise Angina**: Exercise-induced angina
- **Slope of ST**: ST segment slope during peak exercise
- **Number of Vessels Fluro**: Number of major vessels colored by fluoroscopy
- **Thallium**: Thallium stress test results

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development, testing, and analysis purposes.

### Prerequisites

You need to have Python 3.8 or higher installed on your system. The project requires the following Python packages:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

For AWS SageMaker deployment (optional):
```
boto3>=1.20.0
sagemaker>=2.0.0
```

### Installing

Follow these steps to set up your development environment:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

5. **Download the dataset**
   - Visit [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)
   - Download `heart.csv` and place it in the project directory

6. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook heart_disease_lr_analysis.ipynb
   ```

## Deployment

### AWS SageMaker Deployment (AWS Academy Lab Environment)

This project includes preparation for deployment to AWS SageMaker for scalable, production-ready inference. However, **deployment was not completed due to AWS Academy Learner Lab restrictions**, which limit permissions for creating SageMaker resources in educational environments.

#### Planned Deployment Process:

1. **Model Serialization**
   - Export trained logistic regression model (weights, bias, scaler)
   - Package model artifacts in SageMaker-compatible format

2. **Training Job Configuration**
   - Define training job with custom scikit-learn estimator
   - Configure instance type (e.g., `ml.m5.large`)
   - Set hyperparameters (learning rate, regularization lambda, iterations)

3. **Endpoint Creation**
   - Create model endpoint configuration
   - Deploy model to SageMaker endpoint
   - Configure auto-scaling policies

4. **Inference Testing**
   - Test endpoint with sample patient data
   - Example: Input `[Age=60, Cholesterol=300, ...]` → Output: `Probability=0.68 (High Risk)`

#### Deployment Limitations:

**Status**:  **Not Deployed - AWS Academy Lab Restrictions**

The deployment to AWS SageMaker could not be completed due to **AWS Academy Learner Lab access limitations**:

- **Restricted IAM Permissions**: AWS Academy Lab environments have read-only permissions for many services, preventing the creation of SageMaker training jobs, models, and endpoints
- **Limited Service Access**: SageMaker service is typically restricted in educational lab environments
- **S3 Bucket Limitations**: Restricted permissions to create or modify S3 buckets required for model artifacts
- **EC2 Instance Provisioning**: Unable to provision dedicated EC2 instances for SageMaker endpoints
- **Temporary Session Credentials**: Lab sessions have time-limited credentials that expire, making long-running deployments impractical


## Built With

* **[Python 3.14](https://www.python.org/)** - Programming language
* **[NumPy 2.4.2](https://numpy.org/)** - Numerical computing and array operations
* **[Pandas 3.0.0](https://pandas.pydata.org/)** - Data manipulation and analysis
* **[Matplotlib 3.10.8](https://matplotlib.org/)** - Data visualization and plotting
* **[Scikit-learn 1.8.0](https://scikit-learn.org/)** - Machine learning utilities (preprocessing, metrics)
* **[Jupyter Notebook](https://jupyter.org/)** - Interactive development environment
* **[AWS SageMaker](https://aws.amazon.com/sagemaker/)** - Cloud ML deployment platform (planned)

### Key Libraries Used:

- `sklearn.model_selection.train_test_split` - Dataset splitting
- `sklearn.preprocessing.StandardScaler` - Feature normalization
- `sklearn.metrics` - Model evaluation (accuracy, precision, recall, F1-score)
- `matplotlib.pyplot` - Visualization of decision boundaries and metrics

## Authors

* **William Camilo Hernandez Deaza** - *Initial work and implementation* - [Willicami22](https://github.com/Willicami22)


---

## Additional Information

### Model Performance

**Optimal Configuration** (λ=1.0):
- **Test Accuracy**: 82.72%
- **Test F1-Score**: 0.80
- **Precision**: 0.82
- **Recall**: 0.78
- **Weight Norm Reduction**: 5.49% (compared to λ=0)

### Key Findings

1. **Regularization Impact**: L2 regularization with λ=1.0 improved F1-score by 1.43% and reduced overfitting
2. **Feature Importance**: Age and ST Depression showed strong discriminative power in 2D analysis
3. **Class Balance**: Dataset is relatively balanced (~55% disease presence), minimizing bias concerns
4. **Model Interpretability**: Decision boundaries clearly separate risk groups with probability-based predictions


### Acknowledgments

* Dataset provided by [Kaggle](https://www.kaggle.com/datasets/neurocipher/heartdisease)
* Inspired by cardiovascular health research and preventive medicine
* Thanks to the open-source community for excellent ML libraries