# Breast Cancer Classification — MLOps Capstone

An end-to-end Machine Learning Operations (MLOps) project built with **Databricks, PySpark, Scikit-learn, and MLflow**. This project demonstrates data exploration, classification model training, experiment tracking, model selection, model registration with Unity Catalog, and real-time model serving.

## Project Description

This capstone project develops a binary classification pipeline using the **Breast Cancer Wisconsin Diagnostic dataset** from Scikit-learn. It explores machine learning workflows in Databricks and applies MLOps practices to track experiments, compare models, register a selected model, and prepare it for deployment through a serving endpoint.

The project is designed as an educational demonstration of an end-to-end ML lifecycle, from data preparation to model deployment.

> **Disclaimer:** This project uses a public educational dataset. It is not intended for clinical diagnosis or medical decision-making.

## Objectives

- Explore and preprocess a structured dataset using Python and Spark.
- Train and evaluate multiple classification algorithms.
- Track experiments, parameters, metrics, and model artifacts with MLflow.
- Compare candidate models and select a model based on evaluation metrics.
- Log models with input/output signatures for model management.
- Register the selected model in Databricks Unity Catalog.
- Deploy and test a real-time serving endpoint, subject to workspace availability.
- Maintain project dependencies and source code in GitHub.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Model development and evaluation |
| Databricks Free Edition | Cloud notebook and ML development environment |
| Apache Spark / PySpark | Distributed data processing and exploration |
| Pandas | Tabular data analysis |
| NumPy | Numerical operations |
| Scikit-learn | Dataset, preprocessing, model training, and evaluation |
| MLflow | Experiment tracking, model logging, and model registry |
| Unity Catalog | Centralized model registration and governance |
| Databricks Model Serving | Real-time inference deployment |
| Git and GitHub | Version control and project documentation |

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through `sklearn.datasets.load_breast_cancer`.

- **Samples:** 569
- **Features:** 30 numerical features
- **Target:** Binary classification
- **Classes:** Malignant and benign

The target encoding used by Scikit-learn is:

- `0` — Malignant
- `1` — Benign

The dataset contains numerical measurements computed from digitized images of breast mass samples. Data exploration and model evaluation are performed before selecting a candidate model.

## Machine Learning Workflow

1. **Data preparation:** Load the dataset, inspect its schema, and prepare features and labels.
2. **Exploratory data analysis:** Review feature distributions and dataset characteristics.
3. **Train-test split:** Split the dataset into training and testing subsets using a fixed random state.
4. **Model training:** Train and compare Logistic Regression, Random Forest, and Gradient Boosting classifiers.
5. **Experiment tracking:** Record model parameters, evaluation metrics, and artifacts in MLflow.
6. **Model selection:** Compare accuracy, F1 score, and ROC-AUC to select a candidate model.
7. **Model logging:** Log the selected model with an input example and model signature.
8. **Model registration:** Register the model in Unity Catalog.
9. **Model serving:** Configure a Databricks serving endpoint and validate prediction requests when the workspace supports deployment.
10. **Version control:** Store project code, dependencies, and documentation in GitHub.

## Experiment Results

The following metrics were recorded during the initial experiments using an 80/20 train-test split and `random_state=42`.

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.982 | 0.986 | 0.995 |
| Random Forest | 0.956 | 0.966 | 0.994 |
| Gradient Boosting | 0.947 | 0.959 | 0.991 |

**Initial selected model: Logistic Regression**

Logistic Regression achieved the highest recorded accuracy, F1 score, and ROC-AUC among the three candidates in these experiments.

These values represent the initial recorded evaluation results and may vary if preprocessing, data splitting, package versions, or model parameters change.

**Important evaluation consideration:** The default F1 score treats class `1` (benign) as the positive class. Because malignant cases are class `0`, malignant-class recall and the confusion matrix should also be reviewed before drawing conclusions about model performance.

## MLflow Experiment Tracking

MLflow is used to organize and compare the model experiments. The recorded information includes:

- Model algorithm and experiment parameters
- Test accuracy
- Test F1 score
- Test ROC-AUC
- Logged model artifacts
- The final selected model

The experiment tracking UI in Databricks can be used to inspect individual runs and compare their performance.

## Model Registry

The selected model is intended to be registered in Unity Catalog using the following three-level name:

`workspace.mlops.breast_cancer_classifier`

The registered model supports versioned model management. The model is logged with a signature specifying the expected input and output structure, which is required for Unity Catalog model registration.

The `champion` and `challenger` aliases can be used to identify the current preferred model version and a candidate version for comparison after successful registration and validation.

## Model Serving

The project includes a Databricks Model Serving deployment workflow for exposing the registered classifier through an HTTP endpoint.

The serving endpoint can be used to:

- Submit feature data for inference
- Receive classification predictions
- Test the model independently of the training notebook
- Validate the transition from model development to deployment

Endpoint availability and deployment behavior depend on the Databricks workspace, compute availability, permissions, and applicable service limits. A configured endpoint should be considered operational only after it reaches a ready state and successfully returns predictions.

## Repository Structure

```text
mlops-classification-capstone/
├── notebooks/
│   └── mlops_classification_capstone
├── requirements.txt
├── README.md
└── LICENSE
```

The notebook contains the data preparation, model training, experiment tracking, evaluation, and model management workflow. Update the notebook filename in this tree if it differs in your repository.

## Getting Started

### Prerequisites

- A Databricks workspace with notebook access
- Python 3
- Git
- A GitHub account
- Access to MLflow and, for the deployment stage, the relevant Unity Catalog and model-serving features

### Install dependencies

Clone the repository:

```bash
git clone https://github.com/Koheinn/mlops-classification-capstone.git
cd mlops-classification-capstone
```

Create and activate a virtual environment if running locally:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Run the project

1. Open your Databricks workspace.
2. Import or open the project notebook from the `notebooks/` directory.
3. Execute the notebook cells in order.
4. Review the dataset exploration and model evaluation results.
5. Inspect the recorded MLflow experiment runs.
6. Register the selected model in Unity Catalog.
7. If model serving is available, deploy the registered model and test the endpoint.

The notebook may require environment-specific configuration when run outside Databricks.

## Limitations

- The project uses a relatively small, prepackaged educational dataset.
- The initial comparison uses a single train-test split rather than comprehensive cross-validation.
- Accuracy, F1 score, and ROC-AUC alone do not fully describe the risk of missing malignant cases.
- Databricks Free Edition may restrict compute resources, model serving, or other platform features.
- Real-time serving, model aliases, and deployment should only be described as completed after successful verification in the workspace.

## Future Improvements

- Add more extensive exploratory data analysis and visualizations.
- Implement cross-validation and systematic hyperparameter tuning.
- Add a confusion matrix, classification report, and malignant-class recall.
- Track preprocessing and model parameters consistently across all experiments.
- Add automated tests and a reproducible training script.
- Implement model validation before promoting a candidate to `champion`.
- Add a CI workflow for code quality and automated testing.
- Compare local inference with cloud model serving.
- Extend the project with model monitoring and data-drift detection.

## Learning Outcomes

This project provides practical experience with:

- Building and evaluating classification models
- Working with Spark and Python-based data science tools
- Tracking machine learning experiments with MLflow
- Managing registered model versions
- Preparing models for real-time inference
- Using Git and GitHub to document an ML project

## Author

**Koheinn**

- GitHub: https://github.com/Koheinn

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
