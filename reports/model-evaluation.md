Three classification algorithms were evaluated using the same training and test datasets, with an 80/20 train-test split and random_state=42. The models were compared using accuracy, F1 score, and ROC-AUC.

Experiment Comparison
Model	Accuracy	F1 Score	ROC-AUC
Logistic Regression	0.982	0.986	0.995
Random Forest	0.956	0.966	0.994
Gradient Boosting	0.947	0.959	0.991
Findings
Logistic Regression achieved the best overall performance among the three evaluated models. It obtained an accuracy of 98.2%, an F1 score of 0.986, and a ROC-AUC of 0.995.

Random Forest achieved an accuracy of 95.6%, an F1 score of 0.966, and a ROC-AUC of 0.994. Gradient Boosting achieved an accuracy of 94.7%, an F1 score of 0.959, and a ROC-AUC of 0.991.

All three models demonstrated strong classification performance on the test dataset. Logistic Regression performed best on all three reported metrics, making it the preferred candidate for the next stage of the pipeline.

Model Selection Rationale
Logistic Regression was selected because it achieved the highest accuracy, F1 score, and ROC-AUC among the evaluated models. It also provides a relatively simple and interpretable baseline for a classification task.

The selected model will be validated before registration and deployment. Since the dataset's positive class represents benign tumors, further evaluation should pay particular attention to recall for the malignant class to understand the risk of false-negative predictions.

MLflow Experiment Tracking
All three experiments were logged to MLflow. The tracked information includes the algorithm, random state, test-set proportion, evaluation metrics, and model artifacts. This enables experiment comparison and supports reproducibility.

The next step is to validate the selected Logistic Regression model, log it as the final candidate, and register it in Unity Catalog if the required functionality and permissions are available in Databricks Free Edition.
