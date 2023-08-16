import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

 

# Step 1: Data Preparation
# Load your dataset (replace 'Churn_Modelling.csv' with your actual data file)
data = pd.read_csv('E:\Churn_Modelling.csv')

 

# Separate features and target variable
X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])  # Remove columns that are not features or target
y = data['Exited']

 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 

# Define preprocessing steps for numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

 

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

 

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

 

# Step 3: Model Selection and Training
# Build pipelines with preprocessing and model training for each algorithm
pipelines = {
    'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                            ('model', LogisticRegression(random_state=42))]),

    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', RandomForestClassifier(random_state=42))]),

    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('model', GradientBoostingClassifier(random_state=42))])
}

 

# Train, evaluate, and visualize each model
for name, pipeline in pipelines.items():
    print(f"Training and evaluating {name}...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

 

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

 

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    print("=" * 40)