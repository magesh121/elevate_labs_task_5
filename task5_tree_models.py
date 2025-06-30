# Enhanced Decision Tree Analysis with Improved Accuracy and Comprehensive Visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, 
                           classification_report, roc_curve, auc, precision_recall_curve,
                           precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load dataset
print("📊 Loading heart disease dataset...")
df = pd.read_csv("heart.csv")

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# Feature matrix and target
X = df.drop('target', axis=1)
y = df['target']

# Data preprocessing
print("\n🔧 Preprocessing data...")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ==================== HYPERPARAMETER TUNING ====================
print("\n🔍 Performing hyperparameter tuning...")

# Grid search for Decision Tree
dt_param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

print(f"Best Decision Tree parameters: {dt_grid.best_params_}")
print(f"Best cross-validation score: {dt_grid.best_score_:.4f}")

# ==================== MODEL TRAINING ====================
print("\n🚀 Training models...")

# Train optimized decision tree
y_pred_dt = best_dt.predict(X_test)
y_pred_proba_dt = best_dt.predict_proba(X_test)[:, 1]

# Random Forest with tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_
y_pred_gb = best_gb.predict(X_test)
y_pred_proba_gb = best_gb.predict_proba(X_test)[:, 1]

# ==================== VISUALIZATIONS ====================
print("\n📈 Creating visualizations...")

# 1. Dataset Overview
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Heart Disease Dataset Overview', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Gender distribution
gender_counts = df['sex'].value_counts()
axes[0, 1].pie(gender_counts.values, labels=['Female', 'Male'], autopct='%1.1f%%', 
               colors=['lightcoral', 'lightblue'])
axes[0, 1].set_title('Gender Distribution')

# Target distribution
target_counts = df['target'].value_counts()
axes[0, 2].bar(['No Disease', 'Disease'], target_counts.values, 
               color=['lightgreen', 'salmon'], alpha=0.7)
axes[0, 2].set_title('Target Distribution')
axes[0, 2].set_ylabel('Count')

# Cholesterol distribution
axes[1, 0].hist(df['chol'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Cholesterol Distribution')
axes[1, 0].set_xlabel('Cholesterol')
axes[1, 0].set_ylabel('Frequency')

# Blood pressure distribution
axes[1, 1].hist(df['trestbps'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1, 1].set_title('Blood Pressure Distribution')
axes[1, 1].set_xlabel('Blood Pressure')
axes[1, 1].set_ylabel('Frequency')

# Heart rate distribution
axes[1, 2].hist(df['thalach'], bins=20, alpha=0.7, color='gold', edgecolor='black')
axes[1, 2].set_title('Max Heart Rate Distribution')
axes[1, 2].set_xlabel('Max Heart Rate')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("outputs/dataset_overview.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Importance Comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')

# Decision Tree feature importance
dt_importance = best_dt.feature_importances_
dt_indices = np.argsort(dt_importance)[::-1]
axes[0].bar(range(len(dt_importance)), dt_importance[dt_indices], color='skyblue')
axes[0].set_title('Decision Tree')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Importance')
axes[0].set_xticks(range(len(dt_importance)))
axes[0].set_xticklabels([selected_features[i] for i in dt_indices], rotation=45)

# Random Forest feature importance
rf_importance = best_rf.feature_importances_
rf_indices = np.argsort(rf_importance)[::-1]
axes[1].bar(range(len(rf_importance)), rf_importance[rf_indices], color='lightgreen')
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Importance')
axes[1].set_xticks(range(len(rf_importance)))
axes[1].set_xticklabels([selected_features[i] for i in rf_indices], rotation=45)

# Gradient Boosting feature importance
gb_importance = best_gb.feature_importances_
gb_indices = np.argsort(gb_importance)[::-1]
axes[2].bar(range(len(gb_importance)), gb_importance[gb_indices], color='salmon')
axes[2].set_title('Gradient Boosting')
axes[2].set_xlabel('Features')
axes[2].set_ylabel('Importance')
axes[2].set_xticks(range(len(gb_importance)))
axes[2].set_xticklabels([selected_features[i] for i in gb_indices], rotation=45)

plt.tight_layout()
plt.savefig("outputs/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Model Performance Comparison
models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
accuracies = [
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_gb)
]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'], alpha=0.7)
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy values on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/model_accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. ROC Curves
plt.figure(figsize=(10, 8))
y_pred_probas = [y_pred_proba_dt, y_pred_proba_rf, y_pred_proba_gb]
colors = ['blue', 'green', 'red']

for i, (model_name, y_pred_proba, color) in enumerate(zip(models, y_pred_probas, colors)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Precision-Recall Curves
plt.figure(figsize=(10, 8))

for i, (model_name, y_pred_proba, color) in enumerate(zip(models, y_pred_probas, colors)):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color=color, lw=2, 
             label=f'{model_name} (AUC = {pr_auc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/precision_recall_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

y_preds = [y_pred_dt, y_pred_rf, y_pred_gb]
colors = ['Blues', 'Greens', 'Reds']

for i, (model_name, y_pred, color) in enumerate(zip(models, y_preds, colors)):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=axes[i], cmap=color, values_format='d')
    axes[i].set_title(model_name)

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. Decision Tree Visualization (optimized)
plt.figure(figsize=(20, 12))
plot_tree(best_dt, feature_names=selected_features, class_names=["No Disease", "Disease"], 
          filled=True, rounded=True, fontsize=10)
plt.title("Optimized Decision Tree Visualization", fontsize=16, fontweight='bold')
plt.savefig("outputs/optimized_decision_tree.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. Learning Curves (Cross-validation scores)
depths = range(1, 16)
dt_scores = []
rf_scores = []

for depth in depths:
    # Decision Tree
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_score = cross_val_score(dt_temp, X_selected, y, cv=5).mean()
    dt_scores.append(dt_score)
    
    # Random Forest
    rf_temp = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
    rf_score = cross_val_score(rf_temp, X_selected, y, cv=5).mean()
    rf_scores.append(rf_score)

plt.figure(figsize=(12, 8))
plt.plot(depths, dt_scores, marker='o', label='Decision Tree', linewidth=2, markersize=8)
plt.plot(depths, rf_scores, marker='s', label='Random Forest', linewidth=2, markersize=8)
plt.title('Learning Curves: Depth vs Cross-Validation Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/learning_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 10. Feature Selection Analysis
plt.figure(figsize=(12, 8))
feature_scores = selector.scores_
feature_names = X.columns
feature_indices = np.argsort(feature_scores)[::-1]

plt.bar(range(len(feature_scores)), feature_scores[feature_indices], 
        color='purple', alpha=0.7)
plt.title('Feature Selection Scores (F-statistic)', fontsize=16, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('F-statistic Score')
plt.xticks(range(len(feature_scores)), 
           [feature_names[i] for i in feature_indices], rotation=45)
plt.tight_layout()
plt.savefig("outputs/feature_selection_scores.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================== RESULTS SUMMARY ====================
print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

def print_model_metrics(model_name, y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

y_preds = [y_pred_dt, y_pred_rf, y_pred_gb]
y_pred_probas = [y_pred_proba_dt, y_pred_proba_rf, y_pred_proba_gb]

for model_name, y_pred, y_pred_proba in zip(models, y_preds, y_pred_probas):
    print_model_metrics(model_name, y_test, y_pred, y_pred_proba)

print(f"\n🎯 Best Model: {models[np.argmax(accuracies)]}")
print(f"🏆 Best Accuracy: {max(accuracies):.4f} ({max(accuracies)*100:.2f}%)")

print(f"\n📁 All visualizations saved in 'outputs/' directory:")
print("  - dataset_overview.png")
print("  - correlation_heatmap.png")
print("  - feature_importance_comparison.png")
print("  - model_accuracy_comparison.png")
print("  - roc_curves.png")
print("  - precision_recall_curves.png")
print("  - confusion_matrices.png")
print("  - optimized_decision_tree.png")
print("  - learning_curves.png")
print("  - feature_selection_scores.png")

print("\n✅ Analysis complete!") 