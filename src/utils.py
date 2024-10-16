from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()

def get_roc_plot(y_val, y_val_proba) -> plt.Figure:
    # Plot ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='best')
    return fig

def get_precision_recall_plot(y_val, y_val_proba) -> plt.Figure:
    precision, recall, thresholds_pr = precision_recall_curve(y_val, y_val_proba)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(thresholds_pr, precision[:-1], 'b--', label='Precision')
    ax.plot(thresholds_pr, recall[:-1], 'g-', label='Recall')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')
    return fig