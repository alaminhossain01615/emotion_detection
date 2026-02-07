import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
from itertools import cycle

class Evaluate:
    def __init__(self,model,device):
        self.model=model
        self.device=device

    def plot_history(self,history):
        epochs_range = range(len(history['train_loss']))

        # Plot Loss Curve
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, history['test_loss'], label='Test Loss')
        plt.title('Training and Test Loss Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
        plt.plot(epochs_range, history['test_acc'], label='Test Accuracy')
        plt.title('Training and Test Accuracy Curve')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self,test_data,class_names=["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]):
        print("\n---Confusion Matrix---")
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for test_x, test_y in test_data:
                test_x = test_x.to(self.device)
                outputs = self.model(test_x)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(test_y.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(9, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.show()
    
    def get_predictions(self, data_loader):
        all_preds = []
        all_labels = []
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def print_classification_report(self, data_loader, class_names=None):
        true_labels, pred_labels, _ = self.get_predictions(data_loader)
        
        print("\n" + "="*70)
        print("Classification report")
        print("="*70)
        print(classification_report(true_labels,pred_labels,target_names=class_names, digits=4))
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        
        print("\nEvaluation Metrics (Weighted Average):")
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-Score:   {f1:.4f}")
        print("="*70 + "\n")

    def plot_per_class_metrics(self, data_loader, class_names=None):
        true_labels, pred_labels, _ = self.get_predictions(data_loader)
        
        num_classes = len(np.unique(true_labels))
        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        f1 = f1_score(true_labels, pred_labels, average=None)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()