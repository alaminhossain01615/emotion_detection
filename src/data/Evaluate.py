import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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