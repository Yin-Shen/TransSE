import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def load_data(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                sequences.append(parts[1])
                labels.append(int(parts[2]))
    return np.array(sequences), np.array(labels)

def one_hot_encode(sequences):
    encoder = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    return np.array([[encoder.get(base, [0,0,0,0]) for base in seq] for seq in sequences])

def load_model(model_architecture, model_weights):
    with open(model_architecture, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    return model

def evaluate_model(model, test_data, test_labels, species):
    predictions = model.predict(test_data)
    predictions_binary = (predictions > 0.5).astype(int).flatten()
    
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(test_labels, predictions_binary)
    precision = precision_score(test_labels, predictions_binary)
    recall = recall_score(test_labels, predictions_binary)
    f1 = f1_score(test_labels, predictions_binary)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({species})')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{species}.png')
    plt.close()
    
    return roc_auc, accuracy, precision, recall, f1

def main():
    human_test_seq, human_test_labels = load_data('data/human_test.txt')
    mouse_test_seq, mouse_test_labels = load_data('data/mouse_test.txt')
    
    human_test_data = one_hot_encode(human_test_seq)
    mouse_test_data = one_hot_encode(mouse_test_seq)
    
    human_model = load_model('model_architecture_se_human.json', 'model_weights_se_human.h5')
    mouse_model = load_model('model_architecture_se_mouse.json', 'model_weights_se_mouse.h5')
    
    human_results = evaluate_model(human_model, human_test_data, human_test_labels, 'human')
    mouse_results = evaluate_model(mouse_model, mouse_test_data, mouse_test_labels, 'mouse')
    
    print("Human model performance:")
    print(f"AUC: {human_results[0]:.4f}")
    print(f"Accuracy: {human_results[1]:.4f}")
    print(f"Precision: {human_results[2]:.4f}")
    print(f"Recall: {human_results[3]:.4f}")
    print(f"F1-score: {human_results[4]:.4f}")
    
    print("\nMouse model performance:")
    print(f"AUC: {mouse_results[0]:.4f}")
    print(f"Accuracy: {mouse_results[1]:.4f}")
    print(f"Precision: {mouse_results[2]:.4f}")
    print(f"Recall: {mouse_results[3]:.4f}")
    print(f"F1-score: {mouse_results[4]:.4f}")

if __name__ == "__main__":
    main()
