import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, models, optimizers, callbacks, initializers, backend as K
from tensorflow.keras.layers import LSTM, Input, Conv1D, MaxPooling1D, Activation, Flatten, Dense, Dropout, GRU
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def train_merged_model(data_train, label_train, data_val, label_val, data_test, label_test):
    np.random.seed(666)
    random_sequences = np.random.randint(0, 4, size=(data_train.shape[0], 3000, 4))
    random_labels = np.zeros((data_train.shape[0],))
    data_train = np.concatenate((data_train, random_sequences), axis=0)
    label_train = np.concatenate((label_train, random_labels), axis=0)

    inputs = Input(shape=(3000, 4))
    conv1 = Conv1D(filters=64, kernel_size=7, activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    norm1 = BatchNormalization()(pool1)
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(norm1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    dropout1 = Dropout(0.3)(pool2)
    norm2 = BatchNormalization()(dropout1)
    conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(norm2)
    lstm1 = LSTM(64, return_sequences=True)(conv3)
    lstm2 = LSTM(64)(lstm1)
    norm3 = BatchNormalization()(lstm2)
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(norm3)
    output_layer = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    checkpointer = ModelCheckpoint(filepath='model_weights_se_merge.h5', verbose=1, save_best_only=True)

    time_start = time.time()
    result = model.fit(
        data_train, label_train, batch_size=128, epochs=100, validation_data=(data_val, label_val),
        callbacks=[checkpointer, early_stopping])

    json_string = model.to_json()
    with open('model_architecture_se_merge.json', 'w') as f:
        f.write(json_string)
    model.save_weights('model_weights_se_merge.h5')

    model.load_weights('model_weights_se_merge.h5')

    score1 = model.evaluate(data_test, label_test, verbose=0)
    print('Test loss:', score1[0])
    print('Test accuracy:', score1[1])

    time_end = time.time()
    print('training time : %d sec' % (time_end-time_start))

    pred = model.predict(data_test)
    fpr, tpr, _ = roc_curve(label_test, pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label_test, pred)
    pr_auc = average_precision_score(label_test, pred)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig('roc_pr_curves_se_merge.png')
    plt.close()

    return model

def train_species_model(data_train, label_train, data_val, label_val, data_test, label_test, species, merged_model):
    np.random.seed(666)
    random_sequences = np.random.randint(0, 4, size=(data_train.shape[0], 3000, 4))
    random_labels = np.zeros((data_train.shape[0],))
    data_train = np.concatenate((data_train, random_sequences), axis=0)
    label_train = np.concatenate((label_train, random_labels), axis=0)

    class_weights = {0: 1.0, 1: 0.6}

    model1 = Sequential()
    for layer in merged_model.layers[:-5]: 
        model1.add(layer)

    model1.add(GRU(32, return_sequences=True))
    model1.add(GRU(16))
    model1.add(Dense(8, activation="relu", name='output_layer1'))
    model1.add(Dense(1, activation="sigmoid", name='output_layer'))
    model1.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    checkpointer = ModelCheckpoint(filepath=f'model_weights_se_{species}.h5', verbose=1, save_best_only=True)

    time_start = time.time()
    result = model1.fit(
        data_train, label_train, batch_size=128, epochs=100, validation_data=(data_val, label_val),
        callbacks=[checkpointer, early_stopping], class_weight=class_weights
    )

    json_string = model1.to_json()
    with open(f'model_architecture_se_{species}.json', 'w') as f:
        f.write(json_string)
    model1.save_weights(f'model_weights_se_{species}.h5')

    model1.load_weights(f'model_weights_se_{species}.h5')

    score1 = model1.evaluate(data_test, label_test, verbose=0)
    print('Test loss:', score1[0])
    print('Test accuracy:', score1[1])

    time_end = time.time()
    print('training time : %d sec' % (time_end-time_start))

    pred = model1.predict(data_test)
    fpr, tpr, _ = roc_curve(label_test, pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label_test, pred)
    pr_auc = average_precision_score(label_test, pred)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({species})')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({species})')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f'roc_pr_curves_se_{species}.png')
    plt.close()

def main(args):
    human_train_seq, human_train_labels = load_data(args.human_train)
    human_val_seq, human_val_labels = load_data(args.human_val)
    human_test_seq, human_test_labels = load_data(args.human_test)
    
    mouse_train_seq, mouse_train_labels = load_data(args.mouse_train)
    mouse_val_seq, mouse_val_labels = load_data(args.mouse_val)
    mouse_test_seq, mouse_test_labels = load_data(args.mouse_test)

    merged_train_seq = np.concatenate((human_train_seq, mouse_train_seq))
    merged_train_labels = np.concatenate((human_train_labels, mouse_train_labels))
    merged_val_seq = np.concatenate((human_val_seq, mouse_val_seq))
    merged_val_labels = np.concatenate((human_val_labels, mouse_val_labels))
    merged_test_seq = np.concatenate((human_test_seq, mouse_test_seq))
    merged_test_labels = np.concatenate((human_test_labels, mouse_test_labels))

    merged_train_data = one_hot_encode(merged_train_seq)
    merged_val_data = one_hot_encode(merged_val_seq)
    merged_test_data = one_hot_encode(merged_test_seq)

    merged_model = train_merged_model(merged_train_data, merged_train_labels, merged_val_data, merged_val_labels, merged_test_data, merged_test_labels)

    human_train_data = one_hot_encode(human_train_seq)
    human_val_data = one_hot_encode(human_val_seq)
    human_test_data = one_hot_encode(human_test_seq)

    train_species_model(human_train_data, human_train_labels, human_val_data, human_val_labels, human_test_data, human_test_labels, 'human', merged_model)

    mouse_train_data = one_hot_encode(mouse_train_seq)
    mouse_val_data = one_hot_encode(mouse_val_seq)
    mouse_test_data = one_hot_encode(mouse_test_seq)

    train_species_model(mouse_train_data, mouse_train_labels, mouse_val_data, mouse_val_labels, mouse_test_data, mouse_test_labels, 'mouse', merged_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TransSE model')
    parser.add_argument('--human_train', required=True, help='Path to human training data')
    parser.add_argument('--human_val', required=True, help='Path to human validation data')
    parser.add_argument('--human_test', required=True, help='Path to human test data')
    parser.add_argument('--mouse_train', required=True, help='Path to mouse training data')
    parser.add_argument('--mouse_val', required=True, help='Path to mouse validation data')
    parser.add_argument('--mouse_test', required=True, help='Path to mouse test data')

    args = parser.parse_args()
    main(args)
