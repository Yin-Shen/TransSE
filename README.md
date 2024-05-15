# TransSE: Transfer Learning-based Super Enhancer Prediction

## Introduction
TransSE is a deep learning model for predicting super enhancers from DNA sequences using transfer learning. The model is trained on both human and mouse datasets to learn generalizable features and patterns conserved across species. TransSE demonstrates strong cross-species transferability and outperforms existing methods in identifying super enhancers.
Key features of TransSE:
* Utilizes a convolutional neural network architecture to capture sequence patterns
* Employs transfer learning to leverage knowledge from human and mouse datasets
* Achieves high predictive performance on both human and mouse enhancers
* Exhibits robust cross-species transferability
* Enables motif analysis and SNP prioritization in super enhancers

## Steps to Install and Run TransSE
### 1. Clone the TransSE repository:
```
git clone https://github.com/Yin-Shen/TransSE.git
cd TransSE
```
### 2. Install the required dependencies:
```
pip install -r requirements.txt
```
### 3. Prepare your input data:
Place your human training, validation, and test data in the data/human/ directory
Place your mouse training, validation, and test data in the data/mouse/ directory

### 4. Run TransSE:
```
python transse.py --train_data data/human/train.txt data/mouse/train.txt \
                  --val_data data/human/val.txt data/mouse/val.txt \
                  --test_data data/human/test.txt data/mouse/test.txt \
                  --output_dir output/
```
### 5. Evaluate the trained model:
```
python evaluate.py --model_path output/transse_model.h5 \
                   --test_data data/human/test.txt data/mouse/test.txt
```

### For more details on the TransSE model architecture, training process, and evaluation metrics, please refer to the original publication.
