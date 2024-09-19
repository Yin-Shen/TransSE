# TransSE: Transfer Learning-based Super Enhancer Prediction

## Introduction
TransSE is an innovative deep learning framework designed for predicting super enhancers (SEs) and typical enhancers (TEs) from DNA sequences using transfer learning. By leveraging data from both human and mouse genomes, TransSE learns generalizable features and patterns conserved across species, demonstrating robust cross-species transferability and improved performance in SE identification.
## Key Features

- **Advanced Architecture**: Utilizes a convolutional neural network to capture complex sequence patterns
- **Transfer Learning**: Leverages knowledge from human and mouse datasets to SE prediction accuracy
- **Cross-Species Applicability**: Achieves high predictive performance on both human and mouse SEs
- **Robust Transferability**: Exhibits strong performance when applied across species
- **Integrated Analysis**: Enables motif enrichment analysis and SNP prioritization within predicted SEs
- **User-Friendly Interface**: Offers a web server for accessible SE and TE predictions

## Steps to Install and Run TransSE
### 1. Clone the TransSE repository:
```
git clone https://github.com/Yin-Shen/TransSE.git
cd TransSE
```
### 2. Install the required dependencies:
```
Deep Learning
tensorflow>=2.4.0
keras>=2.4.0

Data Processing and Analysis
pandas>=1.2.0
numpy>=1.19.0
scipy>=1.5.0

Bioinformatics
biopython>=1.78

Data Visualization
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=4.14.0

Machine Learning Utilities
scikit-learn>=0.24.0

Jupyter Notebook (for development and demonstration)
jupyter>=1.0.0

Other Utilities
tqdm>=4.50.0

```
### 3. Prepare your input data:
Place your human training, validation, and test data in the data/human/ directory
Place your mouse training, validation, and test data in the data/mouse/ directory

### 4. Run TransSE:
```
python transse.py --human_train data/human_train.txt --human_val data/human_val.txt --human_test data/human_test.txt --mouse_train data/mouse_train.txt --mouse_val data/mouse_val.txt --mouse_test data/mouse_test.txt
```
### 5. Evaluate the trained model:
```
python evaluate.py 
```

### For more details on the TransSE model architecture, training process, and evaluation metrics, please refer to the original publication.
