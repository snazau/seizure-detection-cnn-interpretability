## 📝 An Interpretability Framework for Convolutional Neural Network-Based Electroencephalography Analysis Discovers New Spatial and Spectral Epileptic Biomarkers

### 📌 Overview

Automated seizure detection using CNNs often suffers from a lack of transparency.
This framework addresses the "black-box" nature of these models by providing an interpretability method that uncovers meaningful EEG patterns.
The proposed method is based on 2 key elements:
* Frequency range importance: Uses Grad-CAM on Wavelet-transformed data to assign an importance score to a frequency range
* Spatial region importance: Employs an occlusion-based method to quantify the contribution of specific channels (regions) by replacing channel activity with baseline signal.

This repository provides a demo example of using the proposed interpretation method.

![](assets/figures/general_pipeline.png)

### 🚀 Getting Started

#### Installation
```bash
git clone https://github.com/snazau/seizure-detection-cnn-interpretability.git
cd seizure-detection-cnn-interpretability
pip install -r requirements.txt
```

#### Quick Start

1. [Download](https://drive.google.com/drive/folders/1wUE1LQ1RtL1IfPJCmI_CqwJwbU0qrCZ3?usp=sharing) files with sample data and a checkpoint of the trained model.
2. Run the demo
```bash
python main.py
```
3. In the `assets/visualization`, you should find an extended version of local interpretation (Figure 4 in the paper)
![](assets/figures/local_interpretation_example_Blues.png)

### 📁 Repository Structure

```
seizure-detection-cnn-interpretability
│   gradcam.py                  # Class that implements Grad-CAM
│   main.py                     # Contains small interpretation example and core logic of the proposed method
│   README.md                   # README file
│   requirements.txt            # List of dependencies
│   resnet.py                   # ResNet-18 architecture adapted to work with data in time-frequency domain after CWT 
│   utils.py                    # Auxiliary functions for data processing
│   visualization.py            # Code for creation of 5×5 heatmap matrices and building final interpretation result 
├───assets                      # Resources required for the demo run
│   ├───checkpoints             # Checkpoint of a trained model
│   ├───eeg                     # Sample of EEG data
│   ├───figures                 # Illustrations
│   ├───prediction_examples     # Sample of a prediction
│   └───visualizations          # The folder where the visualization results will be saved
```

### 🗪 Comments

This section links the mathematical formulations presented in the paper with their implementations in this repository.

#### Channel importance $CI_j$ (Eq. 7)
Channel Importance is used for region importance $RI^R$ estimation. It is measured as the change in the model’s prediction when the signal from the channel of interest is replaced by the baseline signal.
The implementation for this is located in the `main.py` (Lines 90-109). The code iterates one-by-one through all channels, performs the occlusion using a baseline signal, and calculates the resulting impact on the model's output

#### Time-frequency importance map $FTI$ (Eq. 12)
To identify important frequency ranges, we compute a time-frequency importance map ($FTI$) using Grad-CAM for each 10-sec segment.
The implementation for this is located in `main.py` (Lines 77-86). This block gets a time-frequency map for each segment, which later will be used to estimate the importance of the individual frequencies (Lines 111-113)

#### Frequency-spatial importance (Eq. 14)
Frequency-spatial importance score $FRI_{[f_0, f_1]}^R$ is defined as the dot product of region importance $RI^R$ and frequency range importance $FI_{[f_0, f_1]}$.
Considering 5 frequency bands and 5 regions, this generates the 5×5 heatmap used for global and local interpretation.
The implementation for this is located in `visualization.py` (Lines 19-35). The code calculates frequency range importance $FI_{[f_0, f_1]}$, dot product, and produces the final interpretability matrix.