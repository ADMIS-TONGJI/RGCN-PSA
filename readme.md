# Code for RGCN-PSA

**Paper**: Cross-Region Graph Convolutional Network with Periodicity Shift Adaptation for Wide-area SST Prediction

**Authors**:  
Han Peng, Wengen Li*, Chang Jin, Yichao Zhang, Jihong Guan*, Hanchen Yang, and Shuigeng Zhou (*Corresponding authors)

**Affiliation**:  
Department of Computer Science and Technology, Tongji University, China; Department of Computing, The Hong Kong Polytechnic University, China; Shanghai Key Lab of Intelligent Information Processing and School of Computer Science, Fudan University, China

---

## Table of Contents
- [Code for RGCN-PSA](#code-for-rgcn-psa)
  - [Table of Contents](#table-of-contents)
    - [Quick Start](#quick-start)
    - [Dependencies](#dependencies)
    - [Data Preparation](#data-preparation)
    - [Code Structure](#code-structure)
    - [Citation](#citation)

---

### Quick Start
```bash
# Clone the repository
git clone https://github.com/ADMIS-TONGJI/RGCN-PSA.git

# Install dependencies
pip install -r requirements.txt

# Run North Pacific dataset
python main_NP.py

# Run South Atlantic dataset
python main_SA.py
```

---

### Dependencies
- Python 3.8.19
- PyTorch 2.3.1
- Key libraries: NumPy, SciPy, OpenCV, etc.  
See full requirements in [requirements.txt](./requirements.txt).

---

### Data Preparation

- **Public Data (Raw SST Data)**  
  Download sea surface temperature (SST) data from NOAA's Optimum Interpolation SST dataset:  
  🔗 [https://www.ncei.noaa.gov/products/optimum-interpolation-sst](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)  

- **Preprocessed Data (Ready-to-Use)**  
  Download our processed datasets from Google Drive:  
  🔗 [https://drive.google.com/drive/folders/1RO91kEj2geNtqnF9KtPRODsNjoR9Qus_](https://drive.google.com/drive/folders/1RO91kEj2geNtqnF9KtPRODsNjoR9Qus_)  

---

### Code Structure
```
.
├── RGCN_PSA_NORTH_PACIFIC_1.conf
├── RGCN_PSA_SOUTH_ATLANTIC_1.conf
├── data
│   ├── NORTH_PACIFIC_1
│   │   └── NORTH_PACIFIC_1.npz
│   └── SOUTH_ATLANTIC_1
│       └── SOUTH_ATLANTIC_1.npz
├── lib
│   ├── dataloader.py
│   ├── logger.py
│   └── metrics.py
├── main_NP.py
├── main_SA.py
├── model
│   ├── GCN.py
│   ├── GCRAN_TS.py
│   ├── RAA.py
│   ├── SelfAttention.py
│   ├── TSAttention.py
│   ├── TimeEncoder.py
│   └── train.py
├── readme.md
└── requirements.txt
```

### Citation
If this code aids your research, please cite our paper:  
```bibtex
 coming soon
```
