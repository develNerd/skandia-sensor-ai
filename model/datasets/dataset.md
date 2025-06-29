# TempSensor Anomaly Detection Dataset

This dataset is organized for anomaly detection tasks involving temperature sensors. It includes data for training, testing, and ground truth annotations to support supervised or semi-supervised anomaly detection models.

## 📁 Folder Structure

TempSensor/  
└── sensor/  
    ├── train/          # Normal samples used for training  
    ├── test/           # Test samples (may include anomalies)  
    └── ground_truth/   # Binary labels (0 = normal, 1 = anomaly) for the test set  
└── data_extension/     # An extension of the anomalib dataset class/module. To be able to use the anomalib library, an extension of these classes is needed for the library to properly load the datasets fro training
    ├── Dataset/          # Normal samples used for training  
    ├── test/




### Folder Details:

- **train/**  
  Contains normal time series data collected from the temperature sensor. These are used to train the model under the assumption of no anomalies.

- **test/**  
  Contains time series data that may include both normal and anomalous samples. This data is used to evaluate the performance of the model.

- **ground_truth/**  
  Provides binary labels corresponding to the test set. Each label indicates whether a particular time point is anomalous (`1`) or normal (`0`).

---

## 🔍 Use Case

This dataset can be used to:
- Train models on normal behavior (e.g., autoencoders, One-Class SVMs)
- Detect anomalies in unseen test data
- Evaluate detection performance using ground truth annotations

---

## 📝 Notes

- All files are assumed to be in a time-series format (e.g., CSV or NumPy arrays).
- Ensure that the test and ground truth data align by index or timestamp.

