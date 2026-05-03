FloraFauna — Disease Detection System for Plants and Animals

FloraFauna is a machine learning–based system designed to detect diseases in plants and animals using image data. The project focuses on early identification of diseases to support agriculture, veterinary care, and environmental monitoring.

Overview

The system leverages trained models to classify diseases from input images. It aims to assist farmers, researchers, and field workers by providing quick and reliable predictions.

Objectives
Detect diseases in plants and animals using image classification
Provide a scalable and modular ML pipeline
Enable easy experimentation through notebooks
Support real-world agricultural and ecological applications
Repository Structure
FloraFauna/
│
├── app/            # Application interface (API / UI / inference scripts)
├── data/           # Dataset (raw and processed data)
├── models/         # Trained models and model-related code
├── notebooks/      # Jupyter notebooks for experimentation and training
├── tools/          # Utility scripts (preprocessing, augmentation, etc.)
├── .gitignore
└── README.md
Features
Image-based disease classification
Modular project structure for scalability
Separate pipelines for training and inference
Experiment tracking via notebooks
Reusable preprocessing and utility tools
Tech Stack
Python
TensorFlow / PyTorch (depending on implementation)
OpenCV
NumPy, Pandas
Jupyter Notebook
Workflow
Data Collection and Preprocessing
Model Training using notebooks
Model Evaluation and tuning
Saving trained models in /models
Deployment or inference via /app
Installation
# Clone the repository
git clone https://github.com/your-username/florafauna.git

# Navigate to project folder
cd florafauna

# Install dependencies
pip install -r requirements.txt
Usage
Run Training (via notebooks)
Open the notebooks/ directory
Execute training notebooks step by step
Run Inference
python app/main.py
Dataset
Contains labeled images of plant and animal diseases
Includes preprocessing and augmentation steps in /tools

Note: Dataset may need to be added manually if not included in the repository.

Future Improvements
Add real-time detection via mobile/web interface
Expand dataset for more disease classes
Integrate explainable AI (XAI) for predictions
Deploy as a cloud-based service
Add multi-language support for wider accessibility
Applications
Smart agriculture systems
Veterinary diagnostics
Environmental monitoring
Research and academic use
Contributing

Contributions are welcome. Please fork the repository and submit a pull request with improvements or new features.

License

This project is licensed under the MIT License.

Author

Malla Sai Snehaja
B.Tech Computer Science
IIIT Kottayam
