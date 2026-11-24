IEEE-CIS Fraud Detection System



An end-to-end machine learning application for detecting potentially

&#x20;

fraudulent financial transactions using XGBoost, FastAPI, and a



web-based frontend.



Project Overview



This project uses a trained XGBoost classifier to estimate the probability



&#x20;that a financial transaction is fraudulent.



The application provides a web interface where users can submit transaction



&#x20;data and receive a prediction, fraud probability, and transaction classification.



Application Features



\- Fraud detection using a trained XGBoost model.

\- Transaction preprocessing and feature engineering.

\- Frequency encoding and aggregation-based features.

\- REST API developed with FastAPI.

\- Interactive web frontend.

\- Fraud probability estimation.

\- Fraudulent and legitimate transaction classification.

\- Input validation and error handling.



Application screenshots



Home page



!\[Home page](screenshots/home.png)



Legitimate Transaction



!\[Legitimate Transaction](screenshots/legitimate-transaction.png)



Fraud Transaction



!\[Fraudulent Transaction](screenshots/fraudulent-transaction.png)



Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- FastAPI

\- Uvicorn

\- Joblib

\- HTML

\- CSS

\- JavaScript



Project Structure



ieee-cis-fraud-detection/

│

├── frontend/

│ └── index.html

│

├── tests/

│ └── test\_api.py

│

├── main.py

├── preprocessing.py

├── requirements.txt

├── README.md

│

├── xgb\_model.pkl

├── freq\_maps.pkl

├── agg\_maps.pkl

├── encoders.pkl

├── feature\_colum.pkl

└── feature\_dtype.pkl



Installation and Setup



1\. Clone the repository



git clone https://github.com/ugwuanyi-timothy/ieee-cis-fraud-detection.git



2\. Navigate into the project directory



cd ieee-cis-fraud-detection



3\. Create and activate a virtual environment



python -m venv venv



Windows:



venv\\Scripts\\activate



4\. Install dependencies



pip install -r requirements.txt



Running the Application



Start the FastAPI server:



python -m uvicorn main:app --host 127.0.0.1 --port 8014



Open the application in your browser:



http://127.0.0.1:8014/



API Documentation



Interactive API documentation is available at:



http://127.0.0.1:8014/docs



Prediction Endpoint



POST "/predict"



The endpoint accepts transaction data in JSON format and returns a prediction.



Example Response



{

&#x20;   "fraud\_probability": 0.6763,

&#x20;   "prediction": 1,

&#x20;   "result": "fraudulent"

}



Prediction Labels



Prediction| Meaning

0| Legitimate transaction

1| Fraudulent transaction



The fraud probability represents the model's estimated probability of fraud,



&#x20;not a guarantee that a transaction is fraudulent.



Dataset



This project is based on the "IEEE-CIS Fraud Detection dataset"



(https://www.kaggle.com/competitions/ieee-fraud-detection) hosted on Kaggle.



The original dataset is not included in this repository.



Model Evaluation



Model performance was evaluated during the Kaggle competition using the ROC-AUC metric.



Detailed evaluation results and methodology will be documented alongside



&#x20;the model evaluation materials.



Disclaimer



This project is intended for educational, research, and demonstration purposes.



&#x20;Predictions should not be used as the sole basis for real-world financial decisions.

