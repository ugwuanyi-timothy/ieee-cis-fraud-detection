IEEE-CIS Fraud Detection System



Project Overview



An end-to-end machine learning application that predicts whether a financial transaction is fraudulent or legitimate.



The system uses a trained XGBoost classifier and a FastAPI backend, with a web-based frontend that allows users to submit transaction data and view prediction results.



Key Features



\- Fraud detection using XGBoost.

\- Transaction preprocessing and feature engineering.

\- Frequency encoding and aggregation mappings.

\- REST API built with FastAPI.

\- Interactive web frontend.

\- Fraud probability estimation.

\- Legitimate and fraudulent transaction classification.



Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- FastAPI

\- Uvicorn

\- HTML

\- CSS

\- JavaScript



Project Structure



The project contains the trained model, preprocessing resources, API application, frontend, and dependency configuration.



Running the Application



1\. Clone the repository.



2\. Install the dependencies:

&#x20;  

&#x20;  pip install -r requirements.txt



3\. Start the API:

&#x20;  

&#x20;  python -m uvicorn main:app --host 127.0.0.1 --port 8014



4\. Open the application in your browser:

&#x20;  

&#x20;  "http://127.0.0.1:8014/"



5\. Upload a transaction JSON file to receive a prediction.



API Documentation



FastAPI provides interactive API documentation at:



"http://127.0.0.1:8014/docs"



Prediction Output



The application returns:



\- Fraud probability

\- Prediction label

\- Transaction classification



Example:



{

&#x20; "fraud percentage": 0.6763,

&#x20; "prediction": 1,

&#x20; "result": "fraudulent"

}



Dataset



The project is based on the IEEE-CIS Fraud Detection dataset from Kaggle.



Disclaimer



This project is intended for educational and demonstration purposes. Predictions should not be treated as the sole basis for real financial decisions.

