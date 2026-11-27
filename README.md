# IEEE-CIS Fraud Detection System



An end-to-end machine learning application for detecting potentially fraudulent financial transactions using **XGBoost**, **FastAPI**, and a web-based frontend.



## Live Application



- **Live Demo:** https://ieee-cis-fraud-detection-1.onrender.com

- **GitHub Repository:** https://github.com/ugwuanyi-timothy/ieee-cis-fraud-detection



---



## Project Overview



Financial fraud detection is a challenging classification problem. Fraudulent transactions are relatively uncommon, and identifying suspicious patterns requires effective data preprocessing, feature engineering, and machine learning techniques.



For this project, I developed a machine learning solution using the IEEE-CIS Fraud Detection dataset from Kaggle.



The goal was not just to train a model, but to build an application that can accept transaction data, process it using the same preprocessing pipeline used during model development, and return a prediction.



The final application uses a trained **XGBoost classifier** to estimate the probability that a transaction is fraudulent and classify it as either potentially fraudulent or legitimate.



---



## Application Screenshots



### 1. Home Page



!\[Application Home Page](screenshots/home.png)



### 2. Legitimate Transaction



!\[Legitimate Transaction](screenshots/legitimate-transaction.png)



### 3. Fraudulent Transaction



!\[Fraudulent Transaction](screenshots/fraudulent-transaction.png)



---



## Key Features



- Machine learning-based fraud detection using XGBoost.

- Fraud probability estimation.

- Classification of transactions as legitimate or potentially fraudulent.

- FastAPI backend for handling prediction requests.

- Web-based frontend for submitting transaction data.

- Feature engineering and categorical encoding.

- Optimized aggregation mappings for deployment.

- CPU-only XGBoost deployment to reduce unnecessary memory requirements.

- Automated API testing using pytest.



---



## Model Performance



The model was evaluated using the Kaggle IEEE-CIS Fraud Detection competition leaderboard.



| Evaluation | ROC-AUC Score |

|:---|---:|

| Public Leaderboard | 0.9406 |

| Private Leaderboard | 0.9225 |

| Local score | 0.9290 |



The private leaderboard score provides an indication of how the model performed on the competition's hidden evaluation data.



---



## Feature Engineering and Preprocessing



During experimentation, I worked on feature engineering, categorical encoding, aggregation features, and model configuration to improve the model's predictive performance.



The preprocessing pipeline includes:



- Frequency encoding of selected categorical features.

- Aggregation features based on transaction-related identifiers.

- Combination features involving card, address, and email information.

- Label encoding of categorical variables.

- Alignment of processed data with the model's expected feature columns.



The final model expects **357 input features**.



### Aggregation Map Optimization



One of the engineering challenges was reducing the memory and storage overhead associated with the aggregation mappings used during preprocessing.



The original aggregation mappings were converted into a compact representation while preserving their aggregation values.



| Version | File Size |

|:---|---:|

| Original aggregation mappings | 42.27 MiB |

| Compact aggregation mappings | 3.03 MiB |



This reduced the serialized aggregation artifact by approximately **92.8%**.



The compact mappings were compared against the original mappings to verify that they produced identical aggregation values.



### Deployment Optimization



I also switched to the CPU-only XGBoost package for deployment to avoid unnecessary GPU-related dependencies and reduce installation overhead.



This helped make the application more suitable for deployment in a memory-constrained hosting environment.



---



## Technologies Used



| Technology | Purpose |

|:---|:---|

| Python | Main programming language |

| Pandas | Data manipulation |

| NumPy | Numerical operations |

| Scikit-learn | Data preprocessing |

| XGBoost | Machine learning model |

| FastAPI | Backend API |

| Uvicorn | ASGI server |

| HTML/CSS/JavaScript | Frontend development |

| Pytest | Automated testing |

| Git and GitHub | Version control |

| Render | Application deployment |



---



## Project Structure



```text

ieee-cis-fraud-detection/

│

├── main.py

├── preprocessing.py

│

├── xgb_model.pkl

├── freq_maps.pkl

├── compact_agg_maps.pkl

├── encoders.pkl

├── feature_colum.pkl

├── feature_dtype.pkl

│

├── frontend/

│ └── index.html

│

├── tests/

│ └── test_api.py

│

├── screenshots/

│ ├── home.png

│ ├── legitimate-transaction.png

│ └── fraudulent-transaction.png

│

├── requirements.txt

├── requirements-dev.txt

├── .python-version

├── .gitignore

└── README.md

```



---



## Running the Application Locally



### 1. Clone the Repository



```bash

git clone https://github.com/ugwuanyi-timothy/ieee-cis-fraud-detection.git

```



### 2. Navigate into the Project Directory



```bash

cd ieee-cis-fraud-detection

```



### 3. Create a Virtual Environment



```bash

python -m venv venv

```



### 4. Activate the Virtual Environment



On Windows:



```bash

venv\\Scripts\\activate

```



### 5. Install Dependencies



```bash

python -m pip install -r requirements.txt

```



### 6. Start the Application



```bash

uvicorn main:app --reload

```



Open the following address in your browser:



```text

http://127.0.0.1:8000

```



The API documentation is available at:



```text

http://127.0.0.1:8000/docs

```



---



## API Usage



The application exposes a prediction endpoint:



**Endpoint:** `POST /predict`



The endpoint accepts transaction data and returns a fraud prediction.



### Example Response



```json

{

&#x20;   "fraud_probability": 0.6763,

&#x20;   "prediction": 1,

&#x20;   "result": "fraudulent"

}

```



### Prediction Interpretation



| Prediction | Meaning |

|:---:|:---|

| 0 | Legitimate transaction |

| 1 | Potentially fraudulent transaction |



The fraud probability represents the model's estimated probability of fraud. It should not be interpreted as a definitive determination that a transaction is fraudulent.



---



## Testing



Automated tests were implemented using pytest to verify the API's behaviour.



### Install Development Dependencies



```bash

python -m pip install -r requirements-dev.txt

```



### Run the Tests



```bash

pytest -v

```



**Test Result: 5 tests passed.**



The automated API tests use mocked model outputs to verify API behaviour. They are not a substitute for evaluating the model on real transaction data.



The application was also manually tested using legitimate and fraudulent transaction examples. The deployed application returned results consistent with the local application.



---



## Dataset



The project uses the IEEE-CIS Fraud Detection dataset.



**Dataset Source:**



https://www.kaggle.com/competitions/ieee-fraud-detection



The dataset contains transaction and identity information used to develop machine learning models for detecting fraudulent financial transactions.



---



## Limitations and Future Improvements



Potential areas for further development include:



- Integration with a real-time transaction processing system.

- Monitoring model performance after deployment.

- Improving model explainability.

- Adding more comprehensive input validation.

- Exploring additional fraud detection techniques.

- Implementing authentication and access control.



---



## Disclaimer



This project was developed for educational and portfolio demonstration purposes.



The predictions are generated by a machine learning model and should not be used as the sole basis for real-world financial decisions.