# Home-credit-default-risk
This project is an end-to-end MLOps pipeline for building and deploying a credit scoring model. The model predicts the probability of default for loan applicants using data from the Home Credit dataset.

## Datset description
There are 8 tables of interest in total. Let’s take a look at each of those tables below. These descriptions have been provided by the Home Credit Group.

application_{train|test}.csv

This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
Static data for all applications. One row represents one loan in our data sample.
bureau.csv

All client’s previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
For every loan in our sample, there are as many rows as the number of credits the client had in the Credit Bureau before the application date.
bureau_balance.csv

Monthly balances of previous credits in Credit Bureau.
This table has one row for each month of history of every previous credit reported to Credit Bureau — i.e. the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
POS_CASH_balance.csv

Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample — i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
credit_card_balance.csv

Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample — i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
previous_application.csv

All previous applications for Home Credit loans of clients who have loans in our sample.
There is one row for each previous application related to loans in our data sample.
installments_payments.csv

Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
There is a) one row for every payment that was made plus b) one row each for a missed payment.
One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
Source: Home Credit Group (Kaggle)

## Project Organisation
```app_scoring/```

Contains the dashboard code made with streamlit and deploy in streamlit cloud

```notebooks``` 

Contains notebooks for data exploration, prepocessing, pycaret, shap, hyperparameters tuning feature selection, modeling and traking

```settings/```

Contains parameters for the project

```src/```

Contains source code for the API, and modules used in our code such as loading datasets, utility functions, and modules for preprocessing.

## Installation

### 1. Clone the repository
```
   git clone https://github.com/yourusername/home-credit-default-risk.git
    cd home-credit-default-risk  
```

### 2. Create a virtual environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### 3. Install the dependencies
``` bash
pip install -r requirements.txt

```
### 4. Running mlflow on localhost port 5000
   ```bash
   python -m mlflow ui
   ```
### 5. Running the API
```bash
python src/API/app.py
```
### 6. Running the dashboard

```bash
python app_scoring/app.py
```
### 7. Running the notebooks with papemill

```bash
./run_scoring.sh

```



