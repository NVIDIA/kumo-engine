# Home Credit Default Risk Dataset

A **subset** of the **[Home Credit Default Risk Dataset](https://www.kaggle.com/c/home-credit-default-risk)** in which each table is shortened to 200 lines.

<p align="center">
  <img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png" />
</p>

## Task

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.
[Home Credit](https://www.homecredit.net) strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

## Description

* `APPLICATION.csv` *(main table)*: Static data for all applications. One row represents one loan in our data sample.
* `BUREAU.csv`: All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample). For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
* `BUREAU_BALANCE.csv`: Monthly balances of previous credits in Credit Bureau. This table has one row for each month of history of every previous credit reported to Credit Bureau – *i.e.* the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
* `POS_CASH_BALANCE.csv`: Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – *i.e.* the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
* `CREDIT_CARD_BALANCE.csv`: Monthly balance snapshots of previous credit cards that the applicant has with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
* `PREVIOUS_APPLICATION.csv`: All previous applications for Home Credit loans of clients who have loans in our sample. There is one row for each previous application related to loans in our data sample.
* `INSTALLMENTS_PAYMENTS.csv`: Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample. There is a) one row for every payment that was made plus b) one row each for missed payment. One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
