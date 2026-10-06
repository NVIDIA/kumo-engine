# PKDD'99 Financial Dataset

A **subset** of the **[Finanical Test Dataset of the PKDD'99 Discovery Challenge](https://sorry.vse.cz/~berka/challenge/PAST)** in which the `trans.csv` file was shortened to 5000 lines.
This [dataset](https://relational.fit.cvut.cz/dataset/Financial) is part of the **[Relational Dataset Repository](https://relational.fit.cvut.cz)**.

<p align="center">
  <img height="900px" src="https://relational.fit.cvut.cz/assets/img/datasets-generated/financial.svg" />
</p>

## Task

A bank wants to improve their services by finding interesting groups of clients (*e.g.*, to differentiate between good and bad clients).
The bank managers have only vague idea on who is a good client (whom to offer some additional services) and who is a bad client (whom to watch carefully to minimize the bank looses).
Fortunately, the bank stores data about their clients, the accounts (transactions within several months), the loans already granted, the credit cards issued.
The standard task is to predict the loan outcome for finished loans (`"A"` vs `"B"` in `LOAN.STATUS`) at the time of the loan start (defined by `LOAN.DATE`).
Two factors have a great impact on model accuracy: (1) Was the temporal constraint respected? (2) Was the problem formulated as (`"A" = "C"` vs. `"B" = "D"`), or (`"A"` vs. `"B"` vs. `"C"` vs. `"D"`).

## Description

* `ACCOUNT.csv`: Each record describes static characteristics of an account.
* `CLIENT.csv`: Each record describes characteristics of a client.
* `DISP.csv`: Each record relates together a client with an account.
* `ORDERS.csv`: Each record describes characteristics of a payment order.
* `TRANS.csv`: Each record describes one transaction on an account.
* `LOAN.csv`: Each record describes a loan granted for a given account.
* `CARD.csv`: Each record describes a credit card issued to an account.
* `DISTRICT.csv`: Each record describes demographic characteristics of a district.
