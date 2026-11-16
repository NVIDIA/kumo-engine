# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.data.utils import get_columns_from_query
from kumo.testing import withPackage

sql_queries = [
    ("SELECT DISTRICT_ID, AMOUNT FROM"
     " ACCOUNT INNER JOIN TRANS ON"
     " ACCOUNT.ACCOUNT_ID = TRANS.ACCOUNT_ID"),
    ("SELECT * FROM foo"),
    ("SELECT bar FROM foo WHERE foo.bar = 1"),
    ("SELECT month, sales, SUM(sales) OVER (ORDER BY month) "
     "AS cumulative_sales FROM sales_data"),
    ("SELECT COUNT(CustomerID), Country FROM Customers "
     "GROUP BY Country ORDER BY COUNT(CustomerID) DESC"),
    ("SELECT ID, Name, Age FROM Employee "
     "WHERE Name IN (SELECT Name FROM Manager "
     "INNER JOIN Award ON Manager.ID = Award.Employee_ID)"),
    ("WITH employee_ranking AS (SELECT employee_id, salary, "
     "RANK() OVER (ORDER BY salary DESC) as ranking FROM employee) "
     "SELECT employee_id, salary FROM employee_ranking "
     "WHERE ranking <= 5 ORDER BY ranking"),
    ("WITH employee_ranking AS (SELECT employee_id, salary, "
     "RANK() OVER (ORDER BY salary DESC) as ranking FROM employee), "
     "manager_ranking AS (SELECT employee_id, salary FROM manager) "
     "SELECT employee_id, salary FROM "
     "employee_ranking JOIN manager_ranking ON "
     "employee_ranking.employee_id = manager_ranking.employee_id"),
]
expected_columns = [
    {'DISTRICT_ID', 'AMOUNT', 'TRANS.ACCOUNT_ID', 'ACCOUNT.ACCOUNT_ID'},
    {'*'},
    {'bar', 'foo.bar'},
    {'month', 'sales'},
    {'CustomerID', 'Country'},
    {'ID', 'Name', 'Age', 'Manager.ID', 'Award.Employee_ID'},
    {'employee_id', 'salary'},
    {'employee_id', 'salary'},
]


@withPackage('sql_metadata')
def test_get_columns_from_query():
    for sql_query, column in zip(sql_queries, expected_columns):
        assert set(get_columns_from_query(sql_query)) == column
