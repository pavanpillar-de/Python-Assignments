import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# DATA LOADING
# =====================================================
"""Loads application_train.csv """

application_train = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\application_train.csv", encoding="latin1")
   


bureau = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\bureau.csv")
bureau_balance = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\bureau_balance.csv")
pos_cash_balance = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\POS_CASH_balance.csv")
#previous_application = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\previous_application.csv")
#credit_card_balance = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\credit_card_balance.csv")
#installments_payments = pd.read_csv(r"C:\Users\pavan\Downloads\dashboard\installments_payments.csv")
