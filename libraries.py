import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
""""from where out datasets are comming"""
import yfinance as yf
"""finta is use to finanacial technical analysis
    where TA means technical analysis"""
from finta import TA
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report