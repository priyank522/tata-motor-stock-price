import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from KNN_model import _train_KNN
from Random_Forest import _train_random_forest
from libraries import *
import warnings
warnings.filterwarnings('always')


"""Here We defining some Constants"""

NUM_DAYS = 250    # The number of days of historical data to retrieve
INTERVAL = '1d'     # Sample rate of historical data
symbol = 'TATAMOTORS.NS'      # Symbol of tatamotors
# List of symbols for technical indicators
INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

"""Step 1: Now we pull the historical data using yfinance"""

start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )
end = datetime.datetime.today()

data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
print(data.head())

""""Step 2: Now we will clean our data"""
def _exponential_smooth(data, alpha):
    """"here ewm means exponantial weighted function
        where alpha is smoothing function for the data"""
    return data.ewm(alpha=alpha).mean()

data = _exponential_smooth(data, 0.65)

tmp1 = data.iloc[-60:]
#tmp1['close'].plot()

""""Step:3 """
def _get_indicator_data(data):

    for indicator in INDICATORS:
        """TA means technical analysis"""
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    """"here ema means exponantial moving average
        which is also known as moving average"""
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['Adj Close'])

    return data


data = _get_indicator_data(data)
live_pred_data = data.iloc[-16:-11]
print("live pred data",live_pred_data)


def _produce_prediction(data, window):

    """Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    window: number of days, or rows to look ahead to see what the price did"""

    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    return data

data = _produce_prediction(data, window=15)
del (data['close'])
data = data.dropna()  # Some indicators produce NaN values for the first few rows, we just remove them here
data.tail() #use to get last n rows

"""step 3 over"""

def cross_Validation(data):
    # Split data into equal partitions of size len_train

    num_train = 10  # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40  # Length of each train-test set

    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    ensemble_RESULTS = []

    i = 0
    while True:

        # Partition the data into chunks of size len_train every num_train days
        df = data.iloc[i * num_train: (i * num_train) + len_train]
        i += 1
        print(i * num_train, (i * num_train) + len_train)

        if len(df) < 40:
            break

        y = df['pred']
        features = [x for x in df.columns if x not in ['pred']]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7 * len(X) // 10, shuffle=False)
        warnings.filterwarnings("ignore")

        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        knn_model = _train_KNN(X_train, y_train, X_test, y_test)

        rf_prediction = rf_model.predict(X_test)
        knn_prediction = knn_model.predict(X_test)

        print('rf prediction is ', rf_prediction)
        print('knn prediction is ', knn_prediction)
        print('truth values are ', y_test.values)

        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)

        print(rf_accuracy*100, knn_accuracy*100)
        rf_RESULTS.append(rf_accuracy)
        knn_RESULTS.append(knn_accuracy)

    try:
        print('RF Accuracy = ' + str((sum(rf_RESULTS) / len(rf_RESULTS))*100)+"%")
        print('KNN Accuracy = ' + str((sum(knn_RESULTS) / len(knn_RESULTS))*100) +"%")

        live_pred_data['close'].plot()
        del (live_pred_data['close'])
        prediction = knn_model.predict(live_pred_data)
        print("pridiction for after 15days after today:",prediction)

        for i in prediction:
            if(i==0):
                print("stock will go down")
            else:
                print("stock will go up")
    except ZeroDivisionError:
        print("zero division error")

cross_Validation(data)
