# tata-motor-stock-prediction 

This is python and datascience project to predict stock value which will give predict stock go up or own after 15 days from today.

## How to use :

* Clone this repository to your system.
* Run the main.py file In Your device.
* Get output about whether the stock will go up or down after 15 days.

## About the dataset:

* Here in our project we import yfinanace module to get dataset of particular stock.

### yfinanace:

  * The yfinance is one of the famous modules in Python, which is used to collect online data, and with it, we can collect the financial data of Yahoo. With the help of the yfinance module, we retrieve and collect the company's financial information (such as financial ratios, etc.)
  
### How to import and use yfinanace:

```
import yfinance as yf

#downloading data from yfinanace
data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
```

## used libraries:
* numpy
* pandas
* matplotlib
* datetime
* finta
* sklearn

## Process On data:

1. get data from yfinance library.
2. use exponential smoot function to smooth our data.
```
def _exponential_smooth(data, alpha):
   return data.ewm(alpha=alpha).mean()
```
3. put technical analysis data and remove unwanted data.
4. create boundary on data to make analysis on that.
5. Run on different model.
6. got the desire output. 

## Model Used by us:
* Random Forest Algorith.
* K -nearest neighbor Algorithm.


## Credits:

#### Patel Priyank  :
* work on data cleaning process, random forest module.
* Contributed in README and PPT.

#### Mitali Mistry :
* work on data cleaning, data visulization and KNN model.
* Contributed in README and PPT. 
### vivek prajapati 
 * work on data cleaning and error sloving.
 * conributed README. 
