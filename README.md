# Index-Portfolio-Construction
Key Libraries Used:

`numpy`: Used for numerical operations and array manipulations.

`math`: Provides mathematical functions and constants.

`pandas`: Used for data manipulation and analysis with DataFrames.

`matplotlib`: Used for data visualization and plotting.

`gurobipy`: Interacts with gurobi solver for optimization tasks.

`yfinance`: Used to fetch historical stock data from Yahoo Finance.

```python
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import yfinance as yf
import os
import csv
```
1. Crawl the list of S&P500 companies from Wikipedia and save it as a `csv` file.
```python
if not os.path.exists("S&P500.csv"):
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500.csv')
```
2. Load the data.
```python
df = pd.read_csv("S&P500.csv")
stocks = list(df['Symbol'])

# Yahoo Finance package has some error while downloading these stocks
stocks.remove('BF.B')
stocks.remove('BRK.B')
stocks.remove('GOOGL') #there exists GOOG stock which is the same as GOOGL
```
The list comprises 501 common stocks issued by 500 large-cap companies that are traded on American stock exchanges. These stocks are included in the S&P500 index, which is a market-capitalization-weighted index of the top 500 publicly traded companies in the US. The S&P500 index is widely regarded as one of the best indicators of the overall health of the US stock market, and is commonly used as a benchmark for portfolio performance.

It is important to note that the S&P500 index is not static, and is periodically reconstituted to ensure that it accurately reflects the current state of the US stock market. As such, the list of 501 common stocks included in the index may change over time as companies are added or removed.

To perform any analysis on stock data, we need to first obtain the data itself. In this step, we will download historical stock data for the S&P500 companies using the `yfinance` package. Once we have the data, we will extract the closing prices for each stock.

To download the data, we will use the `yf.download()` function. This function takes a list of stock symbols and a start and end date, and returns a `pandas` DataFrame containing the historical stock data for the specified time period.

After downloading the data, we will extract the closing prices for each stock. The closing price is the final price at which a stock is traded on a given day, and is one of the most commonly used metrics for evaluating a stock's performance.

By obtaining and extracting the historical closing prices for the S&P500 stocks, we will be able to analyze trends and patterns in the stock market, and use this information to construct an optimized index portfolio.

Therefore, our next step is to download historical data for all S&P500 companies and extract the closing prices for each stock.
```python
cols_with_na = train_df.columns[train_df.isna().any()].tolist()
cols_with_na = [list(t) for t in cols_with_na]
for i in range(len(cols_with_na)):
    if cols_with_na[i][0] == 'Adj Close':
        string = str(cols_with_na[i][1]) 
        stocks.remove(string)
```

```python
start_date = '2017-12-31'
end_date = '2022-12-31'
train_df = yf.download(stocks, start = start_date, end = end_date)
#print(train_df.head())

start_date = '2023-01-01'
test_df = yf.download(stocks, start = '2023-01-01', end = '2023-05-04')
#print(test_df.head())
```
```python
train_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
train = train_df['Close']
#print(train.head())

test_df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
test = test_df['Close']
#print(test.head())

num_stocks = len(train.axes[1])

stocks = []
for i in train.axes[1]:
    stocks.append(i)
cov = train.cov()
```
1. **Function Signature**: `def build_model(q):`\
The function takes a single argument `q`, which represents the number of stocks to be picked from the S&P 500.

2. **Model Initialization**: `stock_mod = gp.Model("Stock")` \
A Gurobi model named "Stock" is created to formulate and solve the stock selection optimization problem.

4. **Model Parameters**: `stock_mod.Params.LogToConsole = 0`\
The LogToConsole parameter is set to `0` to suppress Gurobi's output from being printed in the console during optimization.

6. **Decision Variables**: 
  - `x`: A binary variable matrix (MVar) of size (`num_stocks`, `num_stocks`). num_stocks represents the total number of stocks in the S&P 500. It is used to model the stock selection decision for pairs of stocks. 
  - `y`: A binary variable vector of size `num_stocks`. It represents the stock selection decision for individual stocks.

7. **Objective Function**: \
The objective is to maximize the quadratic expression `Z`, which represents the expected return of the selected stocks. It is constructed by summing up the covariance values between pairs of stocks multiplied by their corresponding decision variables `x[i][j]`.

8. **Set Objective**: `stock_mod.setObjective(Z, GRB.MAXIMIZE)` \
The objective function Z is set to be maximized.

9. **Constraints**: 
  - `stock_mod.addConstr(gp.quicksum(y[i] for i in range(num_stocks)) == q):` Exactly `q` stocks should be selected from the S&P 500. 
  - `stock_mod.addConstrs((gp.quicksum(x[i]) == 1 for i in range(num_stocks))):` Each stock can only be chosen once, ensuring diversification. 
  - `stock_mod.addConstrs((x[i][j] <= y[j] for i in range(num_stocks))):` If a stock is not selected `(y[j] == 0)`, then all the pairs with that stock `(x[i][j])` should be `0`.

10. **Optimization**: `stock_mod.optimize()` \
The model is optimized to find the best combination of stocks that maximizes the expected return while satisfying the constraints.

11. **Result Extraction**: The function returns the selected stocks as a binary matrix and a binary vector. 
  - The binary matrix `x[i][j].X` represents the stock selection status (`0` or `1`) for pairs of stocks `(i, j)`.  
  - The binary vector `y[i].X` represents the stock selection status (`0` or `1`) for individual stocks.

12. **Error Handling**: The code checks if the optimization status is not `gp.GRB.OPTIMAL` (i.e., not successfully optimized) and prints "test" (though this code block is unreachable due to the return statement before it).
```python
def build_model(q):
    """
    Build a model to pick q stocks from S&P 500.
    """
    stock_mod = gp.Model("Stock")
    stock_mod.Params.LogToConsole = 0
    
    x = stock_mod.addMVar((num_stocks, num_stocks), vtype = GRB.BINARY)
    y = stock_mod.addVars(num_stocks, vtype = GRB.BINARY)
    
    Z = gp.QuadExpr()
    for i in range(num_stocks):
        for j in range(num_stocks):
            Z += cov.iloc[i][j] * x[i][j]
    
    stock_mod.setObjective(Z, GRB.MAXIMIZE)
    
    stock_mod.addConstr(gp.quicksum(y[i] for i in range(num_stocks)) == q)
    stock_mod.addConstrs((gp.quicksum(x[i]) == 1 for i in range(num_stocks)))
    
    for j in range(num_stocks):
        stock_mod.addConstrs((x[i][j] <= y[j] for i in range(num_stocks)))
        
    stock_mod.optimize()
    
    return \
        [[x[i][j].X for j in range(num_stocks)] for i in range(num_stocks)], \
        [y[i].X for i in range(num_stocks)]

    if stock_mod.Status != gp.GRB.OPTIMAL:
        print("test")
    return x, y
```
1. `get_selected_stocks(y, stock_tickers):`
  - Input: Binary vector `y` representing stock selection status, and `stock_tickers` containing ticker symbols of all stocks.
  - Output: A list of selected stock tickers based on their corresponding `y` values (stocks with `y[i] > 0`).
2. `get_represented_stocks(num_stocks, x, stock_chosen, stock_tickers):`
  - Input: `num_stocks` (total number of stocks), binary matrix `x` representing stock pair selection status, `stock_chosen` representing a chosen stock ticker, and `stock_tickers` containing ticker symbols of all stocks.
  - Output: A list of stock tickers represented by the chosen stock `stock_chosen` (stocks with `x[i][stock_tickers.index(stock_chosen)] > 0`).
3. `get_fraction(num_stocks, x, y, start_prices):`
  - Input: `num_stocks` (total number of stocks), binary matrix `x` representing stock pair selection status, binary vector `y` representing stock selection status, and `start_prices` containing the initial prices of all stocks.
  - Output: A list of fractions that represent the proportion of investment in each stock. The function calculates the investment weights based on the chosen stocks.
4. `normalized_return(df):`
  - Input: A DataFrame `df` containing historical stock prices for a particular stock.
  - Output: The normalized return (percentage) between the first and last day's price for the stock.
5. `get_returns(data, num_stocks, fractions):`
  - Input: A DataFrame data containing historical stock prices for all chosen stocks, `num_stocks` (total number of stocks), and fractions representing the investment weights in each stock.
  - Output: The total return of the portfolio calculated based on the investment fractions and the stock returns.
6. `get_variance(num_stocks, fractions, cov):`
  - Input: `num_stocks` (total number of stocks), fractions representing the investment weights in each stock, and DataFrame `cov` containing the covariance matrix of stock returns.
  - Output: The portfolio variance calculated based on the investment fractions and the covariance matrix.
```python
def get_selected_stocks(y, stock_tickers):
    """
    Return tickers of the stocks chosen for the portfolio
    """
    selected_stocks = []
    for i in range(len(y)):
        if y[i] > 0:
            selected_stocks.append(stock_tickers[i])
    return selected_stocks


def get_represented_stocks(num_stocks, x, stock_chosen, stock_tickers):
    """
    Return tickers of the stocks represented by each of the chosen stock
    """
    represented_stocks = []
    for i in range(num_stocks):
        if x[i][stock_tickers.index(stock_chosen)] > 0:
            represented_stocks.append(stock_tickers[i])
    return represented_stocks


def get_fraction(num_stocks, x, y, start_prices):
    """
    Return the fraction that we will invest in each stock
    """
    weights = [0.0] * num_stocks
    for j in range(num_stocks):
        if y[j] > 0:
            for i in range(num_stocks):
                weights[j] += start_prices[i] * int(x[i][j])
    
    return [weights[i] / sum(weights) for i in range(num_stocks)]


def normalized_return(df):
    """
    Calculate the return between the first and last day for each stock
    """
    first_day_price = df.iloc[0]
    last_day_price = df.iloc[-1]
    return (last_day_price - first_day_price) / first_day_price


def get_returns(data, num_stocks, fractions):
    """
    Calculate the total return for our index portfolio
    """
    total_return = 0.0
    returns = normalized_return(data)
        
    for i in range(num_stocks):
        total_return += fractions[i] * (1.0 + returns[i])
        
    return total_return - 1


def get_variance(num_stocks, fractions, cov):
    """
    Calculate the variance of our portfolio
    """
    var = 0
    for i in range(num_stocks):
        for j in range(num_stocks):
            var += fractions[i] * fractions[j] * cov.iloc[i][j]
    return var
```
- **Importing Libraries**: The code imports necessary libraries, including display from `IPython.display`.
- **Loop over** `q_range`: The code sets up a loop over the range of q values from 1 to 14 (inclusive) with a step size of 1 (range(1, 15, 1)).
- **Building the Optimization Model**: Inside the loop, the code calls the `build_model(q)` function to build an optimization model for each value of `q`. The function aims to select a specific number of stocks `(q)` from the S&P 500 based on historical stock data.
- **Data Processing**:
  - start_prices: The initial prices of stocks are extracted from the `train` DataFrame (presumably containing historical stock price data).
- **Portfolio Composition and Performance Evaluation**:
  - The code uses several functions (`get_selected_stocks`, `get_represented_stocks`, `get_fraction`, `get_returns`, `get_variance`) to evaluate the performance and composition of the optimized portfolio for each value of `q`.
  - It calculates the total returns, variance, and weights of each selected stock, as well as the number of stocks represented by each selected stock.
  - It constructs a DataFrame `df` to summarize the portfolio composition for each value of `q`, including the selected stocks, their weights, the clusters they represent, and the number of stocks represented.
- **Displaying the DataFrame**: The `display(df)` function is used to show the DataFrame df with portfolio composition and information for each value of `q`.
- **Returns and Variance Tracking**: The code keeps track of the returns for the training and test datasets (`returns_train` and `returns_test`) and the variance for the training dataset (`variance_train`).
- **Printing Returns**: The code prints the `returns_test` list at each iteration to check the values during the execution.

```python
from IPython.display import display

q_range = range(1, 15, 1)
returns_train = []
returns_test = []
variance_train = []


for q in q_range:
    q = min(q, 500)
    x, y = build_model(q)
    start_prices = train.iloc[0]

    selected_stocks = get_selected_stocks(y, stocks)
    rep = []
    for j in range(len(selected_stocks)):
        represented_stocks = get_represented_stocks(num_stocks, x, selected_stocks[j], stocks)
        rep.append(represented_stocks)

    weights = []
    for j in range(len(selected_stocks)):
        fractions = get_fraction(num_stocks, x, y, start_prices)[stocks.index(selected_stocks[j])]
        weights.append(fractions)

    num_rep = []
    for j in range(len(selected_stocks)):
        num = len(rep[j])
        num_rep.append(num)
    
    fractions = get_fraction(num_stocks, x, y, start_prices)
    returns_train.append(get_returns(train, num_stocks, fractions) * 100)
    returns_test.append(get_returns(test, num_stocks, fractions) * 100)
    print(returns_test) #this is printed to make sure because we had some issue running previous iterations
    variance_train.append(get_variance(num_stocks, fractions, cov))    
        
    df = pd.DataFrame({
        'Ticker': selected_stocks,
        'Weight': weights,
        'Cluster': rep,
        'Num Stocks Rep': num_rep   
    })
    display(df)
```
```
> Set parameter Username
Academic license - for non-commercial use only - expires 2024-02-04
[28.446481178117455]
Ticker	Weight	Cluster	Num Stocks Rep
0	NVR	1.0	[A, AAL, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN,...	488
[28.446481178117455, 28.65989161863185]
Ticker	Weight	Cluster	Num Stocks Rep
0	BKNG	0.12078	[AAL, ALK, APA, BA, BEN, BIIB, BK, BKR, BXP, C...	73
1	NVR	0.87922	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	415
[28.446481178117455, 28.65989161863185, 25.458899023709503]
Ticker	Weight	Cluster	Num Stocks Rep
0	AZO	0.044432	[APA, ATO, BKR, CAH, CF, CNP, COP, CTRA, CVX, ...	33
1	BA	0.084893	[AAL, ALK, BA, BEN, BIIB, BXP, C, CCL, CMA, DA...	51
2	NVR	0.870675	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	404
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835]
Ticker	Weight	Cluster	Num Stocks Rep
0	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
1	BA	0.052121	[AAL, BA, BIIB, BXP, CCL, DAL, DXC, GILD, HII,...	28
2	BKNG	0.058535	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	37
3	NVR	0.849971	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	393
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607]
Ticker	Weight	Cluster	Num Stocks Rep
0	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
1	BA	0.044376	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	26
2	BIIB	0.009660	[BIIB, GILD, INCY]	3
3	BKNG	0.058535	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	37
4	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877]
Ticker	Weight	Cluster	Num Stocks Rep
0	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
1	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
2	BIIB	0.007746	[BIIB, GILD]	2
3	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
4	MKTX	0.003819	[INCY, INTC, VZ]	3
5	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
1	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
2	BIIB	0.007746	[BIIB, GILD]	2
3	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
4	MKTX	0.002930	[INCY, VZ]	2
5	NFLX	0.000889	[INTC]	1
6	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
2	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
3	BIIB	0.007746	[BIIB, GILD]	2
4	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
5	MKTX	0.002930	[INCY, VZ]	2
6	NFLX	0.000889	[INTC]	1
7	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AAL	0.000000	[]	0
2	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
3	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
4	BIIB	0.007746	[BIIB, GILD]	2
5	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
6	MKTX	0.002930	[INCY, VZ]	2
7	NFLX	0.000889	[INTC]	1
8	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AAL	0.000000	[]	0
2	AAP	0.000000	[]	0
3	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
4	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
5	BIIB	0.007746	[BIIB, GILD]	2
6	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
7	MKTX	0.002930	[INCY, VZ]	2
8	NFLX	0.000889	[INTC]	1
9	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AAL	0.000000	[]	0
2	AAP	0.000000	[]	0
3	AAPL	0.000000	[]	0
4	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
5	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
6	BIIB	0.007746	[BIIB, GILD]	2
7	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
8	MKTX	0.002930	[INCY, VZ]	2
9	NFLX	0.000889	[INTC]	1
10	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AAL	0.000000	[]	0
2	AAP	0.000000	[]	0
3	AAPL	0.000000	[]	0
4	ABBV	0.000000	[]	0
5	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
6	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
7	BIIB	0.007746	[BIIB, GILD]	2
8	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
9	MKTX	0.002930	[INCY, VZ]	2
10	NFLX	0.000889	[INTC]	1
11	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AAL	0.000000	[]	0
2	AAP	0.000000	[]	0
3	AAPL	0.000000	[]	0
4	ABBV	0.000000	[]	0
5	ABC	0.000000	[]	0
6	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
7	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
8	BIIB	0.007746	[BIIB, GILD]	2
9	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
10	MKTX	0.002930	[INCY, VZ]	2
11	NFLX	0.000889	[INTC]	1
12	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
[28.446481178117455, 28.65989161863185, 25.458899023709503, 26.494267554143835, 26.55380938811607, 26.515486324323877, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583, 26.51855278281583]
Ticker	Weight	Cluster	Num Stocks Rep
0	A	0.000000	[]	0
1	AAL	0.000000	[]	0
2	AAP	0.000000	[]	0
3	AAPL	0.000000	[]	0
4	ABBV	0.000000	[]	0
5	ABC	0.000000	[]	0
6	ABT	0.000000	[]	0
7	AZO	0.039373	[APA, ATO, CAH, CF, CNP, COP, CTRA, CVX, DVN, ...	30
8	BA	0.043360	[AAL, BA, BXP, CCL, DAL, DXC, HII, KMI, LVS, N...	25
9	BIIB	0.007746	[BIIB, GILD]	2
10	BKNG	0.057646	[ALK, BEN, BK, BKR, C, CMA, DD, DISH, EIX, FRT...	36
11	MKTX	0.002930	[INCY, VZ]	2
12	NFLX	0.000889	[INTC]	1
13	NVR	0.848057	[A, AAP, AAPL, ABBV, ABC, ABT, ACGL, ACN, ADBE...	392
```
Based on the context provided, it seems that the table being referred to is the DataFrame df that summarizes the composition of the optimized portfolio for each value of `q`. By exporting this table, we can analyze the clusters of stocks that remain constant for different values of `q`. This information can be useful in determining the stability and diversification of the optimized portfolio.

Therefore, we can export the table to obtain a more detailed view of the data and better understand the patterns that emerge. By examining the clusters of stocks that remain constant, we can identify any potential issues with the portfolio's diversity and adjust the optimization accordingly. This can ultimately lead to a better performing portfolio and more effective investment strategy.
```python
df = pd.DataFrame({
    'Num. Stocks': q_range,
    'Train Set Return': returns_train,
    'Test Set Return': returns_test,
    'Train Set Variance': variance_train
})
df = df.reset_index(drop = True)
df
```
To gain a better understanding of the performance of your portfolio, it is important to compare its return with a benchmark index, such as the S&P500. This allows you to assess whether your portfolio is outperforming or underperforming the broader market. In order to make an accurate comparison, it is important to consider factors such as the time period being analyzed, the asset allocation of your portfolio, and any fees or expenses associated with your investments. By conducting a thorough analysis of your portfolio's performance relative to the S&P500, you can make more informed decisions about your investment strategy going forward.
```python
sp500_train = (sum(train.iloc[:, -1])/sum(train.iloc[:, 0]) - 1) * 100
print('Return of SP500 on the training period:', round(sp500_train, 3),'%')
sp500_test = (sum(test.iloc[:, -1])/sum(test.iloc[:, 0]) - 1) * 100
print('Return of SP500 on the testing period:', round(sp500_test, 3),'%')
```
Plot the diagram to check the return
```python
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20, 24))

ax1.plot(q_range, returns_train)
ax1.set_xlabel('Number of Stocks in Portfolio', fontsize = 15)
ax1.set_ylabel('Return (%)', fontsize = 15)
ax1.set_title('Return of Portfolio on the Training period vs q', fontsize = 15)
ax1.set_ylim(27, 37)

ax2.plot(q_range, returns_test)
ax2.set_xlabel('Number of Stocks in Portfolio', fontsize = 15)
ax2.set_ylabel('Return (%)', fontsize = 15)
ax2.set_title('Return of Portfolio on the Testing period vs q', fontsize = 15)
ax2.set_ylim(23, 30)

ax3.plot(q_range, variance_train)
ax3.set_xlabel('Number of Stocks in Portfolio', fontsize = 15)
ax3.set_ylabel('Variance', fontsize = 15)
ax3.set_title('Variance vs q', fontsize = 15)

plt.show()
```
![Screenshot](var.avif)
```python
#Schwab 500 Index Fund

SP_ticker = "SWPPX"

SP_train_df = yf.download(SP_ticker, start = '2017-12-31', end = '2022-12-31')
SP_train_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
SP_train = SP_train_df['Close']

SP_test_df = yf.download(SP_ticker, start = '2023-01-01')
SP_test_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
SP_test = SP_test_df['Close']

print('Return of SWPPX on the training period:',round(normalized_return(SP_train) * 100, 3),'%')
print('Return of SWPPX on the testing period:',round(normalized_return(SP_test) * 100, 3), '%')
```
```
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
Return of SWPPX on the training period: 41.488 %
Return of SWPPX on the testing period: 6.766 %
```
```python
#Fidelity 500 Index Fund

SP_ticker = "FXAIX"

SP_train_df = yf.download(SP_ticker, start = '2017-12-31', end = '2022-12-31')
SP_train_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
SP_train = SP_train_df['Close']

SP_test_df = yf.download(SP_ticker, start = '2023-01-01')
SP_test_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
SP_test = SP_test_df['Close']

print('Return of FXAIX on the training period:',round(normalized_return(SP_train) * 100,3),'%')
print('Return of SWPPX on the testing period:',round(normalized_return(SP_test) * 100,3), '%')
```
```
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
Return of FXAIX on the training period: 41.271 %
Return of SWPPX on the testing period: 6.373 %
```
```python
#Vanguard 500 Index Fund

SP_ticker = "VFIAX"

SP_train_df = yf.download(SP_ticker, start = '2017-12-31', end = '2022-12-31')
SP_train_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
SP_train = SP_train_df['Close']

SP_test_df = yf.download(SP_ticker, start = '2023-01-01')
SP_test_df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index(level = 1)
SP_test = SP_test_df['Close']

print('Return of VFIAX on the training period:',round(normalized_return(SP_train) * 100,3),'%')
print('Return of VFIAX on the testing period:',round(normalized_return(SP_test) * 100,3),'%')
```
```
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
Return of VFIAX on the training period: 42.281 %
Return of VFIAX on the testing period: 6.325 %
```
