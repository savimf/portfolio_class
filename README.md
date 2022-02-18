For a more thourough explanation, see the demonstration notebook in this repository.

# 1. Portfolio Class

## 1.1. Objective
Automate the calculation of quantitative metrics and their comparison between several portfolios. Works in paralell with the `Equity` class and the `quant_tools.py` module.

## 1.2. Attributes and Properties
### 1.2.1. Instance
A `Portfolio` object assumes the form

```python
p = Portfolio(name: str, assets: list, init_inv: float, start: datetime, end: datetime, source: str, crypto: bool)
```
where

| Parameter | Description | Optional? |
|-----------|-------------|--------------|
| `name`    | portfolio's name (this parameter is stored in a dictionary inside the class and makes<br /> reference to the `Portfolio` object; as a result, each name must be unique) | No |
| `assets` | assets list to compose the portfolio | No |
| `init_inv` | initial investment (must be sufficient to buy, at least, one share of each asset) | Yes (default: 10.000) |
| `start` and `end`| initial and final date, respectively, to collect the adjust prices of each asset in `assets`<br /> from `source` | Yes (default: `None`)|
| `source`| source to collet the data (for now, only investing.com (`'iv'`), or Yahoo Finance<br /> (`'yf'`) are implemented) | Yes (default: 'iv')|
| `crypto` | since the text formatting is different when downloading cryptos, this parameters must<br /> be set to True to account for it. In this case, the assets list must contain only cryptos | Yes (default: `False`) |

As properties, we have

1. `name` (bool)
2. `assets` (list)
3. `init_inv` (float)
4. `dates` (tuple)
5. `prices` (pd.DataFrame)
6. `init_weights` (np.ndarray)

As for instance attributes:

1. `equity` (`Equity` object)
2. `__gen_eq` (bool - internal attribute)
3. `_cryptos` (list - internal attribute)

**Obs:** the `equity` attribute will be discussed in Sec. 2.

### 1.2.2. Class
1. `registered`: dictionary that stores the name property of the `Portfolio`'s. Given `p` as defined above, one gets

```python
registered = {p.name: p}
```
such that one is able to reference it by its name. This is useful when comparing several `Portfólio`s.

2. `delta`: tolerance to verify if the weights sum to 1. Default: 0.001. That is, if the `init_weights` property is altered, say, `p.init_weights = new_weights`, the weights must satisfy:

```python
np.abs(1 - np.sum(new_weights)) > Portfolio.delta
```
If `True`, `AttributeError` will be raised.

3. `overwrite`: if `True`, one can overwrite a `Portfolio`'s name. Default: `False`.

## 1.3. Instanciating a `Portfolio` object
### 1.3.1. Non-empty
The minimum parameters required to instanciate a `Portfolio` object are: **i)** its name, and **ii)** a list of assets. When one only gives such parameters, however, an empty Portfolio is created. This is useful when one already has a price dataframe (and is not interested in downloading the data)  or when the prices cannot be downloaded from the `yf.Ticker.history()` method from Yahoo Finance or from the `iv.get_stock_historical_data()` investing.com method (until further options are implemented), and only intends to transform it to a `Portfolio` object. To in fact download the data when instanciating an object, the parameters `start` and `end` must be informed. The class will hence download the prices from investing.com or yahoo finance within the time period informed.

For instance, to instanciate a `Portfolio` with the assets ITSA4, SULA11 and SAPR11, between the period 01/01/2020 and 01/01/2021, from investing.com, we do

```python
p1 = Portfolio(
    name='example',
    assets=['ITSA4', 'SULA11', 'SAPR11'],
    start=datetime(2020, 1, 1),
    end=datetime(2021, 1, 1)
)
```
The adjust prices dataframe can be visualized with `p1.prices`. The above procedure automatically constructs an equally-weighted portfolio:

```python
p1.init_weights -> array([0.33333333, 0.33333333, 0.33333333])
```

In order to alter them, say, to 50%, 30% and 20%, we do

```python
p1.init_weights = np.array([.5, .3, .2])
```

For cryptocurrencies, say, BTC and ETH, we download it from Yahoo Finance with

```python
p2 = Portfolio(
    name='example-crypto',
    assets=['BTC-USD', 'ETH-USD'],
    start=datetime(2020, 1, 1),
    end=datetime(2021, 1, 1)
    source='yf',
    crypto=True
)
```

### 1.3.2. Empty
The prices dataframe from a market index, such as IBOV in BR, for instance, are acquired through the `market_index()` function in `quant_tools()` (only IBOV and IFIX are implemented for now). As a result, in order to construct an IBOV `Portfolio`, we download the data from the function just mentioned, create an empty `Portfolio` (not informing the dates), and update the `prices` and the `dates` properties with the downloaded data:

```python
start, end = datetime(2020, 1, 1), datetime(2021, 1, 1)

ibvp_ = qt.market_index('bvsp', start, end)

ibvp = Portfolio('IBVP', ['BVSP'])
ibvp.prices = ibvp_
ibvp.dates = (start, end)
Portfolio.register(ibvp)
```

Notice, however, that in this case it is necessary to register the `Portfolio` to add it in the `Portfolio.registered` dictionary. Although this is not mandatory, it is highly recommended so as to have access to further comparison features.

## 1.4. Methods
### 1.4.1. Magic

1. `__str__`: returns `p.name`
2. `__len__`: returns `len(p.assets)`
3. `__add__`: it is possible to add two `Portfolio`'s, `p1` and `p2`, in order to obtain a new `Portfolio` `p = p1 + p2`. It actually simply concatenates (with an inner join) the prices dataframe of the `Portfolio`'s being added and registers the resulting `Portfolio` under the name `p1.name + p2.name`:

```python
pd.concat(
    [p1, p2],
    axis=1,
    join='inner'
)
```
Like any new `Portfolio`, the initial weight distribution is uniform. Notice, however, that an error may be raised when considering the initial investment of `p1` with all the assets from `p`. That is, if `p1.init_inv` = 1000 suffices to purchase at least one share of its *n* assets, when we add it to `p2`, with another set of assets *m*, 1000 may not suffice when considering *n* + *m* assets. To this end, the addition method considers the minimum allocation required to build the resulting `Portfolio` `p`. If this value is smaller (greater) than `p1.init_inv`, the latter (former) will be used.

### 1.4.2. Instance
Let `p` be a `Portfolio`.

1. `p.d_returns()`: individual or total daily returns

2. `p.m_returns()`: analogous to `d_returns()`, but for monthly returns

3. `p.a_returns()`: analogous to `d_returns()`, but for annual returns (observe, however, that the class implicitly assumes, for now, that `p.prices` refers to **daily** prices

4. `p.acm_returns()`: cumulative returns

5. `p.asset_returns()`: returns a pd.Series with the annual/monthly/total return of the assets composing `p`

6. `p.portfolio_return()`: portfolio returns

7. `p.sample_cov()`: sample covariance matrix

8. `p.benchmark([portfolios])`: plots a benchmark with `p` and the `Portfolios` in the portfolios list.

9. `p.beta(benchmark)`: beta of `p` with the `Portfolio` passed as `benchmark`

10. `p.volatility()`: daily, monthly and annual volatility

11. `p.s_index()`: annualized Sharpe or Sortino index

12. `p.var()`: historical or parametric VaRs (95, 97, 99 and 99.9)

13. `p.cvar()`: historical or parametric CVaRs (95, 97, 99 and 99.9)

14. `p.all_vars()`: pd.DataFrame with the historical, parametric and the parametric adjusted VaRs

15. `p.downside()`: portfolio downside (standard deviation of the negative returns)

16. `p.upside()`: portfolio upside (standard deviation of the positive returns)

17. `p.rol_drawdown()`: rolling maximum drawdown within some time window

18. `p.calc_skewness()`: given the return distribution, returns its skewness

19. `p.calc_kurtosis()`: given the return distribution, returns its kurtosis

20. `p.shapiro_test()`: verify, within a given confidence interval, whether the return distribution assumes a normal distribution

21. `p.metrics()`: pd.DataFrame with a collection of metrics

22. `p.transfer()`: method that transfers the data from the original Portfolio, as the assets, dates and prices, to a new Portfolio


### 1.4.3. Class
Let `p` be a `Portfolio`.

1. `register(p)`: adds `p` in the `Portfolio.registered` dictionary, with `p.name` as key and `p` as value

2. `unregister(name)`: removes the `Portfolio` whose name property is `name` from `Portfolio.registered`

3. `all_rets()`: returns a pd.Series with the returns from all registered Portfolios

4. `all_vols()`: returns a pd.Series with the volatilities from all registered Portfolios

5. `all_sindex()`: returns a pd.Series with the Sharpe or Sortino index from all registered Portfolios

6. `all_weights()`: returns a pd.Series with the weights from all registered Portfolios

7. `all_metrics([portfolios])`: applies `p.metrics()` for all `p` in the portfolios list and consolidates them into a single pd.DataFrame

### 1.4.4. Static
None.

# 2. Equity Class

## 2.1. Objective

To provide assistance in the `Portfolio` class by controlling the evolution in equity of an asset, its number of shares and investment operations, such as short and long operations. More impotantly, with methods to follow the weights distribution and the equity, we can calculate the returns more accurately. In the same line of reasoning, note that the return of an asset in a portfolio must not rely on the number of shares one possessess of it (it will only participate in the portfolio equity). Thus, to obtain a portfolio return one can simply calculate the price variation of its assets (individual returns) and then evalutes the weighted average. As the methods of this class keeps track of the weight evolution, at each step of time the weighted average is evaluated, it is so with the updated weights.

## 2.2. Attributes
### 2.2.1. Instance
An `Equity` object assumes the form

```python
eq = Equity(prices: pd.DataFrame, init_allocations: dict, cryptos: list)
```
where

| Parameter | Description | Optional? |
|-----------|-------------|--------------|
| `prices` | dataframe containing the adjusted prices | No |
| `init_allocations` | dictionary of the assets initial allocations: {asset_A: 100, asset_B: 150} | No |
| `cryptos` | as cryptocurrencies can be purchased in fractions (one can buy 0.15 BTC) one<br /> needs to inform them so as to avoid errors when calculating the number of shares | Yes (default: []) |

As properties, we have
1. `prices` (pd.DataFrame)
2. `init_allocations` (dict)
3. `eq_history` (dict): keeps track of the portfolio operations, with their corresponding dates as keys and a dictionary containing the allocations as values, *e.g.*, given the initial investment of $ 100 in asset_A and $ 150 in asset_B at 01/01/2020, we get
```python
eq.eq_history -> {'2020-01-01': {asset_A: (100, nA), asset_B: (150, nB)}}
```
`nA` and `nB` being the number of shares opertaed (the first entry is always the initial allocation)

4. `n_shares` (pd.Series): Series with the number of shares of each asset

### 2.2.2. Class

None.

## 2.3. Instanciating an `Equity` Object
`Equity` objects are implicitly used inside the `Portfolio` class so as to properly compute the portfolio returns (daily, cumulative, etc). The `prices` parameter is nothing more than the `prices` property from a `Portfolio` object. For the initial allocations, it is automatically set with the `init_inv` and `init_weights` parameters in the form

```python
init_allocations = {
    x[0]: p.init_inv * x[1]
    for x in zip(p.assets, p.init_weights)
}
```
with `p` a `Portfolio`. Assuming we are not interested in cryptocurrencies for now, given `p` we have

```python
p.equity = Equity(p.prices, init_allocations)
```
with `init_allocations` as given above. Clearly, this attribute can only be instanciated once the `prices` and `init_weights` properties have been defined. Notice, however, that if such properties are updated one needs to update `equity` as well. Thus, every time the `prices` or `init_weights` setters are called, a new `Equity` object is instanciated with the updated properties. As the `prices` property is set *before* the `init_weights` property and both define a new `Equity` object, the `__gen_eq` internal attribute forbids the `prices` setter, the first time it is called, to instanciate the `Equity` object. This being the reason for its existence.

## 2.4. Methods

### 2.4.1. Magic
None.

### 2.4.2. Instance
Let `eq` be an `Equity`.

1. `get_total_equity()`: pd.Series or pd.DataFrame with the equity's evolution

2. `__set_total_equity()`: internal method to update the total equity

3. `share_qtd()`: calculates the number of shares that can be bought/sold, given an allocation and a date

4. `__init_equity()`: internal method to construct the initial equity. When `len(eq.eq_history) == 1` (only the initial investment has been made), it is equivalent to `get_total_equity()`

5. `weights_track()`: returns a pd.DataFrame with the evolution of the assets' weights in time

6. `weighted_return()`: returns the equity weighted return. Consider the assets' individual returns, given the initial investment, along with the evolution of their weights

7. `eq_return()`: calculates the equity return. **Obs:** to compute a return of some `Portfolio` object, this is the method responsible for obtaining it

8. `operation()`: realizes a portfolio operation, considering the given allocations and asset prices at the date informed

9. `avg_prices()`: returns a dictionary with the current average prices of the assets composing the portfolio, with the assets as keys and the prices as values

10. `avg_track()`: returns a pd.DataFrame with the evolution of the average price of the assets composing the portfolio. The average changes as purchases are made

### 2.4.3. Class
None.

### 2.4.4. Static

1. `profit`: calculates the profit of a sale, considering the current equity at some date and the average asset prices

