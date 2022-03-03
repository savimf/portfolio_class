import pandas as pd
import numpy as np
import investpy as iv
import yfinance as yf
import quandl
import pypfopt as pf
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd
from sklearn import metrics
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, norm, sem, t
from scipy.optimize import minimize
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import visuals


save_path = 'pictures/'


def wallet(
    assets: list,
    start: dt, end: dt,
    source: str='iv',
    crypto: bool=False,
    us: bool=False,
    **kwargs
) -> pd.DataFrame:
    """Returns a pd.DataFrame with the daily adjusted prices of the assets list,
    within the interval [start, end].

    Args:
        assets (list): assets list to be downloaded.
        start (datetime): start date.
        end (datetime): final date.
        source (str, optional): data source ('iv' or 'yf'). Default: 'iv'.
        crypto (bool, optional): since the text formatting is different when
        downloading cryptos, this parameters must be set to True to account for
        it. In this case, the assets list must contain only cryptos.

    Returns:
        pd.DataFrame.
    """
    prices = pd.DataFrame()

    if sum(1 for d in (start, end) if isinstance(d, dt)) == 0:
        return prices

    if source == 'iv':
        for asset in assets:
            prices[asset] = iv.get_stock_historical_data(
                stock=asset,
                country='brazil' if not us else 'united states',
                from_date=start.strftime('%d/%m/%Y'),
                to_date=end.strftime('%d/%m/%Y'),
                **kwargs
            )['Close']
    elif source == 'yf':
        if not crypto:
            for asset in assets:
                asset = f'{asset}.SA' if not us else asset
                t = yf.Ticker(asset)
                prices[asset] = t.history(
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    interval='1d',
                    **kwargs
                )['Close']

            # excluding the '.SA'
            if not us:
                rename_columns = map(
                    lambda c: c[:-3], prices.columns
                )
                prices.columns = rename_columns
        else:
            for asset in assets:
                t = yf.Ticker(asset)
                prices[asset] = t.history(
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    interval='1d',
                    **kwargs
                )['Close']
    else:
        raise NameError('Invalid source.')

    prices.index = pd.to_datetime(prices.index)
    return prices


def get_quandl(rate: str, start: dt, end: dt) -> pd.DataFrame:
    """Gathers data from quandl.

    Args:
        rate (str): ipca, imab, or selic.
        start (datetime): start date.
        end (datetime): final date.

    Raises:
        NameError: if rate not in ('ipca', 'imab', 'selic')

    Returns:
        pd.DataFrame.
    """
    cod = {
        'ipca': 13522,
        'imab': 12466,
        'selic': 4189
    }

    if rate.lower() not in cod.keys():
        raise NameError('Invalid rate. Use ipca, imab or selic.')

    df = quandl.get(
        f'BCB/{int(cod[rate.lower()])}',
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d')
    )
    df.rename(columns={'Value': rate.upper()}, inplace=True)
    return df


def selic(start: dt, end: dt, is_number: bool=False, period: str='a'):
    """Returns the daily, monthly or annual variation of the Selic rate,
    gathered from quandl.

    Args:
        start (datetime): start date.
        end (datetime): final date.
        is_number (bool, optional): if False, returns a pd.Series
        with the variations, if True, returns the mean value of the period.
        Default: False.
        period (str, optional):
            - 'd' for daily,
            - 'm' for monthly
            - 'a' for annual
            Default: 'a'.

    Raiser:
        IndexError: if 'period' not in ('d', 'm', 'a').

    Returns:
        pd.Series or float.
    """
    s = get_quandl('selic', start, end) / 100
    s = s.squeeze()

    if is_number:
        if period not in ('d', 'm', 'a'):
            raise IndexError("Invalid period. Use 'd', 'm' or 'a'.")

        s = s.mean()

        # annual / monthly / daily
        if period == 'a':
            pass
        elif period == 'm':
            s = (1 + s) ** (1 / 12) - 1
        elif period == 'd':
            s = (1 + s) ** (1 / 252) - 1
    return s


def compound(r: pd.Series) -> float:
    """Compound a return series.

    Args:
        r (pd.Series): return series.

    Returns:
        float.
    """
    # np.log1p(r) = np.log(1 + r)
    # np.expm1(r) = np.exp(r - 1)
    return np.expm1(np.log1p(r).sum())


def returns(prices: pd.DataFrame, which: str='daily', period: str='a'):
    """Returns the daily/monthly/annual returns from the prices dataframe.

    Ex:
        - which = 'daily' (daily returns)
        - which = 'monthly' (monthly returns)
        - which = 'annual' (annual returns)
        - which = 'total' (period total variation)
        - which = 'acm' (cumulative returns)

    Args:
        prices (pd.DataFrame): adjusted prices dataframe.
        which (str, optional): desired return. Default: 'daily'.
        period (str, optional): only valid for which = 'total';
        realizes (1 + r) ** period - 1. Default: 'a'.

    Returns:
        pd.DataFrame or pd.Series.
    """
    r = prices.pct_change().dropna()
    if which == 'daily':
        return r.to_period('D')
    elif which == 'monthly':
        m_rets = r.resample('M').apply(
            compound
        )

        return m_rets
    elif which == 'annual':
        a_rets = r.resample('Y').apply(
            compound
        )

        return a_rets
    elif which == 'total':
        rets = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

        if period not in ('m', 'a'):
            return rets

        n_days = prices.shape[0]
        n_years = n_days / 252
        if period == 'm':
            return (1 + rets) ** (1 / (12 * n_years)) - 1
        elif period == 'a':
            return (1 + rets) ** (1 / n_years) - 1
        raise TypeError("Invalid period. Use 'm' or 'a'.")
    elif which == 'acm':
        return (1 + r).cumprod()
    raise TypeError(
        "Invalid return. Use 'daily', 'total', 'monthly, or 'acm'."
    )


def search(txt: str, n: int):
    """Gathers, from investing.com, the first n search results of
    txt.

    Args:
        txt (str): object of interest: 'tesouro', 'bvsp' or 'ifix'.
        n (int): number of results.

    Returns:
        iv..utils.search_obj.SearchObj: iterator.
    """
    pdt = []
    if txt == 'tesouro':
        pdt = ['bonds']
    elif txt in ('bvsp', 'ifix'):
        pdt = ['indices']

    search_results = iv.search_quotes(
        text=txt,
        products=pdt,
        countries=['brazil'],
        n_results=n
    )

    return search_results


def market_index(index: str, start: dt, end: dt) -> pd.DataFrame:
    """Returns a pd.DataFrame with the adjusted prices of index.

    Args:
        index (str): market index ('ifix' or 'bvsp').
        start (datetime): start date.
        end (datetime): final date.

    Raises:
        NameError: if index not in ('ifix', 'bvsp').

    Returns:
        pd.DataFrame.
    """
    if index not in ('ifix', 'bvsp'):
        raise NameError("Invalid index. User 'ifix' or 'bvsp'.")

    df = search(index, 1).retrieve_historical_data(
        from_date=start.strftime('%d/%m/%Y'),
        to_date=end.strftime('%d/%m/%Y')
    )['Close'].to_frame()

    df.rename(columns={'Close': index.upper()}, inplace=True)
    return df


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the mean absolute error between y_true and
    y_pred.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.

    Returns:
        float.
    """
    return metrics.mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the root mean squared error between y_true
    and y_pred.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.

    Returns:
        float.
    """
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print the MAE and RSME.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.
    """
    print(
        f'MAE: {mae(y_true, y_pred)}\n'
        f'RMSE: {rmse(y_true, y_pred)}'
    )


def mae_cov(cov_true: pd.DataFrame, cov_pred: pd.DataFrame) -> float:
    """Returns the MAE between two covariance dataframes.

    Args:
        cov_true (pd.DataFrame): covariance of observed values.
        cov_pred (pd.DataFrame): covariance of predicted values.

    Returns:
        float.
    """
    r = np.sum(
        np.abs(
            np.diag(cov_true) - np.diag(cov_pred)
        )
    ) / len(np.diag(cov_true))

    return r


def mci(data: pd.Series, confidence: float=.95) -> tuple:
    """Evalutes the mean confidence interval of the sample data. Returns
    a tuple containing:

        (mean, interval lower bound, interval upper bound)

    Args:
        data (pd.Series): sample data.
        confidence (float, optional): confidence level. Default: 0.95.

    Returns:
        tuple.
    """
    a = 1. * np.array(data)
    m, ste = np.mean(a), sem(a)
    h = ste * t.ppf((1 + confidence) / 2., len(a) - 1)
    return m, m-h, m+h


def cornish_fisher_z(z: float, s: float, k: float) -> float:
    """Returns the adjusted z-score, taking into consideration
    the distribution skewness (s) and kurtosis (k), by the Cornish-
    Fisher method.

    Args:
        z (float): z-score of the normal distribution.
        s (float): distribution skewness.
        k (float): distribution kurtosis.

    Returns:
        float.
    """
    return z + (1/6) * s * ((z ** 2 - 1) - \
        (1/6) * (2 * z ** 3 - 5 * z) * s) + \
            (1/24) * (z ** 3 - 3 * z) * (k - 3)


def vars_hist(rets: pd.Series) -> dict:
    """Returns a dictionary with:
        - 95% historical VaR
        - 97% historical VaR
        - 99% historical VaR
        - 99.9% historical VaR
    of the return series.

    Args:
        rets (pd.Series): return series.

    Returns:
        dict.
    """
    if not isinstance(rets, pd.Series):
        raise TypeError('Please insert the returns as a time series.')

    var_95 = np.nanpercentile(rets, 5)
    var_97 = np.nanpercentile(rets, 3)
    var_99 = np.nanpercentile(rets, 1)
    var_99_9 = np.nanpercentile(rets, .1)

    return {
        95: var_95,
        97: var_97,
        99: var_99,
        99.9: var_99_9
    }


def vars_gaussian(rets: pd.Series, modified: bool=False) -> dict:
    """Returns a dictionary with:
        - 95% parametric VaR
        - 97% parametric VaR
        - 99% parametric VaR
        - 99.9% parametric VaR
    of the return series.

    Args:
        rets (pd.Series): return series.
        modified (bool, optional): if True, considers the skewness
        and kurtosis of the distribution and realizes the Cornish-
        Fisher correction.

    Returns:
        dict.
    """
    lvls = (95, 97, 99, 99.9)

    # z-scores
    zs = [norm.ppf(1 - lvl / 100) for lvl in lvls]

    if modified:
        s, k = skew(rets), kurtosis(rets)
        zs = [cornish_fisher_z(z, s, k) for z in zs]

    vol = rets.std()
    var = {
        v[0]: (rets.mean() + v[1] * vol)
        for v in zip(lvls, zs)
    }
    return var


def cvars_hist(rets: pd.Series) -> dict:
    """Returns a dictionary with:
        - 95% historical CVaR
        - 97% historical CVaR
        - 99% historical CVaR
        - 99.9% historical CVaR
    of the return series.

    Args:
        rets (pd.Series): return series.

    Returns:
        dict.
    """
    var = vars_hist(rets)

    c_vars = {
    i[0]: -rets[rets <= i[1]].mean()
    for i in var.items()
    }
    return c_vars


def cvars_gaussian(rets: pd.Series, modified: bool=False) -> dict:
    """Returns a dictionary with:
        - 95% parametric CVaR
        - 97% parametric CVaR
        - 99% parametric CVaR
        - 99.9% parametric CVaR
    of the return series.

    Args:
        rets (pd.Series): return series.
        modified (bool, optional): if True, considers the skewness
        and kurtosis of the distribution and realizes the Cornish-
        Fisher correction.

    Returns:
        dict.
    """
    var = vars_gaussian(rets, modified)

    c_vars = {
    i[0]: -rets[rets <= i[1]].mean()
    for i in var.items()
    }
    return c_vars


def vol(weights: np.ndarray, cov: pd.DataFrame, annual: bool=True) -> float:
    """Returns a portfolio's volatility.

    Args:
        weights (np.ndarray): array of weights.
        cov (pd.DataFrame): covariance matrix.
        annual (bool, optional): if True, returns the annualized
        volatility. Default: True.

    Returns:
        float.
    """
    vol = np.sqrt(
        np.dot(weights.T, np.dot(cov, weights))
    )

    if annual:
        return vol * np.sqrt(252)
    return vol


def max_drawdown(rets: pd.Series) -> float:
    """Returns the maximum drawdown of the return series.

    Args:
        rets (pd.Series): return series.

    Returns:
        float.
    """
    acm = (1 + rets).cumprod()
    peaks = acm.cummax()
    drawdown_ = acm / peaks - 1
    return drawdown_.min()


def beta(portfolio_r: pd.Series, benchmark_r: pd.Series) -> float:
    """Returns the beta of the portfolio and some benchmark,
    considering their return series.

    Args:
        portfolio_r (pd.Series): portfolio return series.
        benchmark_r (pd.Series): benchmark return series.

    Returns:
        float.
    """
    df = pd.concat(
        [portfolio_r, benchmark_r],
        axis=1,
        join='inner'
    )

    Y = df.iloc[:,0]
    X = df.iloc[:,1]
    X = sm.add_constant(X)

    linear_model = sm.OLS(Y, X)
    return linear_model.fit().params[1]


def sharpe(ret: float, vol: float, risk_free_rate: float=.03) -> float:
    """Returns the Sharpe index, given the return, volatilidade and
    risk-free rate.

    Args:
        ret (float): portfolio return.
        vol (float): portfolio volatility.
        risk_free_rate (float, optional): risk-free rate. Default: 0.03.

    Returns:
        float.
    """
    return (ret - risk_free_rate) / vol


def minimize_vol(target_return: float, exp_rets: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    """Returns the weights of the minimum variance portfolio, given:
    target return, expected returns, and the covariance matrix.

    Args:
        target_return (float): portfolio target return.
        exp_rets (pd.Series): portfolio expected return.
        cov (pd.DataFrame): covariance matrix.

    Returns:
        np.ndarray.
    """
    n = exp_rets.shape[0]  # number of assets
    init_guess = np.repeat(1/n, n)

    # constraints
    bounds = ((0.0, 1.0), ) * n  # n tuples
    return_is_target = {
        'type': 'eq',
        'args': (exp_rets, ),
        'fun': lambda w, exp_rets: target_return - exp_rets.dot(w)
    }
    weights_to_one = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    }

    results = minimize(
        vol,
        init_guess,
        args=(cov,),
        method='SLSQP',
        options={'disp': False},
        constraints=(return_is_target, weights_to_one),
        bounds=bounds
    )
    return results.x


def maximize_sr(exp_rets: pd.Series, cov: pd.DataFrame, risk_free_rate: float=.03) -> np.ndarray:
    """Returns the weights of the maximum Sharpe ratio portfolio, given:
    expected returns, the covariance matrix and the risk-free rate.

    Args:
        exp_rets (pd.Series): portfolio expected return.
        cov (pd.DataFrame): covariance matrix.
        risk_free_rate (float, optional): risk-free rate. Default: 0.03.

    Returns:
        np.ndarray.
    """
    def neg_sharpe_ratio(weights: np.ndarray, exp_rets: pd.Series, cov: pd.DataFrame, risk_free_rate: float=.03) -> float:
        """Returns the negative Sharpe index, given an array of weights.

        Args:
            weights (np.ndarray): array of weights.
            exp_rets (pd.Series): expected returns.
            cov (pd.DataFrame): covariance matrix.
            risk_free_rate (float, optional): risk-free rate. Default: 0.03.

        Returns:
            float.
        """
        r = exp_rets.dot(weights)
        v = vol(weights, cov, annual=False)
        return - (r - risk_free_rate) / v

    n = exp_rets.shape[0]  # number of assets
    init_guess = np.repeat(1/n, n)

    # constraints
    bounds = ((0.0, 1.0), ) * n  # n tuples

    weights_to_one = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    }

    results = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(exp_rets, cov, risk_free_rate),
        method='SLSQP',
        options={'disp': False},
        constraints=(weights_to_one),
        bounds=bounds
    )
    return results.x


def gmv(cov: pd.DataFrame) -> np.ndarray:
    """Returns the weights of the Global Minimum Variance
    portfolio.

    Args:
        cov (pd.DataFrame): covariance matrix.

    Returns:
        np.ndarray.
    """
    n = cov.shape[0]
    return maximize_sr(np.repeat(1, n), cov, 0)


def optimal_weights(exp_rets: pd.DataFrame, cov: pd.DataFrame, n_points: int=10) -> np.ndarray:
    """Returns an array of weights that minimizes the volatility, given
    the expected returns and the covariance matrix. Considers the extrema
    of the expected returns to create a list with n_points equally spaced
    returns. For each return, the minimize_vol function is applied.

    Args:
        exp_rets (pd.DataFrame): expected returns.
        cov (pd.DataFrame): covariance matrix.
        n_points (int. optional): number of equally spaced intervals,
        between the extrema of exp_rets. Default: 10.

    Returns:
        np.ndarray.
    """
    target_returns = np.linspace(exp_rets.min(), exp_rets.max(), int(n_points))
    weights = [
        minimize_vol(target_return, exp_rets, cov)
        for target_return in target_returns
    ]
    return weights


def find_port_min_vol(portfolios: pd.DataFrame, col_name: str='Volatility') -> pd.DataFrame:
    """Returns the minimum volatility portfolio among the portfolios
    dataframe. The search assumes that the column name whose values
    are the portfolios' volatilities is col_name.

    Args:
        portfolios (pd.DataFrame): dataframe containing several
        portfolios.
        col_name (str, optional): column name containing the
        portfolios' volatilities. Default: Volatility.

    Returns:
        pd.DataFrame.
    """
    min_vol = portfolios[col_name].min()

    port_min_vol = portfolios.loc[portfolios[col_name] == min_vol]
    port_min_vol = port_min_vol.T.rename(
        columns={port_min_vol.index[0]: 'Values'}
    )
    return port_min_vol


def find_port_max_sr(portfolios: pd.DataFrame, col_name: str='Sharpe_Ind') -> pd.DataFrame:
    """Returns the maximum Sharpe ratioy portfolio among the portfolios
    dataframe. The search assumes that the column name whose values
    are the portfolios' Sharpe indices is col_name.

    Args:
        portfolios (pd.DataFrame): dataframe containing several
        portfolios.
        col_name (str, optional): column name containing the
        portfolios' volatilities. Default: Sharpe_Ind.

    Returns:
        pd.DataFrame.
    """
    max_sr = portfolios[col_name].max()

    port_max_sr = portfolios.loc[portfolios[col_name] == max_sr]
    port_max_sr = port_max_sr.T.rename(
        columns={port_max_sr.index[0]: 'Values'}
    )
    return port_max_sr


def run_cppi(
    risky_r: pd.DataFrame,
    safe_r: pd.DataFrame=None,
    m: int=3,
    start: float=1000.,
    floor: float=.8,
    risk_free_rate: float=.03,
    drawdown: float=None
) -> dict:
    """Realizes a backtest regarding the CPPI strategey, given the
    risky assets returns (risky_r) and those of the safe assets.
    Returns a dictionary containing:

    - wealth: investment historical values, with the strategy
    - risky_wealth: investment historical values, without the strategy
    - risk_budget: cushion history
    - risky_allocation: weights that have been attributed to risky_r
    - m: cushion multiplier
    - start: investment's initial value
    - floor: minimum values history

    Args:
        risky_r (pd.DataFrame): risky asset returns.
        safe_r (pd.DataFrame, optional): safe asset returns. If None,
        risk_free_rate / 12 is considered. Default: None.
        m (int, optional): cushion multiplier. Default: 3.
        start (float, optional): investment's initial value. Default: 1000.
        floor (float, optional): minimum accepted value (in percentage).
        Default: 0.8.
        risk_free_rate (float, optional): risk-free rate; only valid if
        safe_r == None. Default: 0.1.
        drawdown (float, optional): maximum accepted drawdown; updates the
        floor value at each step. Default: None.

    Returns:
        dict.
    """
    # parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    peak = start
    floor_value = start * floor

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = risk_free_rate / 12

    account_hist = pd.DataFrame().reindex_like(risky_r)
    cushion_hist = pd.DataFrame().reindex_like(risky_r)
    risky_w_hist = pd.DataFrame().reindex_like(risky_r)
    floor_value_hist = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        # updating the floor if there is a drawdown constraint
        if drawdown:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
            floor_value_hist.iloc[step] = floor_value

        # calculating the weights
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)  # no weight greater than 1
        risky_w = np.maximum(risky_w, 0)  # no weight lower than 0
        safe_w = 1 - risky_w

        # allocations
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        # updating the values considering the allocations
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + \
            safe_alloc * (1 + safe_r.iloc[step])

        # storing them
        cushion_hist.iloc[step] = cushion
        risky_w_hist.iloc[step] = risky_w
        account_hist.iloc[step] = account_value

    risky_wealth = start * (1 + risky_r).cumprod()

    return {
        'wealth': account_hist,
        'risky_wealth': risky_wealth,
        'risk_budget': cushion_hist,
        'risky_allocation': risky_w_hist,
        'm': m,
        'start': start,
        'floor': floor_value if drawdown is None else floor_value_hist
    }


def get_eff(exp_rets: pd.Series, cov: pd.DataFrame, n_points: int=25) -> pd.DataFrame:
    """Returns a pd.DataFrame with the n_points portfolios composing
    the efficient frontier. The dataframe contains the columns: 'Returns'
    and 'Volatility'.

    Args:
        exp_rets (pd.Series): return series.
        cov (pd.DataFrame): covariance matrix.
        n_points (int, optional): number of portfolios to build the frontier.
        Defaults to 25.

    Returns:
        pd.DataFrame.
    """
    weights = optimal_weights(exp_rets, cov, n_points)

    rets = [exp_rets.dot(w) for w in weights]
    vols = [vol(w, cov) for w in weights]
    return pd.DataFrame({'Returns': rets, 'Volatility': vols})


def plot_portfolios(
    portfolios: pd.DataFrame, color: str='brg',
    size: tuple=(12, 10), is_return: bool= False,
    save: bool=False
):
    """Plots the portfolios from the given dataframe in the vol x ret
    plane, highlighting in blue (red) the minimum volatility (maximum
    Sharpe ratio) portfolio.

    Args:
        portfolios (pd.DataFrame): dataframe containing several
        portfolios.
        color (str, optional): color palette. Default: 'brg'.
        size (tuple, optional): plot size. Default: (12, 10).
        save (bool, optional): if True, saves the plot in a png
        with dpi=200, and name 'portfolios' in save_path.
    """
    plt.figure(figsize=size)
    cor = color
    ax = sns.scatterplot(
        x='Volatility', y='Returns',
        hue='Sharpe_Ind', data=portfolios,
        palette=cor
    )

    norm = plt.Normalize(
        0,
        portfolios['Sharpe_Ind'].max()
    )

    sm = plt.cm.ScalarMappable(cmap=cor, norm=norm)
    sm.set_array([])

    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    port_min_vol = find_port_min_vol(portfolios).T
    port_max_sr = find_port_max_sr(portfolios).T
    plt.scatter(
        x=port_max_sr['Volatility'],
        y=port_max_sr['Returns'], c='red',
        marker='o', s=200
    )
    plt.scatter(
        x = port_min_vol['Volatility'],
        y = port_min_vol['Returns'], c='blue',
        marker='o', s=200
    )

    plt.title('Portfolios')

    if save:
        plt.savefig(save_path + 'portfolios.png', dpi=200)

    return ax if is_return else plt.show()


def plot_eff(
    exp_rets: pd.DataFrame, cov: pd.DataFrame,
    n_points: int=25, risk_free_rate: float=.03,
    show_cml: bool=False, show_ew: bool=False,
    show_gmv: bool=False, plot_in: str='sns',
    size: tuple=(15, 6), style: str='.-',
    name: str=None, is_return: bool=False,
    **kwargs
):
    """Plots the efficient frontier, given the expected returns
    and the covariance matrix.

    Args:
        exp_rets (pd.DataFrame): expected returns.
        cov (pd.DataFrame): covariance matrix.
        n_points (int, optional): number of points to be shown in
        the frontier. Default: 25.
        risk_free_rate (float, optional): risk-free rate. Default: 0.03.
        show_cml (bool, optional): if True, plots the line that connects
        the risk-free asset with the maximum Sharpe ratio portfolio (known
        as the capital market line.). Default: False.
        show_ew (bool, optional): if True, plots the equally-weighted port-
        folio. Default: False.
        show_gmv (bool, optional): if True, plots the GVM portfolio.
        Default: False.
        plot_in (str, optional): 'sns' or 'go'. Default: 'sns'.
        size (tuple, optional): plot size. Default: (15, 6).
        style (str, optional): linestyle (only valid if plot_in == 'sns').
        Default: '.-'.
        name (str, optional): if != None, saves the plot in the 'save_path'
        directory. Default: None.
        is_return (bool, optional): if True, returns the plot, instead of just
        showing it. Default: False.

    Raises:
        TypeError: if plot_in not in ('sns', 'go').
        NameError: if len(name) == 0.
    """
    eff = get_eff(exp_rets, cov, n_points)

    if plot_in == 'sns':
        ax = eff.plot.line(
            x='Volatility', y='Returns',
            figsize=size, style=style,
            legend=False, **kwargs
        )
        if show_ew:
            n = exp_rets.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = exp_rets.dot(w_ew)
            v_ew = vol(w_ew, cov)

            ax.plot(
                [v_ew], [r_ew],
                color='goldenrod', marker='o',
                markersize=10, label='EW'
            )
        if show_gmv:
            w_gmv = gmv(cov)
            r_gmv = exp_rets.dot(w_gmv)
            v_gmv = vol(w_gmv, cov)

            ax.plot(
                [v_gmv], [r_gmv],
                color='midnightblue', marker='o',
                markersize=10, label='GMV'
            )
        if show_cml:
            ax.set_xlim(left=0)

            w_msr = maximize_sr(exp_rets, cov, risk_free_rate)
            r_msr = exp_rets.dot(w_msr)
            v_msr = vol(w_msr, cov)

            # add capital market line
            cml_x = [0, v_msr]
            cml_y = [risk_free_rate, r_msr]

            ax.plot(
                cml_x, cml_y,
                color='green',
                marker='o',
                linestyle='dashed',
                markersize=10,
                linewidth=2,
                label='Cap. Market Line'
            )
        plt.title('Efficient Frontier and the Cap. Market Line')
        plt.ylabel('Return')
        plt.legend()

        if name:
            if len(name) > 0:
                plt.savefig(save_path + str(name) + '.png', dpi=200)
            else:
                raise NameError('Figure name must have, at least, one character.')

        return ax if is_return else plt.show()
    elif plot_in == 'go':
        titles = [
            'Efficient Frontier and the Cap. Market Line',
            'Volatility',
            'Return'
        ]

        layout = layout_settings(titles)
        fig = go.Figure(layout=layout)

        fig.add_trace(
            go.Scatter(
                x=eff['Volatility'],
                y=eff['Returns'],
                mode='lines+markers',
                marker={
                    'line': {
                        'color': '#333',
                        'width': .5
                    }
                },
                name='Efficient Frontier',
                **kwargs
            )
        )
        if show_ew:
            n = exp_rets.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = exp_rets.dot(w_ew)
            v_ew = vol(w_ew, cov)

            fig.add_trace(
                go.Scatter(
                    x=[v_ew],
                    y=[r_ew],
                    mode='markers',
                    name='EW',
                    marker={
                        'color': 'yellow',
                        'size': 10
                    }
                )
            )
        if show_gmv:
            w_gmv = gmv(cov)
            r_gmv = exp_rets.dot(w_gmv)
            v_gmv = vol(w_gmv, cov)

            fig.add_trace(
                go.Scatter(
                    x=[v_gmv],
                    y=[r_gmv],
                    mode='markers',
                    name='GMV',
                    marker={
                        'color': 'darkblue',
                        'size': 10
                    }
                )
            )
        if show_cml:
            w_msr = maximize_sr(exp_rets, cov, risk_free_rate)
            r_msr = exp_rets.dot(w_msr)
            v_msr = vol(w_msr, cov)

            fig.add_trace(
                go.Scatter(
                    x=[0, v_msr],
                    y=[risk_free_rate, r_msr],
                    mode='lines+markers',
                    name='Cap. Market Line',
                    marker={
                        'color': 'green',
                        'size': 10,
                        'opacity': .7,
                        'line': {'width': .8}
                    },
                    line={'dash': 'dash'}
                )
            )
        return fig if is_return else fig.show()
    else:
        raise TypeError("Invalid parameter. Use 'sns' or 'go'.")


def comparison(vol_opt: float, vol_eq: float, ret_opt: float, ret_eq: float, risk_free_rate: float) -> None:
    """Plots the comparison between the optimized values of a portfolio
    (variables ending with opt) and another portfolio (serving as benchmark,
    e.g., the equally-weighted; variables ending with eq).

    Args:
        vol_opt (float): volatility of the optimized portfolio.
        vol_eq (float): volatility  of the benchmark portfolio.
        ret_opt (float): returns of the optimized portfolio.
        ret_eq (float): returns of the benchmak portfolio.
        risk_free_rate (float): risk-free rate.
    """
    vol_opt = round(vol_opt, 4)
    vol_eq = round(vol_eq, 4)

    sgn = '+'
    if vol_opt > vol_eq:
        sgn = '-'
    print('Optimized volatility: '
        f'{vol_opt * 100} %\n'
        'Benchmark volatility: '
        f'{vol_eq * 100} %\n'
        f'Percentage difference: {sgn} {round(np.abs(1 - vol_opt / vol_eq) * 100, 4)} %\n'
    )

    ret_opt = round(ret_opt, 4)
    ret_eq = round(ret_eq, 4)

    sgn = '+'
    if ret_opt < ret_eq:
        sgn = '-'
    print('Optimized returns: '
        f'{ret_opt * 100} %\n'
        'Benchmark returns: '
        f'{ret_eq * 100} %\n'
        f'Percentage difference: {sgn} {round(np.abs(1 - ret_opt / ret_eq) * 100, 4)} %\n'
    )

    sharpe_eq = round(sharpe(ret_eq, vol_eq, risk_free_rate), 4)
    sharpe_opt = round(sharpe(ret_opt, vol_opt, risk_free_rate), 4)

    sgn = '+'
    if sharpe_opt < sharpe_eq:
        sgn = '-'
    print('Optimized Sharpe index: '
        f'{sharpe_opt}\n'
        'Benchmark Sharpe index: '
        f'{sharpe_eq} \n'
        f'Percentage difference: {sgn} {round(np.abs(1 - sharpe_opt / sharpe_eq) * 100, 4)} %\n'
    )


def layout_settings(titles: list=[], **kwargs) -> go.Layout:
    """Settings to use in plotly graphs.

    Args:
        titles (list, optional): list of titles; first
        entry is the figure's title, and the second the
        y-axis label. Default: [].

    Returns:
        go.Layout.
    """
    layout = go.Layout(
        title=titles[0],
        xaxis=dict(
            title=titles[1],
            showgrid=False,
            showspikes=True,
            spikethickness=2,
            spikedash='dot',
            spikecolor='#999999',
            spikemode='across'
        ),
        yaxis=dict(
            title=titles[2],
            showgrid=False
        ),
        plot_bgcolor="#FFF",
        hoverdistance=100,
        spikedistance=1000,
        **kwargs
    )
    return layout


def plot_heat_go(df: pd.DataFrame, title: str='Correlations', color: str='YlOrRd') -> None:
    """Plot a heatmap from plotly, with
        - x = df.columns
        - y = df.columns
        - z = df.corr().

    Args:
        df (pd.DataFrame): dataframe from which the correlations will be
        calculated.
        title (str, optional): title plot. Default:'Correlations'.
        color (str, optional): color scale. Default: 'YlOrRd'.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=df.corr(),
            x=df.columns,
            y=df.columns,
            colorscale=color
        ),

        layout=go.Layout(title=title)
    )

    fig.show()


def plot_returns(
    rets: pd.Series,
    titles: list=None,
    plot_in: str='sns',
    size: tuple=(12, 8),
    is_return: bool=False,
    **kwargs
):
    """Horizontal bar plot of a return series, using #de2d26
    (#3182bd) for negative (positive) values.

    Args:
        rets (pd.Series): return series.
        titles (list, optional): plot labels and title, as
            - [title, xlabel, ylabel].
        Default: None.
        plot_in (str, optional): 'sns' or 'go'. Default: 'sns'.
        size (tuple, optional): plot size. Default: (12, 8).
        is_return (bool, optional): if True, returns the plot, instead of
        just showing it. Default: False.

    Raises:
        TypeError: if plot_in not in ('sns', 'go').
    """
    rets = rets.sort_values(ascending=True)
    colors = ['#de2d26' if r < 0 else '#3182bd' for r in rets.values]

    if plot_in == 'sns':

        plt.subplots(figsize=size)
        ax = sns.barplot(
            x=rets.values,
            y=rets.index,
            palette=colors,
            **kwargs
        )
        plt.title(titles[0])
        plt.xlabel(titles[1])
        plt.ylabel(titles[2])

        return ax if is_return else plt.show()
    elif plot_in == 'go':
        layout = layout_settings(titles)
        fig = go.Figure(layout=layout)

        fig.add_trace(
            go.Bar(
                x=rets.values,
                y=rets.index,
                marker={
                    'color': colors,
                    'line': {
                        'color': '#333',
                        'width': 2
                    }
                },
                hoverinfo='x',
                orientation='h',
                **kwargs
            )
        )
        return fig if is_return else fig.show()
    else:
        raise TypeError("Invalid parameter. Use 'sns' or 'go'.")


def plot_monthly_returns(
    rets: pd.Series,
    title: str='Montly Returns',
    plot_in: str='sns',
    show_mean: bool=True,
    show_median: bool=True,
    size: tuple=(18, 6),
    name: str=None,
    is_return: bool=False,
    **kwargs
):
    """Function to plot the monthly returns given in rets.

    Args:
        rets (pd.Series): monthly return series.
        title (str, optional): plot title. Default: 'Monthly Returns'.
        plot_in (str, optional): 'sns' or 'go'. Default: 'sns'.
        show_mean (bool, optional): if True, shows the return mean.
        Default: True.
        show_median (bool, optional): if True, shows the return median.
        Default: True.
        size (tuple, optional): plot size. Default: (18, 6).
        name (str, optional): if != None, saves the plot in the 'save_path'
        directory. Default: None.
        is_return (bool, optional): if True, returns the plot, instead of
        just showing it. Default: False.

    Raises:
        TypeError: if plot_in not in ('sns', 'go').
        NameError: if len(name) == 0.
    """
    if plot_in == 'sns':
        colors = ['indianred' if r < 0 else 'blue' for r in rets]

        fig, ax = plt.subplots(figsize=size)
        rets.plot.bar(
            ax=ax,
            color=colors,
            label='Returns',
            **kwargs
        )

        if show_mean:
            ax.axhline(y=rets.mean(), ls=':', color='green', label='Mean')
        if show_median:
            ax.axhline(y=rets.median(), ls='-', color='goldenrod', label='Median')

        if show_mean or show_median:
            plt.legend()

        plt.title(title)
        plt.ylabel('%')

        if name:
            if len(name) > 0:
                plt.savefig(save_path + str(name) + '.png', dpi=200)
            else:
                raise NameError('Figure name must have, at least, one character.')

        return ax if is_return else plt.show()
    elif plot_in == 'go':
        colors = ['#de2d26' if r < 0 else '#3182bd' for r in rets.values]
        layout = layout_settings(
            titles=[title, 'Date', '%']
        )
        fig = go.Figure(layout=layout)

        rets.index = rets.index.to_timestamp()
        fig.add_trace(
            go.Bar(
                x=rets.index,
                y=rets.values,
                name='Returns',
                marker={
                    'color': colors,
                    'line': {
                        'color': '#333',
                        'width': 2
                    }
                },
                **kwargs
            )
        )
        if show_mean:
            fig.add_trace(
                go.Scatter(
                    x=[rets.index[0], rets.index[-1]],
                    y=[rets.mean(), rets.mean()],
                    name='mean',
                    marker={
                        'color': '#feb24c',
                        'opacity': .7,
                        'line': {'width': .8}
                    }
                )
            )
        if show_median:
            fig.add_trace(
                go.Scatter(
                    x=[rets.index[0], rets.index[-1]],
                    y=[rets.median(), rets.median()],
                    name='median',
                    marker={
                        'color': '#c51b8a',
                        'opacity': .7,
                        'line': {'width': .8}
                    }
                )
            )
        return fig if is_return else fig.show()
    else:
        raise TypeError("Invalid parameter. Use 'sns' or 'go'.")


def plot_lines(
    dfs: list,
    titles: list=[None, None, None],
    plot_in: str='sns',
    size: tuple=(19, 6),
    name: str=None,
    is_return: bool=False,
    **kwargs
):
    """Plots a lineplot of the dataframes given in dfs.

    Args:
        dfs (list): list of dataframes.
        titles (list, optional): title and labels to be used:
            - [title, xlabel, ylabel]
        Default: [None, None, None].
        plot_in (str, optional): 'sns' or 'go'. Default: 'sns'.
        size (tuple, optional): plot size. Default: (19, 6).
        color (str, optional): color of the lineplot. Default: None.
        name (str, optional): if != None, saves the plot in the save_path
        directory under the name specified. Default: None.
        is_return (bool, optional): if True, returns the plot, instead of
        just showing it. Default: False.

    Raises:
        NameError: if len(name) == 0.
        TypeError: if plot_in not in ('sns', 'go').
    """
    if plot_in == 'sns':
        df_ = dfs[0]
        for df in dfs[1:]:
            df_ = pd.concat(
                [df_, df],
                axis=1,
                join='inner'
            )

        plt.figure(figsize=size)
        ax = sns.lineplot(
            data=df_,
            linewidth=1.5,
            dashes=False,
            **kwargs
        )
        plt.title(titles[0])
        plt.xlabel(titles[1])
        plt.ylabel(titles[2])

        if name:
            if len(name) > 0:
                plt.savefig(save_path + str(name) + '.png', dpi=200)
            else:
                raise NameError('Figure name must have, at least, one character.')

        return ax if is_return else plt.show()
    elif plot_in == 'go':
        layout = layout_settings(titles)
        fig = go.Figure(layout=layout)

        for df in dfs:
            for c in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[c],
                        mode='lines',
                        name=c,
                        **kwargs
                    )
                )
        return fig if is_return else fig.show()
    else:
        raise TypeError("Invalid parameter. Use 'sns' or 'go'.")


def plot_heat_sns(
    df: pd.DataFrame,
    title: str='Correlations',
    color: str='coolwarm',
    size: tuple=(12, 10),
    rotate: bool=False
) -> None:
    """Plot a heatmap of the correlations of df.

    Args:
        df (pd.DataFrame): dataframe.
        title (str, optional): plot title. Default: 'Correlations'.
        color (str, optional): cmap. Default: 'coolwarm'.
        size (tuple, optional): plot size. Default: (12, 10).
    """
    correlations = df.corr()

    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(
            correlations, mask=mask, annot=True,
            cmap=color, fmt='.2f', linewidths=0.05,
            vmax=1.0, square=True, linecolor='white'
        )

        if rotate:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel(None)


def plot_opt_comparisons(rets: dict, vols: dict, sharpes: dict, colors: dict) -> None:
    """Plot a go.Bar with the values of rets, vols, and sharpes.

    Args:
        rets (dict): dictionary with the optimized returns;
            Ex: {r_hrp: ..., r_min_vol: ...}.
        vols (dict): dictionary with the optimized volatilities;
            Ex: {vol_hrp: ..., vol_min_vol: ...}.
        sharpes (dict): dictionary with the optimized Sharpe indices;
            Ex: {sharpe_hrp: ..., sharpe_min_vol: ...}.
        colors (dict): dictionary regarding the colors of each bar:
            Ex: colors = {
                    'rets': r_colors,
                    'vols': vol_colors,
                    'sharpes': sharpe_colors
                },
        where the values are iterators containing the colors.
    """
    data = [
        go.Bar(
            x=rets.index,
            y=rets.values,
            marker={
                'color': colors['rets'],
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            opacity=.7,
            showlegend=False,
            text='R',
            hoverinfo='text+y'
        ),
        go.Bar(
            x=vols.index,
            y=vols.values,
            marker={
                'color': colors['vols'],
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            opacity=.7,
            showlegend=False,
            text='V',
            hoverinfo='text+y'
        ),
        go.Bar(
            x=sharpes.index,
            y=sharpes.values,
            marker={
                'color': colors['sharpes'],
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            opacity=.7,
            showlegend=False,
            text='S',
            hoverinfo='text+y'
        )
    ]

    cfg_layout = go.Layout(
        title='Optimized Results',
        xaxis=dict(
                title='Optimization',
                showgrid=False
            ),
            yaxis=dict(
                title='Recorded Value',
                showgrid=False
            ),
            plot_bgcolor="#FFF",
            hoverdistance=100
    )

    fig = go.Figure(data=data, layout=cfg_layout)
    fig.show()


def plot_weights(
    df: pd.DataFrame,
    titles: list=['Optimized Portfolios', 'Weights', None],
    plot_in: str='sns',
    size: tuple=(19,6),
    template: str='ggplot2',
    is_return: bool=False,
    **kwargs
):
    """Plots a horizontal barplot of df (intended to contain columns as weight vectors).

    Args:
        df (pd.DataFrame): dataframe of weights.
        titles (list, optional): plot title and labels.
        Default: ['Optimized Portfolios', 'Weights', None].
        plot_in (str, optional): 'sns' or 'go'. Default: 'sns'.
        size (tuple, optional): plot size. Default: (19,6).
        template (str, optional): plotly template; only valid if
        plot_in == 'go'. Default: 'ggplot2'.

    Raises:
        TypeError: if plot_in not in ('sns', 'go').
    """
    if plot_in == 'sns':
        df.plot(
            kind='barh', figsize=(19, 6),
            title=titles[0], **kwargs
        )
        plt.xlabel(titles[1])
        plt.ylabel(titles[2])
    elif plot_in == 'go':
        layout = layout_settings(titles)
        fig = go.Figure(layout=layout)

        for c in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df[c],
                    y=df.index,
                    name=c,
                    marker={
                        'line': {
                            'color': '#333',
                            'width': 2
                        }
                    },
                    hoverinfo='x',
                    orientation='h',
                    **kwargs
                )
            )

        fig.update_layout(template=template)
        return fig if is_return else fig.show()
    else:
        raise TypeError("Invalid parameter. Use 'sns' or 'go'.")

#-----------------------------------------------------------------
if __name__ == '__main__':
    main()
