import pandas as pd
import numpy as np
import pypfopt as pf
import quant_tools as qt
import seaborn as sns
import matplotlib.pyplot as plt
from equity_class import Equity
from datetime import datetime as dt, timedelta
from scipy.stats import skew, kurtosis, shapiro
from copy import deepcopy
from functools import wraps
import inspect


def get_default_args(f) -> dict:
    """Identify, in the form of a dictionary, the default
    arguments of the function f.

    Args:
        f (function): function to be analyzed.

    Returns:
        dict: {'arg1': value1, 'arg2': value2}
    """
    signature = inspect.signature(f)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


# risk-free rate default value
rf = .03
class Portfolio():
    # dictionary to store Portfolios (helps when comparing their metrics)
    registered = {}

    # tolerance to verify if the weights sum to 1
    delta = .001

    # if True, one can overwrite a Portfolio's name
    overwrite = False

    def __init__(
        self,
        name: str,
        assets: list,
        init_inv: float=10_000.,
        start: dt=None, end: dt=None,
        source: str='iv',
        crypto: bool=False,
        **kwargs
    ):
        self.name = name
        self.assets = assets
        self.init_inv = float(init_inv)
        self.dates = (start, end)
        self.equity = None
        self.__gen_eq = False
        self._cryptos = [] if not crypto else assets
        self.prices = qt.wallet(
            assets,
            start, end,
            source,
            crypto,
            **kwargs
        )
        self.init_weights = np.repeat(1/len(assets), len(assets))



    def __str__(self) -> str:
        """
        Returns:
            str: Portfolio's name.
        """
        return self.__name


    def __len__(self) -> int:
        """Returns the number of assets comprising the
        Portfolio.

        Returns:
            int.
        """
        return len(self.__assets)


    def __add__(self, second_p, join: str='inner'):
        """Method that allows Portfolios addition by concatenating
        their prices. The resulting Portfolio will receive:
            - name: 'self.name + second_p.name'
            - assets: columns of the concatenated df
            - prices: self.prices concatenated with second_p.prices
            - dates: first and last dates of the concatenated df

        Args:
            second_p (Portfolio): Portfolio to be added.
            join (str, optional): concatenate method. To change it,
            one must use
            self.__add__(second_p, join). Default: 'inner'.

        Raises:
            AttributeError: if isinstance(second_p, Portfolio)
            == False.

        Returns:
            Portfolio.
        """
        if not isinstance(second_p, Portfolio):
            raise AttributeError('Portfolio must be added with another Portfolio.')

        result_p_prices = pd.concat(
            [self.prices, second_p.prices],
            axis=1,
            join=join
        )
        result_p_name = self.name + ' + ' + second_p.name
        result_p_assets = result_p_prices.columns
        x = result_p_prices.iloc[0].max() * result_p_prices.shape[1] * 1.05
        init_inv = x if x > self.init_inv else self.init_inv

        result_p = Portfolio(result_p_name, result_p_assets, init_inv)
        result_p.prices = result_p_prices

        result_p.dates = (result_p.prices.index[0], result_p.prices.index[-1])

        return result_p


    @classmethod
    def register(cls, portfolio) -> None:
        """Adds a Portfolio in the 'registered' dictionary,
        with portfolio.name as key and portfolio as value.

        Args:
            portfolio (Portfolio): Portfolio instance.
        """
        cls.registered[portfolio.name] = portfolio


    @classmethod
    def unregister(cls, name: str=None, is_all: bool=False) -> None:
        """Removes the Portfolio whose name property is 'name' from
        'registered'. If is_all == True, all names are removed.

        Args:
            name (str, optional): Portfolio's name. Default: None.
            is_all (bool, optional) if True, applies clear() onto
            'registered'.

        Raises:
            NameError: if name is None and is_all == False.
        """
        if is_all:
            cls.registered.clear()
        else:
            if name is None:
                raise NameError('Please insert a name.')
            del cls.registered[name]


    @property
    def name(self) -> str:
        """Portfolio's name.

        Returns:
            str.
        """
        return self.__name


    @name.setter
    def name(self, new_name: str) -> None:
        """Assigns a new name to the Portfolio and automatically
        register it.

        Args:
            new_name (str): Portfolio's new name.

        Raises:
            ValueError: if len(new_name) == 0.
            NameError: if new_name in Portfolio.registered.keys()
            (only valid if overwrite == False).
        """
        if len(new_name) == 0:
            raise ValueError('New name cannot be empty.')

        if len(Portfolio.registered) > 0:
            if len(new_name) == 0:
                raise ValueError('New name cannot be empty.')
            elif new_name in Portfolio.registered.keys():
                if Portfolio.overwrite:
                    print('Name overwritten.')
                    pass
                else:
                    raise NameError('There is already a Portfolio with this name.')

        self.__name = new_name
        Portfolio.register(self)


    @property
    def assets(self) -> list:
        """Portfolio's assets list.

        Returns:
            list.
        """
        return list(self.__assets)


    @assets.setter
    def assets(self, new_assets: list) -> None:
        """Assigns new assets to the Portfolio.
        (NOT RECOMMENDED!)

        Args:
            new_assets (list): ['asset1', 'asset2'...]

        Raises:
            ValueError: if len(new_assets) == 0.
        """
        if len(new_assets) == 0:
            raise ValueError('Minimum of 1 asset is required.')

        self.__assets = new_assets


    @property
    def init_inv(self) -> float:
        """Returns the init_inv property.

        Returns:
            float.
        """
        return self.__init_inv


    @init_inv.setter
    def init_inv(self, new_inv: float) -> None:
        """Assigns a new initial investment

        Args:
            new_inv (float): new initial investment.

        Raises:
            AttributeError: if new_inv < 0.
        """
        if new_inv > 0:
            self.__init_inv = float(new_inv)
        else:
            raise AttributeError(
                'Initial investment must be greater than 0.'
            )


    @property
    def init_weights(self) -> np.ndarray:
        """Weight distribution among the Portfolio's assets.

        Returns:
            np.ndarray.
        """
        return self.__init_weights


    @init_weights.setter
    def init_weights(self, new_weights: np.ndarray) -> None:
        """Assigns new weights to the Portfolio. If it contains only
        one asset, no change will take affect. The new weights must
        sum to 1 (with Portfolio.delta as tolerance). Portfolio.registered
        and the equity attribute are automatically updated.

        Args:
            new_weights (np.ndarray): np.array([w1, w2,...])

        Raises:
            AttributeError: if np.abs(1 - np.sum(new_weights)) >
            Portfolio.delta.
        """
        if len(self.assets) == 1:
            new_weights = np.array([1])
        elif np.abs(1 - np.sum(new_weights)) > Portfolio.delta:
            raise AttributeError('Weights must sum to 1.')

        self.__init_weights = new_weights

        if len(self.prices) > 0 and self.__gen_eq:
            init_alloc = {
                    p[0]: self.init_inv * p[1]
                    for p in zip(self.assets, self.init_weights)
                }
            self.equity = Equity(
                self.prices,
                init_alloc,
                cryptos=self._cryptos
            )
        Portfolio.register(self)


    @property
    def dates(self) -> tuple:
        """Initial and final dates of the prices dataframe.

        Returns:
            tuple: (start, end)
        """
        return self.__dates


    @dates.setter
    def dates(self, new_dates: tuple) -> None:
        """Assigns new dates to the Portfolio.
        (NOT RECOMMENDED!)

        Args:
            new_dates (tuple): (start, end).

        Raises:
            ValueError: if only one date is given.
        """
        check = sum(1 for d in new_dates if isinstance(d, dt))
        if check == 2:
            self.__dates = new_dates
        elif check == 1:
            raise ValueError('Please insert both dates.')
        else:
            self.__dates = (None, None)


    @property
    def prices(self) -> pd.DataFrame:
        """Portfolio's price dataframe.

        Returns:
            pd.DataFrame.
        """
        return self.__prices


    @prices.setter
    def prices(self, new_prices: pd.DataFrame) -> None:
        """Assigns a new price dataframe to the Portfolio.
        The equity attribute is automatically updated.
        (NOT RECOMMENDED!)

        Args:
            new_prices (pd.DataFrame).
        """
        self.__prices = new_prices.round(2)
        if len(new_prices) > 0 and self.__gen_eq:
            init_alloc = {
                    p[0]: self.init_inv * p[1]
                    for p in zip(self.assets, self.init_weights)
                }
            self.equity = Equity(
                self.prices,
                init_alloc,
                cryptos=self._cryptos
            )
        self.__gen_eq = True


    def __check(arg_name: str, possible_values: tuple):
        """Decorating function designated to verify a function's
        default arguments. Raises an error if 'arg_name' not in
        'possible_values'.

        Args:
            arg_name (str): name of the default argument.
            possible_values (tuple): possible values it can assume.
        """
        def check_inner(f):
            @wraps(f)
            def check(*args, **kwargs):
                p = get_default_args(f)
                p.update(kwargs)

                if p[arg_name] not in possible_values:
                    raise KeyError(f"{arg_name} invalid. Use {possible_values}.")
                return f(*args, **kwargs)
            return check
        return check_inner


    def d_returns(self, is_portfolio: bool=True):
        """Daily returns.

        Args:
            is_portfolio (bool, optional): if True, yields the return
            of the portfolio, if False, yields the individual return
            of the assets. Default: True.

        Returns:
            pd.Series or pd.DataFrame.
        """
        return self.equity.eq_return(total=is_portfolio)


    def m_returns(self, is_portfolio: bool=True):
        """Monthly returns.

        Args:
            is_portfolio (bool, optional): if True, yields the return
            of the portfolio, if False, yields the individual return
            of the assets. Default: True.

        Returns:
            pd.Series or pd.DataFrame.
        """
        d_rets = self.d_returns(is_portfolio=is_portfolio)

        m_rets = d_rets.resample('M').apply(
            qt.compound
        ).to_period('M')
        return m_rets


    def a_returns(self, is_portfolio: bool=True):
        """Annual returns.

        Args:
            is_portfolio (bool, optional): if True, yields the return
            of the portfolio, if False, yields the individual return
            of the assets. Default: True.

        Returns:
            pd.Series or pd.DataFrame.
        """
        d_rets = self.d_returns(is_portfolio)

        a_rets = d_rets.resample('Y').apply(
            qt.compound
        ).to_period('Y')
        return a_rets


    def acm_returns(self, is_portfolio: bool=True):
        """Cumulative returns.

        Args:
            is_portfolio (bool, optional): if True, yields the return
            of the portfolio, if False, yields the individual return
            of the assets. Default: True.

        Returns:
            pd.Series or pd.DataFrame.
        """
        acm = (1 + self.d_returns(is_portfolio=is_portfolio)).cumprod()
        return acm.dropna()


    def __total_returns(self, is_portfolio: bool=False, *, period: str='a'):
        """Total variation: (init_price - last_price) / init_price.

        Args:
            period (str, optional):
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual
            Default: 'd'.

        Returns:
            pd.Series.
        """
        acm = self.acm_returns(is_portfolio=is_portfolio)
        n_years = acm.shape[0] / 252
        r = acm.iloc[-1] - 1

        if period == 'm':
            return (1 + r) ** (1 / (12 * n_years)) - 1
        elif period == 'a':
            return (1 + r) ** (1 / n_years) - 1
        return r


    @__check('period', ('m', 'a', None))
    def asset_returns(self, *, period: str='a') -> pd.Series:
        """Returns a pd.Series with the annual/monthly/total
        return of the assets composing the portfolio.

        Args:
            period (str, optional):
                - 'a' for annual
                - 'm' for monthly
                - None for total.
            Default: 'a'.

        Returns:
            pd.Series.
        """
        r = self.__total_returns(is_portfolio=False, period=period)
        r.name = None
        return r


    @__check('period', ('m', 'a', None))
    def portfolio_return(self, *, period: str='a') -> float:
        """Gives the portfolio's return.

        Args:
            period (str, optional):
                - 'a' for annual
                - 'm' for monthly
                - None for total.
            Default: 'a'.

        Returns:
            float.
        """
        return self.__total_returns(is_portfolio=True, period=period)


    def sample_cov(self) -> pd.DataFrame:
        """Returns the sample covariance of the assets composing
        the portfolio.

        Returns:
            pd.DataFrame.
        """
        return self.d_returns(is_portfolio=False).cov()


    @__check('plot_in', ('sns', 'go'))
    def benchmark(
        self,
        portfolios: list,
        titles: list=['Benchmark', 'Date', 'Factor'],
        plot_in: str='sns',
        size: tuple=(19, 6),
        name: str=None,
        is_return: bool=False,
        **kwargs
    ):
        """Plots a benchmark with the Portfolio calling this method
        with the Portfolios in 'portfolios'.

        Args:
            portfolios (list): list of Portfolios.
            plot_in (str, optional): 'sns' for seaborn or 'go'
            for plotly. Default: 'sns'.
            fsize (tuple, optional): plot size. Default: (19, 6).

        Raises:
            ValueError: if len(portfolios) == 0.
            TypeError: if any element in 'portfolios' is not a Portfolio.
        """
        if len(portfolios) == 0:
            raise ValueError('Please insert at least one Portfolio.')

        check = sum(1 for p in portfolios if isinstance(p, Portfolio))
        if check != len(portfolios):
            raise TypeError('Please insert only Portfolios.')

        bench = self.acm_returns().to_frame().rename(
            columns={0: self.name}
        )
        for p in portfolios:
            bench = pd.concat(
                [
                    bench,
                    p.acm_returns().to_frame().rename(
                        columns={0: p.name}
                    )
                ],
                axis=1,
                join='inner'
            )
        return qt.plot_lines(
            dfs=[bench], titles=titles,
            plot_in=plot_in, name=name,
            is_return=is_return, **kwargs
        )


    def beta(self, benchmark) -> float:
        """Returns the beta of the Portfolio calling this method with
        the Portfolio passed as benchmark.

        Args:
            benchmark (Portfolio): benchmark Portfolio.

        Raises:
            TypeError: if not isinstance(benchmark, Portfolio)

        Returns:
            float.
        """
        if not isinstance(benchmark, Portfolio):
            raise TypeError('Please insert a Portfolio to serve as benchmark.')

        ret_port = self.d_returns()
        ret_bench = benchmark.d_returns()

        return qt.beta(ret_port, ret_bench)


    def volatility(self, is_portfolio: bool=True, tracking: bool=False, measure: str='median'):
        """Returns the daily, monthly and annual volatility.

        Args:
            is_portfolio (bool, optional): if False, calculate
            the individual volatilities of the assets composing
            the portfolio, through the standard deviation of the
            daily returns. if True, returns the portfolio's volatility,
            considering its weights and its sample covariance.
            Default: True.

        Returns:
            pd.Series or pd.DataFrame,
        """
        def vol_m(v: pd.Series, m: str) -> float:
            if m not in ('mean', 'median'):
                raise ValueError('Invalid measure. Use mean or median.')
            return v.median() if m == 'median' else v.mean()


        if not is_portfolio:
            return pd.DataFrame(
                data=map(
                    lambda p: self.d_returns(is_portfolio=False).std() * np.sqrt(p), (1, 21, 252)
                ),
                index=['Daily', 'Monthly', 'Annual']
            )

        cov = self.sample_cov()
        vol_track = self.equity.weights_track().T.apply(
            lambda w: qt.vol(w, cov, annual=False)
        )

        if tracking:
            return vol_track

        return pd.Series(
            data=map(
                lambda p: vol_m(vol_track * np.sqrt(p), measure),
                (1, 21, 252)
            ),
            index=['Daily', 'Monthly', 'Annual']
        )


    @__check('which', ('sharpe', 'sortino'))
    def s_index(self, risk_free_rate: float=rf, which: str='sharpe') -> float:
        """Returns the annualized Sharpe or Sortino index.

        Args:
            risk_free_rate (float, optional): risk-free rate.
            Default: 0.03.
            which (str, optional): index to be returned
                - 'sharpe'
                - 'sortino'
            Default: 'sharpe'.

        Returns:
            float.
        """
        ret = self.portfolio_return()
        vols = {'sharpe': self.volatility().loc['Annual'], 'sortino': self.downside()}

        return qt.sharpe(ret, vols[which], risk_free_rate)


    @__check('which', (95, 97, 99, 99.9, None))
    def var(
        self, *,
        which: int=None, kind: str='hist',
        period: str='d', is_neg: bool=True,
        modified: bool=False
    ):
        """Returns a pd.Series with the historical or parametric
        VaRs (95, 97, 99 and 99.9), or only one of them, chosen by
        'which'. The parameters must all be named.

        Args:
            which (int, optional): VaR to be returned
                - 95
                - 97
                - 99
                - 99.9
            Default: None.

            kind (str, optional): VaR to be computed
                - historical (hist)
                - parametric (param)
            Default: 'hist'.

            period (str, optional):
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual
            Default: 'd'.

            is_neg (bool, optional): whether to return positive or
            negative values. Default: True.
            modified (bool, optional): valid only for kind='param';
            if True, considers the skewness and kurtosis of the distribution
            and realize the Cornish-Fisher correction.

        Raises:
            KeyError: if period not in ('d', 'm', 'a').

        Returns:
            pd.Series or float.
        """
        if period not in ('d', 'm', 'a'):
            raise KeyError("Invalid period: use 'd', 'm' or 'a'.")

        d_period = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }

        if kind == 'hist':
            var = pd.Series(qt.vars_hist(d_period[period]()))
        elif kind == 'param':
            var = pd.Series(qt.vars_gaussian(d_period[period](), modified))
        else:
            raise IndexError("Invalid VaR: use 'hist' or 'param'.")


        if not is_neg:
            var = -var

        if not which:
            return var
        return var.loc[which]


    @__check('which', (95, 97, 99, 99.9, None))
    def cvar(
        self, *,
        which: int=None, kind: str='hist',
        period: str='d', is_neg: bool=True,
        modified: bool=False
    ):
        """Returns a pd.Series with the historical or parametric
        CVaRs (95, 97, 99 and 99.9), or only one of them, chosen by
        'which'. The parameters must all be named.

        Args:
            which (int, optional): VaR to be returned
                - 95
                - 97
                - 99
                - 99.9
            Default: None.

            kind (str, optional): VaR to be computed
                - historical (hist)
                - parametric (param)
            Default: 'hist'.

            period (str, optional):
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual
            Default: 'd'.

            is_neg (bool, optional): whether to return positive or
            negative values. Default: True.
            modified (bool, optional): valid only for kind='param';
            if True, considers the skewness and kurtosis of the distribution
            and realizes the Cornish-Fisher correction.

        Raises:
            KeyError: if period not in ('d', 'm', 'a').

        Returns:
            pd.Series or float.
        """
        if period not in ('d', 'm', 'a'):
            raise KeyError("Invalid period: use 'd', 'm' or 'a'.")

        d_period = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }

        if kind == 'hist':
            cvar = pd.Series(qt.cvars_hist(d_period[period]()))
        elif kind == 'param':
            cvar = pd.Series(qt.cvars_gaussian(d_period[period](), modified=modified))
        else:
            raise IndexError("Invalid CVaR: use 'hist' or 'param'.")


        if not is_neg:
            cvar = -cvar

        if not which:
            return cvar
        return cvar.loc[which]


    def all_vars(self, period: str='d', is_neg: bool=False) -> pd.DataFrame:
        """Returns a pd.DataFrame with the historical, parametric and the
        parametric adjusted VaRs.

        Args:
            period (str, optional):
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual
            Default: 'd'.

            is_neg (bool, optional): whether to return positive or
            negative values. Default: True.

        Returns:
            pd.DataFrame.
        """
        return pd.DataFrame(
            {'Hist': self.var(period=period, is_neg=is_neg),
             'Parametric': self.var(period=period, kind='param', is_neg=is_neg),
             'Parametric_Adj': self.var(period=period, kind='param', is_neg=is_neg, modified=True)}
        )


    @__check('period', ('d', 'm', 'a'))
    def downside(self, period: str='a') -> float:
        """Returns the portfolio downside (standard deviation of the
        negative returns).

        Args:
            period (str, optional):
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual
            Default: 'd'.

        Returns:
            float.
        """
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets < 0].std() * np.sqrt(factor[period])


    @__check('period', ('d', 'm', 'a'))
    def upside(self, period: str='a') -> float:
        """Returns the portfolio upside (standard deviation of the
        positive returns).

        Args:
            period (str, optional):
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual
            Default: 'd'.

        Returns:
            float.
        """
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets > 0].std() * np.sqrt(factor[period])


    def rol_drawdown(self, window: int=21, is_number: bool=True):
        """Returns the rolling maximum drawdown within the period
        'window'.

        Args:
            window (int, optional): Time window of interest. Default: 21.
            is_number (bool, optional): if True, returns the maximum drawdown,
            if False, returns a pd.DataFrame with the rolling drawdowns.
            Default: True.

        Returns:
            float ou pd.DataFrame
        """
        acm = self.acm_returns()
        rol_max = acm.rolling(window=window).max()
        drawdown_ = acm / rol_max - 1
        max_drawdown = drawdown_.rolling(window=window).min()

        if is_number:
            return max_drawdown.min()
        return max_drawdown.dropna()


    @__check('period', ('d', 'm', 'a'))
    def calc_skewness(self, period: str='d') -> float:
        """Given the return distribution, returns its skewness.

        Args:
            period (str, optional): distribution to be considered.
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual.
            Default: 'd'.

        Returns:
            float.
        """
        r = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }
        return skew(r[period]())


    @__check('period', ('d', 'm', 'a'))
    def calc_kurtosis(self, is_excess: bool=True, period: str='d') -> float:
        """Given the return distribution, returns its kurtosis.

        Args:
            is_excess (bool, optional): if True, returns kurtosis - 3.
            period (str, optional): distribution to be considered.
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual.
            Default: 'd'.

        Returns:
            float.
        """
        r = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }

        if is_excess:
            return kurtosis(r[period]()) - 3
        return kurtosis(r[period]())


    @__check('period', ('d', 'm', 'a'))
    def shapiro_test(self, period: str='d', confidence: float=.05) -> bool:
        """Verify, within a given confidence interval, whether the
        return distribution assumes a normal distribution.

        Args:
            period (str, optional): distribution to be considered.
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual.
            Default: 'd'.
            confidence (float, optional): confidence interval.
            Default: 0.05.

        Returns:
            bool.
        """
        rets = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }
        if shapiro(rets[period]())[1] < confidence:
            return False
        return True


    def metrics(
        self,
        risk_free_rate: float=rf,
        window: int=21,
        period: str='d',
        benchmark=None
    ) -> pd.DataFrame:
        """Returns a pd.DataFrame with a collection of metrics.

        Args:
            risk_free_rate (float, optional): risk-free rate. Default: 0.03.
            window (int, optional): time window for computing the drawdown.
            Default: 21.
            period (str, optional): distribution to be considered.
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual.
            Default: 'd'.
            benchmark (Portfolio, optional): benchmark to compute the beta.
            Default: None.

        Returns:
            pd.DataFrame.
        """
        dict_metrics = {
            'Ann_Return': self.portfolio_return(),
            'Ann_Volatility': self.volatility().loc['Annual'],
            'Sharpe_Ind': self.s_index(risk_free_rate),
            'Sortino_Ind.': self.s_index(risk_free_rate, 'sortino'),
            f'Skewness ({period})': self.calc_skewness(period=period),
            f'Ex_Kurtosis ({period})': self.calc_kurtosis(period=period),
            f'VaR_99.9 ({period})': self.var(which=99.9, period=period, is_neg=False),
            f'CVaR_99.9 ({period})': self.cvar(which=99.9, period=period),
            f'Max_Drawdown ({window})': self.rol_drawdown(window),
            'Downside': self.downside(),
            'Upside': self.upside(),
            f'Normal ({period})': self.shapiro_test(period)
        }

        df_metrics = pd.DataFrame.from_dict(
            dict_metrics,
            orient='index',
            columns=[self.name]
        )

        if isinstance(benchmark, Portfolio):
            df_metrics = df_metrics.T
            df_metrics.insert(4, 'Beta', self.beta(benchmark))
            df_metrics = df_metrics.T

        return df_metrics


    def transfer(self, new_name: str, new_weights: np.ndarray):
        """Method that transfers the data from the original Portfolio,
        as the assets, dates and prices, to a new Portfolio, whose name
        and weights will be new_name and new_weights, respectively.

        Args:
            new_name (str): name of the new Portfolio.
            new_weights (np.array): weights of the new Portfolio.

        Returns:
            Portfolio.
        """
        new_assets = pd.Series(
            data=np.round(new_weights, 5),
            index=self.assets
        )
        new_assets = new_assets[new_assets > 0]
        new_prices = self.prices[new_assets.index]

        x = new_prices.iloc[0].max() * new_prices.shape[1] * 1.05
        new_init_inv = x if x > self.init_inv else self.init_inv
        new_init_weights = new_weights[np.round(new_weights, 5) > 0]

        new_p = Portfolio(
            new_name,
            list(new_assets.index),
            new_init_inv
        )

        new_p.prices = new_prices
        new_p.init_weights = new_init_weights
        new_p.dates = self.dates
        return new_p


    @classmethod
    def all_rets(cls, period: str='a') -> pd.Series:
        """Returns a pd.Series with the returns from all registered
        Portfolios.

        Args:
            period (str, optional): refers to which period the returns
            must be periodized
                - 'm' for monthly
                - 'a' for annual
            Default: 'a'.

        Raises:
            NotImplementedError: if len(Portfolio.registered) == 0.

        Returns:
            pd.Series.
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.portfolio_return(period=period)
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('No registered Portfolios.')


    @classmethod
    def all_vols(cls, period: str='a') -> pd.Series:
        """Returns a pd.Series with the volatilities from all registered
        Portfolios.

        Args:
            period (str, optional): refers to which period the volatilities
            must be periodized
                - 'm' for monthly
                - 'a' for annual
            Default: 'a'.

        Raises:
            NotImplementedError: if len(Portfolio.registered) == 0.

        Returns:
            pd.Series.
        """
        d_per = {
            'd': 'Daily',
            'm': 'Monthly',
            'a': 'Annual'
        }
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.volatility().loc[d_per[period]]
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('No registered Portfolios.')


    @classmethod
    @__check('which', ('sharpe', 'sortino'))
    def all_sindex(cls, risk_free_rate: float=rf, *, which='sharpe') -> pd.Series:
        """Returns a pd.Series with the annualized Sharpe, or Sortino,
        index, from all registered Portfolios.

        Args:
            risk_free_rate (float, optional): risk-free rate. Default: 0.03.
            which (str, optional): index to be returned
                - 'sharpe'
                - 'sortino'
            Default: 'sharpe'.

        Raises:
            NotImplementedError: if len(Portfolio.registered) == 0

        Returns:
            pd.Series.
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.s_index(risk_free_rate, which=which)
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_weights(cls) -> pd.Series:
        """Returns a pd.Series with the weights from the registered
        Portfolios.

        Raises:
            NotImplementedError: if len(Portfolio.registered) == 0

        Returns:
            pd.Series.
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                n: p.init_weights
                for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('No registered Portfolios.')


    @classmethod
    def all_metrics(
        cls,
        portfolios: list=[],
        risk_free_rate: float=rf,
        window: int=21,
        period: str='d',
        benchmark=None
    ) -> pd.DataFrame:
        """Applies p.metrics() for all p in portfolios and consolidates it
        in a single pd.DataFrame.

        Args:
            portfolios (list, optional): list of Portfolio's. Default: [].
            risk_free_rate (float, optional): risk-free rate. Default: 0.03.
            window (int, optional): time window for computing the drawdown.
            Default: 21.
            period (str, optional): distribution to be considered.
                - 'd' for daily
                - 'm' for monthly
                - 'a' for annual.
            Default: 'd'.
            benchmark (Portfolio, optional): benchmark to compute the beta.
            Default: None.

        Raises:
            ValueError: if len(portfolios) == 0.
            AttributeError: if the portfolios list is not composed of
            Portfolio objects only.
            NotImplementedError: if len(Portfolio.registered) == 0.

        Returns:
            pd.DataFrame.
        """
        if len(cls.registered) > 0:
            if len(portfolios) == 0:
                raise ValueError('Please insert at least one Portfolio.')

            check = sum(1 for p in portfolios if isinstance(p, Portfolio))
            if check != len(portfolios):
                raise AttributeError("Please insert Portfolio's only.")


            df = portfolios[0].metrics(risk_free_rate, window, period, benchmark)
            for p in portfolios[1:]:
                df_ = p.metrics(risk_free_rate, window, period, benchmark)
                df = pd.concat(
                    [df, df_],
                    axis=1,
                    join='inner'
                )
            return df

        raise NotImplementedError('No registered Portfolios.')
