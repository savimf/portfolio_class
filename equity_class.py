import pandas as pd
import numpy as np
import quant_tools as qt
from datetime import datetime as dt


class Equity():
    def __init__(
        self,
        prices: pd.DataFrame,
        init_allocations: dict,
        cryptos: list=[]
    ):
        self.prices = prices
        self.__cryptos = cryptos
        self.n_shares = self.share_qtd(init_allocations, check=True, avoid=cryptos)
        self.init_allocations = prices.iloc[0] * self.n_shares
        self.eq_history = {
            prices.index[0].strftime('%Y-%m-%d'): {
                a: (self.init_allocations[a], self.n_shares[a])
                for a in self.n_shares.keys()
            }
        }
        self.__total_equity = pd.DataFrame()


    @property
    def prices(self) -> pd.DataFrame:
        """Returns the prices property.

        Returns:
            pd.DataFrame.
        """
        return self.__prices


    @prices.setter
    def prices(self, new_prices: pd.DataFrame) -> None:
        """Assigns a new prices dataframe.

        Args:
            new_prices (pd.DataFrame): new prices dataframe.
        """
        if len(new_prices) == 0:
            raise ValueError(
                'The prices dataframe cannot be empty.'
            )
        self.__prices = new_prices


    @property
    def init_allocations(self) -> pd.Series:
        """Portfolio's initial allocation.

        Returns:
            pd.Series.
        """
        return self.__init_allocations


    @init_allocations.setter
    def init_allocations(self, new_allocations: dict) -> None:
        """Assigns new allocations.

        Args:
            new_allocations (dict): for instance, if one
            wishes to assign $ 100 in AAPL and $ 150 in
            GOOGL, then
                new_allocations = {
                    'AAPL': 100,
                    'GOOGL': 150
                }

        Raises:
            AttributeError: if new_allocations is an empty
            dictionary.
        """
        if len(new_allocations) > 0:
            self.__init_allocations = pd.Series(new_allocations)
        else:
            raise AttributeError(
                'new_allocations cannot be empty.'
            )


    @property
    def eq_history(self) -> dict:
        """Returns the allocations that have been made,
        along with their dates, in the form of a dictionary.
        Ex: if one has invested $ 100 in APPL and $ 150 in GOOGL
        in 2000-01-01, then

            {'2000-01-01': {'APPL': 100, 'GOOGL': 150}}


        The first date, however, is due to the initial allocation.

        Returns:
            dict.
        """
        return self.__eq_history


    @eq_history.setter
    def eq_history(self, new_history: dict) -> None:
        """Assigns a new eq_history.

        Args:
            new_history (dict): dictionary with
            keys corresponding to dates (YYYY-mm-dd)
            and values corresponding to a dictionary
            containing the allocations.
        """
        if len(new_history) == 0:
            raise ValueError(
                'new_history cannot be empty.'
            )
        self.__eq_history = new_history


    @property
    def n_shares(self) -> pd.Series:
        return self.__n_shares


    @n_shares.setter
    def n_shares(self, new_shares: pd.Series) -> None:
        if len(new_shares) == 0:
            raise ValueError(
                'new_shares cannot be empty.'
            )
        self.__n_shares = new_shares.astype('int32') if self.__cryptos == [] else new_shares


    def get_total_equity(self, total: bool=True):
        """Returns a pd.Series or pd.DataFrame with the
        equity's evolution.

        Args:
            total (bool, optional): if True, returns a
            pd.Series with the portfolio's total equity,
            if False, returns a pd.DataFrame with the assets'
            individual equities. Default: True.

        Returns:
            pd.Series or pd.DataFrame.
        """
        if len(self.eq_history) == 1:
            return self.__init_equity(total)

        return self.__total_equity.sum(axis=1) if total else self.__total_equity


    def __set_total_equity(self, new_equity) -> None:
        """Assigns a new total_equity.

        Args:
            new_equity (pd.Series or pd.DataFrame)
        """
        self.__total_equity = new_equity


    def share_qtd(self, allocations: dict, date: str=None, check: bool=False, avoid: list=[]) -> pd.Series:
        """Calculates the number of shares that can be bought/sold
        with 'allocations' in a given date.

        Args:
            allocations (dict): dictionary with the stocks to be bought/sold
            as keys and their corresponding investments as values.

                Ex: $ 100 in AAPL and $ 150 in GOOGL
                    {'AAPL': 100, 'GOOGL': 150}.

            date (str, optional): date in which the allocations are to be
            bought/sold. If None, first date is considered. Defaults: None.
            check (bool, optional): if True, will verify if the individual
            allocations are all sufficient o operate a share (see error).
            Default: False.

        Raises:
            AttributeError: if some individual allocation is insufficient to
            operate a share (only valid if check is True).

        Returns:
            pd.Series.
        """
        qtd = None
        alloc = pd.Series(allocations).abs()
        if date is None:
            qtd = (alloc / self.prices.iloc[0]).fillna(0)
        else:
            qtd = (alloc / self.prices.loc[date]).fillna(0)

        if check:
            to_check = [a for a in alloc.index if a not in avoid]
            convert = {
                qtd[a] : int(qtd[a])
                for a in to_check
            }
            qtd = qtd.replace(convert).reindex(index=self.prices.columns)

            if not all(qtd[a] for a in qtd.index if a in to_check):
                raise AttributeError(
                    'Insufficient allocation to operate at least one share ' \
                    'of each. Consider a higher investment.'
                )
        return qtd


    def __init_equity(self, total: bool=True):
        """Returns the portfolio's equity as a pd.Series or
        pd.DataFrame.

        Args:
            total (bool, optional): if True, returns a time series
            with the total equity (sum of the individuals). If False,
            returns a pd.DataFrame with the individual investments.
            Defaults: True.

        Returns:
            pd.Series or pd.DataFrame.
        """
        init_equity = self.prices * self.share_qtd(self.init_allocations)
        return init_equity.sum(axis=1) if total else init_equity


    def weights_track(self, weights_only: bool=True) -> pd.DataFrame:
        """Returns a pd.DataFrame with the evolution of the assets'
        weights in time.

        Args:
            weights_only (bool, optional): if True, returns only
            the evolution of the weights, if False, return the
            portfolio's equity as well. Default: True.

        Returns:
            pd.DataFrame.
        """
        df = self.get_total_equity(total=False).copy()
        for c in df.columns:
            df[f'weight_{c}'] = df[c] / df.sum(axis=1)

        if weights_only:
            df = df.iloc[:, self.prices.shape[1]:]
            return df
        return df


    def weighted_return(self) -> pd.Series:
        """Returns the equity weighted return. Consider the
        assets' individual returns, given the initial investment,
        along with the evolution of their weights.

        Returns:
            pd.Series.
        """
        asset_rets = self.__init_equity(total=False).pct_change()
        weights = self.weights_track().copy()
        weights.columns = asset_rets.columns

        w_ret = weights.mul(asset_rets).dropna()
        return w_ret.sum(axis=1)


    def eq_return(self, total: bool=True) -> pd.Series:
        """Calculates the equity return.

        Args:
            total (bool, optional): if True, gives the
            total equity return, as a weighted average of
            the individual returns, if False, gives the individual
            returns. Default: True.

        Returns:
            pd.Series.
        """
        r = 0
        if len(self.eq_history) == 1:
            r = self.get_total_equity(total).pct_change()
        else:
            r = self.weighted_return() if total \
                else self.__init_equity(total).pct_change()
        return r.dropna()


    def operation(self, allocations: dict, date: str, sell: bool=False) -> None:
        """Realizes a portfolio operation, considering
        the given allocations and asset prices at the date
        informed.

        Args:
            allocations (dict): dicionary with the stocks to be bought/sold
            as keys and its corresponding investment as values.

                Ex: $ 100 in APPL and $ 150 in GOOGL
                    {'AAPL': 100, 'GOOGL': 150}.

            date (str): date in which the allocations are to be
            bought.
            sell (bool, optional): if True, the operation refers to a purchase,
            if False, refers to a sale.

        Raises:
            ValueError: if len(allocations) == 0.
            KeyError: if date does not correspond to a business day.
        """
        if len(allocations) == 0:
            raise ValueError(
                'Allocations must not be empty.'
            )

        shares = None
        try:
            shares = self.share_qtd(allocations, date, check=True, avoid=self.__cryptos)
        except KeyError:
            raise KeyError(
                'Date is not a business day: unable to execute operation.'
            ) from None

        allocations_ = (self.prices.loc[date] * shares).round(2)
        if sell:
            if any(shares > self.n_shares):
                raise AttributeError(
                    'Number of shares is exceeding the current value. '\
                    'Sell operation has failed.'
                )

            current_eq = self.prices.loc[date] * self.n_shares
            profit = Equity.profit(current_eq, self.avg_prices())
            for a in pd.Series(allocations).index:
                print(
                    f'Profit {a}: {profit[a] * 100:.2f}%'
                )
            allocations_ = - allocations_

        previous_equity = self.get_total_equity(total=False)
        self.eq_history[date] = {
            a: (allocations_[a], shares[a])
            for a in shares.keys()
        }

        df = self.prices.copy()
        new_equity = (df * shares).fillna(0) if not sell else -(df * shares).fillna(0)
        new_equity[new_equity.index < date] = 0

        full_equity = pd.concat(
            [previous_equity, new_equity],
            axis=1
        ).groupby(lambda x: x, axis=1).sum()
        full_equity = full_equity.reindex_like(df)

        self.__set_total_equity(full_equity)

        if not sell:
            self.n_shares += shares
        else:
            self.n_shares -= shares


    def avg_prices(self) -> dict:
        """Returns a dictionary with the average prices of
        the assets composing the portfolio, with the assets
        as keys and the prices as values.

        Returns:
            dict.
        """
        avgs = {}
        hist = self.eq_history

        for a in self.prices.columns:
            prices = []
            shares = []
            dates = []
            for k in hist.keys():
                if hist[k][a][0] <= 0:
                    continue

                prices.append(hist[k][a][0])
                shares.append(hist[k][a][1])
                dates.append(k)

            prices = np.array(prices)
            shares = np.array(shares)
            avgs[a] = prices.dot(shares) / shares.sum()
        return avgs


    def avg_track(self) -> pd.DataFrame:
        """Returns a pd.DataFrame with the evolution of the
        average price of the assets composing the portfolio.
        The average changes as purchases are made.

        Returns:
            pd.DataFrame.
        """
        avgs = {}
        hist = self.eq_history

        for a in self.prices.columns:
            prices = []
            shares = []
            avg = {}
            for k in hist.keys():
                if hist[k][a][0] <= 0:
                    avg[k] = avg_
                    continue

                prices.append(hist[k][a][0])
                shares.append(hist[k][a][1])

                prices = np.array(prices)
                shares = np.array(shares)
                avg_ = prices.dot(shares) / shares.sum()
                avg[k] = avg_
                prices = list(prices)
                shares = list(shares)

            avgs[a] = avg

        first_date = list(hist.keys())[0]
        first_avgs = {
            a: np.repeat(avgs[a][first_date], len(self.prices))
            for a in avgs.keys()
        }

        df_avgs = pd.DataFrame(
            data=first_avgs,
            index=self.prices.index
        )

        for a in avgs.keys():
            for d in list(hist.keys())[1:]:
                df_avgs.loc[df_avgs.index >= d, a] = avgs[a][d]

        return df_avgs


    @staticmethod
    def profit(current_eq: pd.Series, avgs: dict) -> pd.Series:
        """Calculates the profit of a sale, considering the
        current equity at some date and the average asset prices.

        Args:
            current_eq (pd.Series): current equity
            avgs (dict): dictionary of average prices
            regarding the portfolio assets, in the form
            {'asset_A': 1500, 'asset_B': 1000}

        Returns:
            pd.Series.
        """
        avgs = pd.Series(avgs)
        return (current_eq - avgs) / avgs
