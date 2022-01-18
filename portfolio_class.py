import pandas as pd
import numpy as np
import pypfopt as pf
import quant_tools as qt
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
from scipy.stats import skew, kurtosis, shapiro
from copy import deepcopy
from functools import wraps
import inspect


def get_default_args(f) -> dict:
    """Identifica, em forma de dicionário, os argumentos
    default da função f.

    Args:
        f (function): função a ser analisada.

    Returns:
        dict: dicionário no formato
        {'arg1': value1, 'arg2': value2}
    """
    signature = inspect.signature(f)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


rf = .03
class Portfolio():
    # dicionário para armazenar os portfólios (facilita na comparação das métricas)
    registered = {}

    # tolerância para verificar se a soma de pesos é igual a 1
    delta = .001

    # se True, permite a sobrescrita dos nomes
    overwrite = False

    def __init__(
        self,
        name: str,
        tickers: list,
        start: dt=None, end: dt=None,
        source: str='iv',
        crypto: bool=False
    ):
        self.name = name
        self.tickers = tickers
        self.weights = np.repeat(1/len(self.tickers), len(self.tickers))
        self.dates = (start, end)
        self.prices = qt.carteira(
            self.tickers,
            self.dates[0],
            self.dates[1],
            source,
            crypto
        )


    def __str__(self) -> str:
        """Método mágico str.

        Returns:
            str: retorna o nome do Portfolio.
        """
        return self.__name


    def __len__(self) -> tuple:
        """Método mágico len.

        Returns:
            tuple: retorna o número de tickers.
        """
        return len(self.__tickers)


    def __add__(self, second_p, join: str='inner'):
        """Método mágico que possibilita a soma de dois
        objetos da classe Portfolio, pela concatenação
        dos preços de ambos. O Portfolio resultante re-
        ceberá:
            - nome: 'self.name + second_p.name'
            - tickers: colunas do df concatenado
            - prices: concatenação de self.prices com
            second_p.prices
            - dates: primeira e última data do df
            concatenado

        Args:
            second_p (Portfolio): Portfolio a ser somado.
            join (str, optional): tipo da concatenação.
            Para alterá-la, é necessário usar
            self.__add__(second_p, join). Padrão: 'inner'.

        Raises:
            AttributeError: se isinstance(second_p, Portfolio)
            == False.

        Returns:
            Portfolio
        """
        if not isinstance(second_p, Portfolio):
            raise AttributeError('Portfolio deve ser somado com outro Portfolio.')

        result_p_prices = pd.concat(
            [self.prices, second_p.prices],
            axis=1,
            join=join
        )
        result_p_name = self.name + ' + ' + second_p.name
        result_p_tickers = result_p_prices.columns

        result_p = Portfolio(result_p_name, result_p_tickers)
        result_p.prices = result_p_prices
        result_p.dates = (result_p.prices.index[0], result_p.prices.index[-1])

        return result_p


    @classmethod
    def register(cls, portfolio) -> None:
        """Adiciona o Portfólio no dicionário registered,
        sendo portfolio.name a chave e portfolio o valor.

        Args:
            portfolio (Portfolio): objeto da classe Portfolio.
        """
        cls.registered[portfolio.name] = portfolio


    @classmethod
    def unregister(cls, name: str=None, is_all: bool=False) -> None:
        """Remove o Portfolio de nome 'name' do dicionário
        registered. Se is_all = True, todos os nomes são
        apagados.

        Args:
            name (str, optional): nome do Portfolio. Padrão: None.
            is_all (bool, optional) se True, realize o método
            clear() em registered.

        Raises:
            NameError: se name is None e is_all == False.
        """
        if is_all:
            cls.registered.clear()
        else:
            if name is None:
                raise NameError('Favor inserir um nome.')
            del cls.registered[name]


    @property
    def name(self) -> str:
        """Retorna o nome do Portfólio.

        Returns:
            str
        """
        return self.__name


    @name.setter
    def name(self, new_name: str) -> None:
        """Atribui um novo nome a Portfolio. A alteração só
        é permitida se len(new_name) != 0 e se new_name não
        estiver registrado. O novo nome é automaticamente
        registrado.

        Args:
            new_name (str): novo nome do Portfolio.

        Raises:
            ValueError: se len(new_name) == 0.
            NameError: se new_name in Portfolio.registered.keys().
        """
        if len(new_name) == 0:
            raise ValueError('Nome deve ter no mínimo um caracter.')

        if len(Portfolio.registered) > 0:
            if len(new_name) == 0:
                raise ValueError('Nome deve ter no mínimo um caracter.')
            elif new_name in Portfolio.registered.keys():
                if Portfolio.overwrite:
                    print('Nome sobrescrito.')
                    pass
                else:
                    raise NameError('Já existe um portfolio com este nome.')

        self.__name = new_name
        Portfolio.register(self)


    @property
    def tickers(self) -> list:
        """Lista de ativos do Portfolio.

        Returns:
            list
        """
        return self.__tickers


    @tickers.setter
    def tickers(self, new_tickers: list) -> None:
        """Atribui novos tickers do Portfolio.
        (ALTERAÇÃO NÃO RECOMENDADA!)

        Args:
            new_tickers (list): se len(new_tickers) == 0.

        Raises:
            ValueError: uma lista com no mínimo um ticker deve ser
            fornecida.
        """
        if len(new_tickers) == 0:
            raise ValueError('Favor inserir uma lista com, no mínimo, um ticker.')

        self.__tickers = new_tickers


    @property
    def weights(self) -> np.ndarray:
        """Distribuição de pesos dos ativos do Portfolio.

        Returns:
            np.ndarray
        """
        return self.__weights


    @weights.setter
    def weights(self, new_weights: np.ndarray) -> None:
        """Atribui novos pesos ao Portfolio. Se o mesmo
        conter apenas um ticker, nenhuma troca será feita,
        pois new_weights = np.array([1]) automaticamente.
        Os novos pesos devem somar para 1, com tolerância
        de Portfolio.delta. Quando a troca é realizada,
        o registro é atualizado.

        Args:
            new_weights (np.ndarray): array com os novos pesos.

        Raises:
            ValueError: se np.abs(1 - np.sum(new_weights)) >
            Portfolio.delta.
        """
        if len(self.tickers) == 1:
            new_weights = np.array([1])
        elif np.abs(1 - np.sum(new_weights)) > Portfolio.delta:
            raise ValueError('Os pesos devem somar para 1.')

        self.__weights = new_weights
        Portfolio.register(self)


    @property
    def dates(self) -> tuple:
        """Retorna as datas que compõem o Portfolio.

        Returns:
            tuple: (start, end)
        """
        return self.__dates


    @dates.setter
    def dates(self, new_dates: tuple) -> None:
        """Atribui novas datas ao Portfolio.
        (ALTERAÇÃO NÃO RECOMENDADA!)

        Args:
            new_dates (tuple): (start, end).

        Raises:
            ValueError: se somente uma das datas for inserida.
        """
        check = sum(1 for d in new_dates if isinstance(d, dt))
        if check == 2:
            self.__dates = new_dates
        elif check == 1:
            raise ValueError('Favor informar ambas as datas.')
        else:
            self.__dates = (None, None)


    @property
    def prices(self) -> pd.DataFrame:
        """Dataframe dos preços diários do Portfolio.

        Returns:
            pd.DataFrame
        """
        return self.__prices


    @prices.setter
    def prices(self, new_prices: pd.DataFrame) -> None:
        """Atribui novos pesos ao Portfolio.
        (ALTERAÇÃO NÃO RECOMENDADA!)

        Args:
            new_prices (pd.DataFrame)
        """
        self.__prices = new_prices


    def d_returns(self, is_portfolio: bool=True, col_name: str='Retornos') -> pd.DataFrame:
        """Retorna os retornos diários do portfólio, se
        is_portfolio=True, ou dos ativos que o compõem, se
        is_portfolio=False.

        Args:
            is_portfolio (bool, optional): refere-se aos retornos
            do portfólio ou dos ativos que o compõem. Padrão: True.
            col_name (str, optional): nome da coluna de retornos. Padrão: 'Retornos'.

        Returns:
            pd.DataFrame
        """
        if is_portfolio:
            ret = (qt.returns(self.prices) * self.weights).sum(axis=1).to_frame()
            ret.rename(columns={0: col_name}, inplace=True)
            return ret.dropna()
        return qt.returns(self.prices).dropna()


    def m_returns(self, is_portfolio: bool=True) -> pd.DataFrame:
        """Retorna os retornos mensais do portfólio, se
        is_portfolio==True, ou dos ativos que o compõem, se
        is_portfolio==False.
        Args:
            is_portfolio (bool, optional): retorno do portfólio
            ou dos ativos que o compõem. Padrão: True.
        Returns:
            pd.DataFrame
        """
        d_rets = self.d_returns(is_portfolio=is_portfolio)

        # dataframe com multindex
        # np.log1p(r) = np.log(1 + r)
        # np.expm1(r) = np.exp(r - 1)
        m_rets = d_rets.groupby(
            [d_rets.index.year, d_rets.index.month]
        ).apply(lambda r: np.expm1(np.log1p(x).sum()))

        # deixando o index como Y-m, em datetime
        m_rets.index = map(
            lambda d: dt.strptime(f'{d[0]}-{d[1]}', '%Y-%m'), m_rets.index
        )
        m_rets.index = m_rets.index.to_period('M')
        return m_rets


    def a_returns(self, is_portfolio: bool=True) -> pd.DataFrame:
        """Retorna os retornos anuais do portfólio, se
        is_portfolio==True, ou dos ativos que o compõem, se
        is_portfolio==False.
        Args:
            is_portfolio (bool, optional): retorno do portfólio
            ou dos ativos que o compõem. Padrão: True.
        Returns:
            pd.DataFrame
        """
        d_rets = self.d_returns(is_portfolio)

        # np.log1p(r) = np.log(1 + r)
        # np.expm1(r) = np.exp(r - 1)
        a_rets = d_rets.groupby(
            d_rets.index.year
        ).apply(lambda r: np.expm1(np.log1p(x).sum()))
        a_rets.index = pd.to_datetime(a_rets.index.astype(str)).to_period('Y')
        return a_rets


    def total_returns(self, period: str='a') -> pd.DataFrame:
        """Retorna a variação total do período,
        (preço final - preço inicial) / preço final.

        Args:
            period (str, optional): refere-se à periodização
            dos retornos ('m' ou 'a'). Padrão: 'a'.

        Returns:
            pd.DataFrame
        """
        return qt.returns(self.prices, which='total', period=period).dropna()


    def acm_returns(self, is_portfolio: bool=True) -> pd.DataFrame:
        """Retorna os retornos acumulados do portfólio, se
        is_portfolio=True, ou dos ativos que o compõem, se
        is_portfolio=False.

        Args:
            is_portfolio (bool, optional): refere-se ao retorno acm
            do portfolio, ou dos ativos individuais. Padrão: True.

        Returns:
            pd.DataFrame
        """
        acm = (1 + self.d_returns(is_portfolio=is_portfolio)).cumprod()
        acm.rename(columns={'Retornos': self.name}, inplace=True)
        return acm.dropna()


    def portfolio_return(self, period: str='a') -> float:
        """Retorna o retorno do portfólio, da forma
        total_returns.dot(weights).

        Args:
            period (bool, optional): refere-se à periodização
            do retorno ('m' ou 'a'). Padrão: 'a'.

        Returns:
            float
        """
        return self.total_returns(period).dot(self.weights)


    def covariance(self) -> pd.DataFrame:
        """Retorna a matrix de covariância dos ativos
        que compõem o Portfolio.

        Returns:
            pd.DataFrame
        """
        return self.d_returns(is_portfolio=False).cov()


    def __check(arg_name: str, possible_values: tuple):
        """Função decoradora designada para verificar os
        argumentos default de uma função. Levanta um erro se
        'arg_name' (nome do argumento default) não pertence a
        'possible_values'.

        Args:
            arg_name (str): nome do argumento default.
            possible_values (tuple): possíveis valores que o
            argumento pode assumir.
        """
        def check_inner(f):
            @wraps(f)
            def check(*args, **kwargs):
                p = get_default_args(f)
                p.update(kwargs)

                if p[arg_name] not in possible_values:
                    raise KeyError(f"{arg_name} inválido. Usar {possible_values}.")
                return f(*args, **kwargs)
            return check
        return check_inner


    @__check('plot_in', ('sns', 'go'))
    def benchmark(
        self,
        portfolios: list,
        plot_in: str='sns',
        size: tuple=(19, 6),
        name: str=None,
        is_return: bool=False
    ) -> None:
        """Plot um benchmark do Portfolio que está chamando
        este método com os Portfolios em 'portfolios'. O plot
        pode ser pelo seaborn ou no plotly.

        Args:
            portfolios (list): lista de Portfolios.
            plot_in (str, optional): onde será plotado o
            benchmark: 'sns' ou 'go'. Padrão: 'sns'.
            fsize (tuple, optional): tamanho do plot. Padrão: (19, 6).

        Raises:
            ValueError: se len(portfolios) == 0.
            TypeError: se algum elemento de portfolios não for
            um Portfolio.
        """
        if len(portfolios) == 0:
            raise ValueError('Favor listar no mínimo um portfólio.')

        check = sum(1 for p in portfolios if isinstance(p, Portfolio))
        if check != len(portfolios):
            raise TypeError('Favor listar somente objetos da classe Portfolio.')

        bench = self.acm_returns()
        for p in portfolios:
            bench = pd.concat(
                [bench, p.acm_returns()],
                axis=1,
                join='inner'
            )

        titles = [
            f'Benchmark: {self.dates[0].strftime("%d/%m/%Y")} - {self.dates[1].strftime("%d/%m/%Y")}',
            'Data',
            'Fator'
        ]

        qt.plot_lines(dfs=[bench], titles=titles, plot_in=plot_in, name=name, is_return=is_return)


    def beta(self, benchmark) -> float:
        """Retorna o beta do Portfolio com o Portfolio
        de benchmark.

        Args:
            benchmark (Portfolio): Portfolio a servir como
            benchmark.

        Raises:
            TypeError: se benchmark não for um Portfolio.

        Returns:
            float: beta.
        """
        if not isinstance(benchmark, Portfolio):
            raise TypeError('Favor inserir um Portfolio.')

        ret_port = self.d_returns(col_name=self.name)
        ret_bench = benchmark.d_returns(col_name=benchmark.name)

        return qt.beta(ret_port, ret_bench)


    def volatility(self, is_portfolio: bool=True):
        """Retorna as volatilidades diária, mensal e anual.

        Args:
            is_portfolio (bool, optional): se False, calcula
            a volatilidade individual dos ativos que compõem
            o portfólio, através do desvio padrão dos retornos
            diários. Se True, retorna a volatilidade do port-
            fólio, considerando os pesos dos ativos e sua ma-
            triz de covariância. Padrão: True.

        Returns:
            pd.DataFrame, se is_portfolio==False,
            pd.Series, se is_portfolio==True.
        """
        if not is_portfolio:
            return pd.DataFrame(
                data=map(
                    lambda p: self.d_returns(is_portfolio=False).std() * np.sqrt(p), (1, 21, 252)
                ),
                index=['Diária', 'Mensal', 'Anual']
            )

        vol_d = qt.vol(self.weights, self.covariance(), annual=False)
        return pd.Series({
            'Diária': vol_d,
            'Mensal': vol_d * np.sqrt(21),
            'Anual': vol_d * np.sqrt(252)
        })


    @__check('which', ('sharpe', 'sortino'))
    def s_index(self, risk_free_rate: float=rf, which: str='sharpe') -> float:
        """Retorna o índice de Sharpe ou de Sortino, a depender
        de 'which', anualizado.

        Args:
            risk_free_rate (float, optional): taxa livre de risco.
            Padrão: 0.03.
            which (str, optional): qual índice deve ser retornado,
            'sharpe' ou 'sortino'. Padrão: 'sharpe'.

        Returns:
            float.
        """
        ret = self.portfolio_return()
        vols = {'sharpe': self.volatility().loc['Anual'], 'sortino': self.downside()}

        return qt.sharpe(ret, vols[which], risk_free_rate)


    @__check('which', (95, 97, 99, 99.9, None))
    def var(
        self, *,
        which: int=None, kind: str='hist',
        period: str='d', is_neg: bool=True,
        modified: bool=False
    ):
        """Retorna um pd.Series com os VaRs, históricos, ou
        paramétricos, 95, 97, 99 e 99.9, ou apenas um deles,
        escolhido através de 'which'. Os parâmetros obrigato-
        riamente devem ser nomeados.

        Args:
            which (int, optional): se somente um dos VaRs
            deve ser retornado: 95, 97, 99 ou 99.9. Padrão: None.
            kind (str, optional): módulo de cômputo do var: histórico
            ('hist') ou paramétrico ('param'). Padrão: 'hist'.
            period (str, optional): VaR dos retornos diários ('d'),
            mensais ('m') ou anuais ('a'). Padrão: 'd'.
            is_neg (bool, optional): se os valores retornados
            devem ser positivos ou negativos. Padrão: True.
            modified (bool, optional): somente válido se kind='param',
            se True, considera a skewness e a curtose da distribuição
            e realiza a correção de Cornish-Fisher.

        Raises:
            KeyError: se period not in ('d', 'm', 'a').

        Returns:
            pd.Series, se which == None, ou float, se which != None.
        """
        if period not in ('d', 'm', 'a'):
            raise KeyError("Period inválido: usar 'd', 'm' ou 'a'.")

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
            raise IndexError('Modo de VaR inválido: usar "hist" ou "param".')


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
        """Retorna um pd.Series com os CVaRs, históricos, ou
        paramétricos, 95, 97, 99 e 99.9, ou apenas um deles,
        escolhido através de 'which'. Os parâmetros obrigato-
        riamente devem ser nomeados.

        Args:
            which (int, optional): se somente um dos CVaRs
            deve ser retornado: 95, 97, 99 ou 99.9. Padrão: None.
            kind (str, optional): módulo de cômputo do CVaR: histórico
            ('hist') ou paramétrico ('param'). Padrão: 'hist'.
            period (str, optional): CVaR dos retornos diários ('d'),
            mensais ('m') ou anuais ('a'). Padrão: 'd'.
            is_neg (bool, optional): se os valores retornados
            devem ser positivos ou negativos. Padrão: True.
            modified (bool, optional): somente válido se kind='param',
            se True, considera a skewness e a curtose da distribuição
            e realiza a correção de Cornish-Fisher.

        Raises:
            KeyError: se period not in ('d', 'm').
            IndexError: se kind not in ('hist', 'param').

        Returns:
            pd.Series, se which == None, ou float, se which != None.
        """
        if period not in ('d', 'm', 'a'):
            raise KeyError("Period inválido: usar 'd', 'm' ou 'a'.")

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
            raise IndexError('Modo de CVaR inválido: usar "hist" ou "param".')


        if not is_neg:
            cvar = -cvar

        if not which:
            return cvar
        return cvar.loc[which]


    def all_vars(self, period: str='d', is_neg: bool=False) -> pd.DataFrame:
        """Retorna um dataframe com os VaRs histórico, paramétrico,
        e paramétrico ajustado.

        Args:
            period (str, optional): VaR dos retornos diários ('d'),
            mensais ('m') ou anuais ('a'). Padrão: 'd'.
            is_neg (bool, optional): se os valores retornados
            devem ser positivos ou negativos. Padrão: True.

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame(
            {'Hist': self.var(period=period, is_neg=is_neg),
             'Parametric': self.var(period=period, kind='param', is_neg=is_neg),
             'Parametric_Adj': self.var(period=period, kind='param', is_neg=is_neg, modified=True)}
        )


    @__check('period', ('d', 'm', 'a'))
    def downside(self, period: str='a') -> float:
        """Retorna o downside (std dos retornos negativos)
        periodizado (diário, mensal ou anual).

        Args:
            period (str, optional): período de interesse. Padrão: 'a'.

        Returns:
            float
        """
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets['Retornos'] < 0].std()[0] * np.sqrt(factor[period])


    @__check('period', ('d', 'm', 'a'))
    def upside(self, period: str='a') -> float:
        """Retorna o upside (std dos retornos positivos)
        periodizado (diário, mensal ou anual).

        Args:
            period (str, optional): período de interesse. Padrão: 'a'.

        Returns:
            float
        """
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets['Retornos'] > 0].std()[0] * np.sqrt(factor[period])


    def rol_drawdown(self, window: int=21, is_number: bool=True):
        """Retorna o(s) drawdown(s) máximo(s) dado o período de tempo
        'window'.

        Args:
            window (int, optional): Período de interesse. Padrão: 21.
            is_number (bool, optional): Se True, retorna o drawdown máximo.
            Se False, retorna um df do drawdown móvel. Padrão: True.

        Returns:
            float ou pd.DataFrame
        """
        acm_rets = self.acm_returns()
        rol_max = acm_rets.rolling(window=window).max()
        drawdown_ = acm_rets / rol_max - 1
        max_drawdown = drawdown_.rolling(window=window).min()

        if is_number:
            return max_drawdown.min()[0]
        return max_drawdown.dropna()


    @__check('period', ('d', 'm', 'a'))
    def calc_skewness(self, period: str='d') -> float:
        """Retorna a skewness da distribuição de retornos diários.

        Args:
            period (str, optional): distribuição a considerar ('d', 'm', 'a').
            Padrão: 'd'.

        Returns:
            float
        """
        r = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }
        return skew(r[period]())[0]


    @__check('period', ('d', 'm', 'a'))
    def calc_curtose(self, is_excess: bool=True, period: str='d') -> float:
        """Retorna a curtose da distribuição de retornos. Os parâmetros são os
        mesmos da função do scipy.stats.

        Args:
            is_excess (bool, optional): se True, retorna curtose - 3.
            period (str, optional): distribuição a considerar ('d', 'm', 'a').
            Padrão: 'd'.

        Returns:
            float
        """
        r = {
            'd': self.d_returns,
            'm': self.m_returns,
            'a': self.a_returns
        }

        if is_excess:
            return kurtosis(r[period]())[0] - 3
        return kurtosis(r[period]())[0]


    @__check('period', ('d', 'm', 'a'))
    def shapiro_test(self, period: str='d', confidence: float=.05) -> bool:
        """Verifica, dentro de um nível de confiança 'confidence',
        se os retornos diários/mensais/anuais assumem uma distribuição normal.

        Returns:
            bool
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
        """Retorna um dataframe com uma coleção de métricas.

        Args:
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.
            window (int, optional): janela de tempo (drawdown). Padrão: 21.
            period (str, optional): período a ser calculado ('d', 'm', 'a').
            Padrão: 'd'.
            benchmark (Portfolio, optional): benchmark (beta). Padrão: None.

        Returns:
            pd.DataFrame
        """
        dict_metrics = {
            'Retorno anual': self.portfolio_return(),
            'Volatilidade anual': self.volatility().loc['Anual'],
            'Ind. Sharpe': self.s_index(risk_free_rate),
            'Ind. Sortino': self.s_index(risk_free_rate, 'sortino'),
            f'Skewness ({period})': self.calc_skewness(period=period),
            f'Ex_Curtose ({period})': self.calc_curtose(period=period),
            f'VaR 99.9 ({period})': self.var(which=99.9, period=period, is_neg=False),
            f'CVaR 99.9 ({period})': self.cvar(which=99.9, period=period),
            f'Max. Drawdown ({window})': self.rol_drawdown(window),
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
        """Método que transfere os dados do Portoflio original,
        como tickers, datas e preços, para um novo Portfolio, cujo
        nome e pesos serão 'new_name' e 'new_weights'.

        Args:
            new_name (str): nome do novo Portfolio.
            new_weights (np.array): pesos do novo Portfolio.

        Returns:
            Portfolio
        """
        new_p = deepcopy(self)
        new_p.name = new_name
        new_p.weights = new_weights

        return new_p


    @classmethod
    def all_rets(cls, period: str='a') -> pd.Series:
        """Retorna um pd.Series com os retornos de todos
        os Portfolios registrados.

        Args:
            period (str, optional): refere-se à periodização
            dos retornos ('m' ou 'a'). Padrão: 'a'.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0.

        Returns:
            pd.Series
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.portfolio_return(period)
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_vols(cls, period: str='a') -> pd.Series:
        """Retorna um pd.Series com as volatilidades (anualizadas)
        de todos os Portfolios registrados.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0.

        Returns:
            pd.Series
        """
        d_per = {
            'd': 'Diária',
            'm': 'Mensal',
            'a': 'Anual'
        }
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.volatility().loc[d_per[period]]
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    @__check('which', ('sharpe', 'sortino'))
    def all_sindex(cls, risk_free_rate: float=rf, *, which='sharpe') -> pd.Series:
        """Retorna um pd.Series com o índice de Sharpe (ou Sortino),
        anualizado, de todos os Portfolios registrados.

        Args:
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.
            which (str, optional): 'sharpe' ou 'sortino'. Padrão: 'sharpe'.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0

        Returns:
            pd.Series
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
        """Retorna um pd.Series com os pesos de todos os Portfolios
        registrados.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0

        Returns:
            pd.Series
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                n: p.weights
                for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_metrics(
        cls,
        portfolios: list=[],
        risk_free_rate: float=rf,
        window: int=21,
        period: str='d',
        benchmark=None
    ) -> pd.DataFrame:
        """Retorna um dataframe com as métricas de todos os Portfolios
        em 'portfolios'.

        Args:
            portfolios (list, optional): lista de Portfolios. Padrão: [].
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.
            window (int, optional): janela de tempo (drawdown). Padrão: 21.
            period (str, optional): período a ser calculado ('d', 'm', 'a').
            Padrão: 'd'.
            benchmark (Portfolio, optional): benchmark (beta). Padrão: None.

        Raises:
            ValueError: se len(portfolios) == 0.
            AttributeError: se houver um elemento de portfolios que não seja
            Portfolio.
            NotImplementedError: se len(Portfolio.registered) == 0.

        Returns:
            pd.DataFrame
        """
        if len(cls.registered) > 0:
            if len(portfolios) == 0:
                raise ValueError('Favor inserir, no mínimo, um Portfolio.')

            check = sum(1 for p in portfolios if isinstance(p, Portfolio))
            if check != len(portfolios):
                raise AttributeError('Favor somente listas Portfolios.')


            df = portfolios[0].metrics(risk_free_rate, window, period, benchmark)
            for p in portfolios[1:]:
                df_ = p.metrics(risk_free_rate, window, period, benchmark)
                df = pd.concat(
                    [df, df_],
                    axis=1,
                    join='inner'
                )
            return df

        raise NotImplementedError('Nenhum portfólio cadastrado.')
