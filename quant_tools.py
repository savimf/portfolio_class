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
from scipy.stats import skew, kurtosis, norm
from scipy.optimize import minimize
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import visuals


save_path = 'pictures/'


def carteira(ativos: list, start: dt, end: dt, source: str='iv', crypto: bool=False) -> pd.DataFrame:
    """Retorna um dataframe com os preços de fechmanto diários de 'ativos',
    dentro do período 'start' e 'end'. É possível utilizar as fontes
    investing.com (source = 'iv') e yahoo finance (source = 'yf'). Criptoativos
    podem ser coletados de yf (usar crypto=True).

    Args:
        ativos (list): lista dos ativos a serem baixados.
        start (datetime): data de início.
        end (datetime): data final.
        source (str, optional): fonte de coleta 'iv' ou 'yf'. Padrão: 'iv'.
        crypto (bool, optional): deve ser setado para True, por questões de
        formatação, se houver somente criptoativos em 'ativos'.

    Returns:
        pd.DataFrame: dataframe com os preços diários dos ativos contidos
        em 'ativos', entre o período 'start' e 'end'.
    """
    carteira_precos = pd.DataFrame()

    if sum(1 for d in (start, end) if isinstance(d, dt)) == 0:
        return carteira_precos

    if source == 'iv':
        for ativo in ativos:
            carteira_precos[ativo] = iv.get_stock_historical_data(
                stock=ativo,
                country='brazil',
                from_date=start.strftime('%d/%m/%Y'),
                to_date=end.strftime('%d/%m/%Y'))['Close']
    elif source == 'yf':
        if not crypto:
            for ativo in ativos:
                t = yf.Ticker(f'{ativo}.SA')
                carteira_precos[ativo] = t.history(
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    interval='1d')['Close']
        else:
            for ativo in ativos:
                t = yf.Ticker(ativo)
                carteira_precos[ativo] = t.history(
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    interval='1d')['Close']
    else:
        raise NameError('Fonte inválida.')

    carteira_precos.index = pd.to_datetime(carteira_precos.index)
    return carteira_precos


def time_fraction(start: dt, end: dt, period: str='d') -> float:
    """Função que calcula a fração de tempo, a partir de 'start' até
    'end', na escala determinada em 'period', considerando somente os
    dias de pregão: 252 dias/ano, 21 dias/mês.

    Ex: considerando que haja 10 meses entre 'start' e 'end':
    period = 'd' retorna 21 * 10 = 210;
    period = 'm' retorna 210/21 = 10;
    period = 'a' retorna 210/252 = 10/12 = 0.833...;

    Ex: considerando que haja 3.5 anos entre 'start' e 'end':
    period = 'd' retorna 252 * 3.5 = 882;
    period = 'm' retorna 252 * 3.5 / 21 = 12 * 3.5 = 42;

    Ex: considerando que haja 30 dias entre 'start' e 'end':
    period = 'm' retorna 30/21 = 1.4286...;
    period = 'a' retorna 30/252 = 0.1190...

    Args:
        start (datetime): data de início.
        end (datetime): data final,
        period (str, optional): escala de tempo: 'd', 'm' ou
        'a'. Padrão: 'd'.

    Returns:
        float: quantos dias/meses/anos há (de pregão)
        entre 'start' e 'end'.
    """
    if isinstance(start, str):
        start = dt.strptime(start, '%d/%m/%Y')

    if isinstance(end, str):
        end = dt.strptime(end, '%d/%m/%Y')

    n_days = rd(end, start).days
    n_months = rd(end, start).months
    n_years = rd(end, start).years

    total = n_days + 21 * n_months + 252 * n_years

    if period == 'd':
        return total
    elif period == 'm':
        return total / 21
    elif period == 'a':
        return total / 252
    raise KeyError("Período inválido -> 'd', 'm' ou 'a'.")


def get_quandl(taxa: str, start: dt, end: dt) -> pd.DataFrame:
    """Retorna um pd.DataFrame, coletado do quandl, da taxa
    ipca (código 12466), ou imab (código 12466), ou selic
    (código 4189) no período [start, end].

    Args:
        taxa (str): ipca, imab, ou selic.
        start (datetime): data de início.
        end (datetime): data final

    Raises:
        NameError: se taxa not in ('ipca', 'imab', 'selic')

    Returns:
        pd.DataFrame
    """
    cod = {
        'ipca': 13522,
        'imab': 12466,
        'selic': 4189
    }

    if taxa.lower() not in cod.keys():
        raise NameError('Taxa inválida. Usar ipca, imab ou selic.')

    df = quandl.get(
        f'BCB/{int(cod[taxa.lower()])}',
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d')
    )
    df.rename(columns={'Value': taxa.upper()}, inplace=True)
    return df


def selic(start: dt, end: dt, is_number: bool=False, period: str='a'):
    """Retorna a variação, diária, mensal ou anual, da taxa Selic
    coletada do quandl entre 'start' e 'end', a depender de 'period'.

    Args:
        start (datetime): data de início.
        end (datetime): data final.
        is_number (bool, optional): se False, retorna um pd.Series
        com as variações. Se True, retorna o valor médio do período.
        Padrão: False.
        period (str, optional): ('d'/'m'/'a'). Padrão: 'a'.

    Raiser:
        IndexError: se 'period' not in ('d', 'm', 'a').

    Returns:
        pd.Series ou float.
    """
    s = get_quandl('selic', start, end) / 100

    if is_number:
        if period not in ('d', 'm', 'a'):
            raise IndexError("Período inválido -> 'd' (diário), 'm' (mensal) ou 'a' (anual).")

        s_mean = s.mean()[0]
        n_months = s.shape[0]

        # selic anual / mensal / diária
        if period == 'a':
            s = (1 + s_mean) ** (12 / n_months) - 1
        elif period == 'm':
            pass
        elif period == 'd':
            s = (1 + s_mean) ** (1 / (21 * n_months)) - 1
    return s


def returns(prices: pd.DataFrame, which: str='daily', period: str='a', scaled: bool=True):
    """Retorna um dataframe ou uma série dos retornos (diários) de prices,
    a depender de 'which', diários, mensais ou anuais, a depender de 'period'.

    Ex: which = 'daily' retorna prices.pct_change().dropna() (retornos diários);
    which = 'total' retorna (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    (retornos totais), que podem ser diários (period = 'd'), mensais
    (period = 'm') ou anuais (period = 'a');
    which = 'acm' retorna os retornos acumulados
    (1 + prices.pct_change().dropna()).cumprod()

    Args:
        prices (pd.DataFrame): dataframe dos preços de fechamento.
        which (str, optional): tipo de retorno desejado: diário/total/
        mensal/acumulado ('daily'/'total'/'monthly'/'acm'). Padrão: 'daily'.
        period (str, optional): retorno diário/mensal/anual 'd'/'m'/'a'
        (válido somente para which = 'total'). Padrão: 'a'.

    Returns:
        pd.DataFrame ou pd.Series: a depender de 'which'; retornos diários
        (dataframe), totais (series) ou acumulados (dataframe).
    """
    r = prices.pct_change().dropna()
    if which == 'daily':
        return r
    elif which == 'monthly':
        return r.groupby(
            [df.index.year, df.index.month]
        ).apply(lambda r: (1 + r).prod() - 1)
    elif which == 'total':
        rets = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

        if not scaled:
            return rets

        n_days = prices.shape[0]
        n_years = n_days / 252
        if period == 'm':
            return (1 + rets) ** (1 / (12 * n_years)) - 1
        elif period == 'a':
            return (1 + rets) ** (1 / n_years) - 1
        raise TypeError("Período inválido: 'm' ou 'a'.")
    elif which == 'acm':
        return (1 + r).cumprod()
    raise TypeError(
        "Tipo de retorno inválido: which -> 'daily', 'total', 'monthly, ou 'acm'."
    )


def search(txt: str, n: int):
    """Função que coleta as 'n' primeiras buscas referentes a
    txt = 'tesouro' ou txt = 'bvsp' ou txt = 'ifix'.

    Args:
        txt (str): objeto a ser pesquisado: 'tesouro', 'bvsp' ou 'ifix'.
        n (int): número de resultados.

    Returns:
        iv..utils.search_obj.SearchObj: iterator de dicionários
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
    """Retorna um pd.DataFrame com os preços de fechamento do índice
    'index', no intervalo [start, end].

    Args:
        index (str): índice ('ifix' ou 'bvsp').
        start (datetime): data de inicio.
        end (datetime): data final.

    Raises:
        NameError: se index not in ('ifix', 'bvsp').

    Returns:
        pd.DataFrame
    """
    if index not in ('ifix', 'bvsp'):
        raise NameError('Índice inválido. Usar "ifix" ou "bvsp".')

    df = search(index, 1).retrieve_historical_data(
        from_date=start.strftime('%d/%m/%Y'),
        to_date=end.strftime('%d/%m/%Y')
    )['Close'].to_frame()

    df.rename(columns={'Close': index.upper()}, inplace=True)
    return df


def mae(y_true: np.array, y_pred: np.array) -> float:
    """Função que calcula o mean absolute error entre
    y_true e y_pred.

    Args:
        y_true (np.array): array dos valores observados.
        y_pred (np.array): array dos valores preditos.

    Returns:
        float: mean absolute error(y_true, y_pred)
    """
    return metrics.mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.array, y_pred: np.array) -> float:
    """Função que calcula o root mean square error entre
    y_true e y_pred.

    Args:
        y_true (np.array): array dos valores observados.
        y_pred (np.array): array dos valores preditos.

    Returns:
        float: root mean square error(y_true, y_pred)
    """
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def error_metrics(y_true: np.array, y_pred: np.array) -> None:
    """Imprime na tela o mae(y_true, y_pred) e rmse(y_true,
    y_pred).

    Args:
        y_true (np.array): array dos valores observados.
        y_pred (np.array): array dos valores preditos.
    """
    print(
        f'MAE: {mae(y_true, y_pred)}\n'
        f'RMSE: {rmse(y_true, y_pred)}'
    )


def mae_cov(cov_past: pd.DataFrame, cov_fut: pd.DataFrame) -> float:
    """Função que calcula o MAE para dois dataframes de covariância,
    passado e futuro, em porcentagem.

    Args:
        cov_past (pd.DataFrame): dataframe da covariância passado.
        cov_fut (pd.DataFrame): dataframe da covariância futuro.

    Returns:
        float: MAE entre os dataframes de covariância, em porcentagem.
    """
    r = np.sum(
        np.abs(
            np.diag(cov_past) - np.diag(cov_fut)
        )
    ) / len(np.diag(cov_past))

    return round(r, 4) * 100


def cornish_fisher_z(z: float, s: float, k: float) -> float:
    """Retorna o z-score ajustado, levando em consideração
    a skewness (s) e a curtose (k) da distribuição, pelo
    método de Cornish-Fisher.

    Args:
        z (float): z-score da distribuição normal.
        s (float): skewness da distribuição.
        k (float): curtose da distribuição.

    Returns:
        float
    """
    return z + (1/6) * s * ((z ** 2 - 1) - \
        (1/6) * (2 * z ** 3 - 5 * z) * s) + \
            (1/24) * (z ** 3 - 3 * z) * (k - 3)


def vars_hist(rets: pd.DataFrame) -> dict:
    """Retorna um dicionário com os 4 VaRs: 95%, 97%, 99% e
    99.9% do dataframe de retornos 'rets'.

    Args:
        rets (pd.DataFrame): dataframe dos retornos.

    Returns:
        dict: {95: ..., 97: ...,
        99: ..., 99.9: ...}
    """
    if not isinstance(rets, pd.DataFrame):
        raise TypeError('Favor usar como entrada pd.DataFrame ou pd.Series.')

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


def vars_gaussian(rets: pd.DataFrame, modified: bool=False) -> dict:
    """Retorna um dicionário com os 4 VaRs-paramétricos: 95%, 97%,
    99% e 99.9% do dataframe de retornos 'rets'. Se modified=True,
    considera a skewness e curtose de 'rets' e realiza a correção de
    Cornish-Fisher.

    Args:
        rets (pd.DataFrame): dataframe dos retornos.

    Returns:
        dict: {95: ..., 97: ...,
        99: ..., 99.9: ...}
    """
    lvls = (95, 97, 99, 99.9)

    # z-scores
    zs = [norm.ppf(1 - lvl / 100) for lvl in lvls]

    if modified:
        s, k = skew(rets), kurtosis(rets)
        zs = [cornish_fisher_z(z, s, k) for z in zs]

    vol = rets.std()
    var = {
        v[0]: (rets.mean() + v[1] * vol)[0]
        for v in zip(lvls, zs)
    }
    return var


def cvars_hist(rets: pd.DataFrame, ret_name: str='Retornos') -> dict:
    """Retorna o Conditional VaR histórico dos retornos em 'rets',
    dados.

    Args:
        rets (pd.DataFrame): dataframe de retornos.
        ret_name (str): nome da coluna de retornos.
        Padrão: 'Retornos'.

    Returns:
        dict: {95: ..., 97: ...,
        99: ..., 99.9: ...}
    """
    var = vars_hist(rets)

    c_vars = {
    i[0]: -rets[rets[ret_name] <= i[1]].mean()[0]
    for i in var.items()
    }
    return c_vars


def cvars_gaussian(rets: pd.DataFrame, ret_name: str='Retornos', modified: bool=False) -> dict:
    """Retorna o Conditional VaR-paramétrico dos retornos em 'rets'. Se
    modified=True, considera a skewness e curtose de 'rets' e realiza a
    correção de Cornish-Fisher.

    Args:
        df (pd.DataFrame): dataframe de retornos.
        ret_name (str): nome da coluna de retornos.
        Padrão: 'Retornos'.

    Returns:
        dict: {95: ..., 97: ...,
        99: ..., 99.9: ...}
    """
    var = vars_gaussian(rets, modified)

    c_vars = {
    i[0]: -rets[rets[ret_name] <= i[1]].mean()[0]
    for i in var.items()
    }
    return c_vars


def vol(pesos: np.array, cov: pd.DataFrame, annual: bool=True) -> float:
    """Retorna a volatilidade, anualizada ou não, a depender
    de 'annual', dados o array de pesos 'pesos' e a matriz
    de covariância 'cov'.

    Args:
        pesos (np.array): array dos pesos dos ativos.
        cov (pd.DataFrame): dataframe de covariância.
        annual (bool, optional): se True, retorna a
        volatilidade anualizada: vol * np.sqrt(252). Padrão: True.

    Returns:
        float.
    """
    vol = np.sqrt(
        np.dot(pesos.T, np.dot(cov, pesos))
    )

    if annual:
        return vol * np.sqrt(252)
    return vol


def beta(ret_carteira: pd.DataFrame, ret_ibvsp: pd.DataFrame) -> float:
    """Calcula o beta da carteira, dados seus retornos diários e
    os retornos do ibovespa.

    Args:
        ret_carteira (pd.DataFrame): dataframe dos retornos diários
        da carteira.
        ret_ibvsp (pd.DataFrame): dataframe dos retornos diários do
        ibovespa.

    Returns:
        float.
    """
    df = pd.concat(
        [ret_carteira, ret_ibvsp],
        axis=1,
        join='inner'
    )

    Y = df.iloc[:,0]
    X = df.iloc[:,1]
    X = sm.add_constant(X)

    linear_model = sm.OLS(Y, X)
    return linear_model.fit().params[1]


def sharpe(ret: float, vol: float, risk_free_rate: float) -> float:
    """Retorna o índice de Sharpe, dados o retorno total anuali-
    zado, volatilidade anualizada e a taxa livre de risco. Pode
    também ser utilizado para retornar o índice de Sortino se
    a volatilidade inserida refletir somente aquela de retornos
    negativos.

    Args:
        ret (float): retorno total anualizado da carteira.
        vol (float): volatilidade anual da carteira.
        risk_free_rate (float): taxa livre de risco.

    Returns:
        float.
    """
    return (ret - risk_free_rate) / vol


def minimize_vol(target_return: float, exp_rets: pd.Series, cov: pd.DataFrame) -> np.array:
    """Retorna os pesos do portfólio de mínima volatilidade, dado
    o retorno 'target_return', os retornos esperados 'exp_rets' e
    a matriz de covariância 'cov'.

    Args:
        target_return (float): retorno do portfólio desejado.
        exp_rets (pd.Series): retornos esperados.
        cov (pd.DataFrame): matriz de covariância.

    Returns:
        np.array
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


def maximize_sr(exp_rets: pd.Series, cov: pd.DataFrame, risk_free_rate: float=.03) -> np.array:
    """Retorna os pesos do portfólio de máximo índice de sharpe,
    dados os retornos esperados, matriz de covariância e a taxa
    livre de risco.

    Args:
        exp_rets (pd.Series): retornos esperados.
        cov (pd.DataFrame): matriz de covariância.
        risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.

    Returns:
        np.array: [description]
    """
    def neg_sharpe_ratio(weights: np.array, exp_rets: pd.Series, cov: pd.DataFrame, risk_free_rate: float=.03) -> float:
        """Retorna o índice de Sharpe negativo, dado o array de pesos.

        Args:
            weights (np.array): pesos dos ativos.
            exp_rets (pd.Series): retornos esperados.
            cov (pd.DataFrame): matriz de covariância
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.

        Returns:
            float: [description]
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


def gmv(cov: pd.DataFrame) -> np.array:
    """Retorna os pesos do portfólio GMV.

    Args:
        cov (pd.DataFrame): matriz de covariância.

    Returns:
        np.array
    """
    n = cov.shape[0]
    return maximize_sr(np.repeat(1, n), cov, 0)


def optimal_weights(exp_rets: pd.DataFrame, cov: pd.DataFrame, n_points: int) -> np.array:
    """Retorna uma lista dos pesos que minimizam a volatilidade,
    dados os retornos esperados 'exp_rets' e a matriz de covariância
    'cov'. Considera o retorno esperado mínimo e o máximo para criar
    uma lista com 'n_points' retornos igualmente espaçados entre eles.
    Para cada retorno desta lista, executa a função minimize_vol.

    Args:
        exp_rets (pd.DataFrame): retornos esperados.
        cov (pd.DataFrame): matriz de covariância.
        n_points (int): número de intervalos, igualmente espaçados,
        entre o menor e o maior retorno.

    Returns:
        np.array
    """
    target_returns = np.linspace(exp_rets.min(), exp_rets.max(), n_points)
    weights = [
        minimize_vol(target_return, exp_rets, cov)
        for target_return in target_returns
    ]
    return weights


def find_port_min_vol(portfolios: pd.DataFrame, col_name: str='Volatilidade') -> pd.DataFrame:
    """Retorna o portfólio de menor volatilidade entre todos
    os 'portfolios'. Realiza a busca assumindo que o nome da
    coluna com os valores das volatilidades é 'col_name'.

    Args:
        portfolios (pd.DataFrame): dataframe de portfólios.
        col_name (str, optional): nome da coluna com os dados
        das volatilidades. Padrão: 'Volatilidade'.

    Returns:
        pd.DataFrame
    """
    min_vol = portfolios[col_name].min()

    port_min_vol = portfolios.loc[portfolios[col_name] == min_vol]
    port_min_vol = port_min_vol.T.rename(
        columns={port_min_vol.index[0]: 'Valores'}
    )
    return port_min_vol


def find_port_max_sr(portfolios: pd.DataFrame, col_name: str='Ind. Sharpe') -> pd.DataFrame:
    """Retorna o portfólio de maior índice de Sharpe entre todos
    os 'portfolios'. Realiza a busca assumindo que o nome da
    coluna com os valores das volatilidades é 'col_name'.

    Args:
        portfolios (pd.DataFrame): dataframe de portfólios.
        col_name (str, optional): nome da coluna com os dados
        dos índices. Padrão: 'Ind. Sharpe'.

    Returns:
        pd.DataFrame
    """
    max_sr = portfolios[col_name].max()

    port_max_sr = portfolios.loc[portfolios[col_name] == max_sr]
    port_max_sr = port_max_sr.T.rename(
        columns={port_max_sr.index[0]: 'Valores'}
    )
    return port_max_sr


def plot_portfolios(
    portfolios: pd.DataFrame, color: str='brg',
    fsize: tuple=(12, 10), is_return: bool= False,
    save: bool=False
):
    """Plota os portfólios no plano vol x ret, destacando em azul o de
    mínima volatilidade e em vermelho o de máximo índice de Sharpe.

    Args:
        portfolios (pd.DataFrame): df contendo os portfólios.
        color (str, optional): palette de cores. Padrão: 'brg'.
        fsize (tuple, optional): tamanho do plot. Padrão: (12, 10).
        save (bool, optional): se True, salva um png do plot, com dpi=200,
        de nome 'portfolios_hd' em 'save_path'.
    """
    plt.figure(figsize=fsize)
    cor = color
    ax = sns.scatterplot(
        x='Volatilidade', y='Retorno',
        hue='Ind. Sharpe', data=portfolios,
        palette=cor
    )

    norm = plt.Normalize(
        0,
        portfolios['Ind. Sharpe'].max()
    )

    sm = plt.cm.ScalarMappable(cmap=cor, norm=norm)
    sm.set_array([])

    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    port_min_vol = find_port_min_vol(portfolios).T
    port_max_sr = find_port_max_sr(portfolios).T
    plt.scatter(
        x=port_max_sr['Volatilidade'],
        y=port_max_sr['Retorno'], c='red',
        marker='o', s=200
    )
    plt.scatter(
        x = port_min_vol['Volatilidade'],
        y = port_min_vol['Retorno'], c='blue',
        marker='o', s=200
    )

    plt.title('Portfólios')

    if save:
        plt.savefig(save_path + 'portfolios_hd.png', dpi=200)

    return ax if is_return else plt.show()


def plot_eff(
    exp_rets: pd.DataFrame, cov: pd.DataFrame,
    n_points: int=25, risk_free_rate: float=.1,
    show_cml: bool=False, show_ew: bool=False,
    show_gmv: bool=False, style: str='.-',
    size: tuple=(12, 6), is_return: bool=False,
    save: bool=False
):
    """Imprime a fronteira eficiente, baseada nos retornos
    esperados e a matriz de covariância.

    Args:
        exp_rets (pd.DataFrame): retornos esperados dos ativos.
        cov (pd.DataFrame): matriz de covariância dos ativos.
        n_points (int, optional): número de pontos a serem exibidos
        na fronteira. Padrão: 25.
        risk_free_rate (float, optional): taxa livre de risco.
        Padrão: 0.1.
        show_cml (bool, optional): se True, imprime a reta que conecta
        o ativo livre de risco com portfólio de máximo índice de Sharpe.
        Padrão: False.
        show_ew (bool, optional): se True, imprime o portfólio de pesos
        iguais. Padrão: False.
        show_gmv (bool, optional): se True, imprime o GVM portfólio.
        Padrão: False.
        style (str, optional): estilo da linha. Padrão: '.-'.
        size (tuple, optional): tamanho do plot. Padrão: (12, 6).
        is_return (bool, optional): se True, retorna o plot, ao invés de
        apenas imprimí-lo. Padrão: False.
        save (bool, optional): se True, salva o plot em save_path com
        nome de gen_portfolios.png. Padrão: False.

    Returns:
        se is_return = True, retorna um ax do matplotlib.
    """
    weights = optimal_weights(exp_rets, cov, n_points)

    rets = [(1 + exp_rets.dot(w)) ** .5 - 1 for w in weights]
    vols = [vol(w, cov, False) for w in weights]

    ef = pd.DataFrame({'Retornos': rets, 'Volatilidade': vols})
    ax = ef.plot.line(x='Volatilidade', y='Retornos', style=style, figsize=size, legend=False)

    plt.ylabel('Retorno')

    if show_ew:
        n = exp_rets.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = (1 + exp_rets.dot(w_ew)) ** .5 - 1
        v_ew = vol(w_ew, cov, False)

        ax.plot([v_ew], [r_ew], color='goldenrod', marker='o', markersize=10, label='EW')

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = (1 + exp_rets.dot(w_gmv)) ** .5 - 1
        v_gmv = vol(w_gmv, cov, False)

        ax.plot([v_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10, label='GMV')

    if show_cml:
        ax.set_xlim(left=0)

        w_msr = maximize_sr(exp_rets, cov, risk_free_rate)
        r_msr = (1 + exp_rets.dot(w_msr)) ** .5 - 1
        v_msr = vol(w_msr, cov, False)

        # add capital market line
        cml_x = [0, v_msr]
        cml_y = [risk_free_rate, r_msr]

        ax.plot(
            cml_x,
            cml_y,
            color='green',
            marker='o',
            linestyle='dashed',
            markersize=10,
            linewidth=2,
            label='Cap. Market Line'
        )

    plt.legend()

    if save:
        plt.savefig(save_path + 'gen_portfolios.png', dpi=200)

    return ax if is_return else plt.show()


def comparison(vol_opt: float, vol_eq: float, ret_opt: float, ret_eq: float, risk_free_rate: float) -> None:
    """Imprime na tela um comparativo percentual entre a carteira
    de pesos otimizados e a carteira de pesos iguais, e também
    o índice de Sharpe da carteira otimizada.

    Args:
        vol_opt (float): volatilidade da carteira otimizada.
        vol_eq (float): volatlidade da carteira de pesos iguais.
        ret_opt (float): retorno da carteira otimizada.
        ret_eq (float): retorno da carteira de pesos iguais.
        risk_free_rate (float): taxa livre de risco.
    """
    vol_opt = round(vol_opt, 4)
    vol_eq = round(vol_eq, 4)

    sgn = '+'
    if vol_opt > vol_eq:
        sgn = '-'
    print('Volatlidade com os pesos otimizados: '
        f'{vol_opt * 100} %\n'
        'Volatilidade com os pesos iguais: '
        f'{vol_eq * 100} %\n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - vol_opt / vol_eq) * 100, 4)} %\n'
    )

    ret_opt = round(ret_opt, 4)
    ret_eq = round(ret_eq, 4)

    sgn = '+'
    if ret_opt < ret_eq:
        sgn = '-'
    print('Retorno com os pesos otimizados: '
        f'{ret_opt * 100} %\n'
        'Retorno com os pesos iguais: '
        f'{ret_eq * 100} %\n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - ret_opt / ret_eq) * 100, 4)} %\n'
    )

    sharpe_eq = round(sharpe(ret_eq, vol_eq, risk_free_rate), 4)
    sharpe_opt = round(sharpe(ret_opt, vol_opt, risk_free_rate), 4)

    sgn = '+'
    if sharpe_opt < sharpe_eq:
        sgn = '-'
    print('Índice de Sharpe com os pesos otimizados: '
        f'{sharpe_opt}\n'
        'Índice de Sharpe com os pesos iguais: '
        f'{sharpe_eq} \n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - sharpe_opt / sharpe_eq) * 100, 4)} %\n'
    )


def layout_settings(titles: list=[]) -> go.Layout:
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
        spikedistance=1000
    )
    return layout


def plot_heat_go(df: pd.DataFrame, title: str='Correlações', color: str='YlOrRd') -> None:
    """Imprime go.Heatmap com x = df.columns, y = df.columns e z = df.corr().

    Args:
        df (pd.DataFrame): dataframe a partir do qual .corr() será aplicado.
        title (str, optional): título do plot. Padrão:'Correlações'.
        color (str, optional): escala do cor. Padrão: 'YlOrRd'.
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


def plot_returns_sns(rets: pd.Series, titles: list=None, size: tuple=(12, 8)) -> None:
    """Gráfico de barras horizontais dos valores (retornos anualizados)
    de 'rets', com esquema de cores: #de2d26 (#3182bd) para retornos
    negativos (positivos).

    Args:
        rets (pd.Series): série com os retornos anualizados.
        titles (list, optional): título e labels do plot:
        [title, xlabel, ylabel]. Padrão: None.
        size (tuple, optional): tamanho do plot. Padrão: (12, 8).
    """
    rets = rets.sort_values(ascending=True)

    cores = ['#de2d26' if v < 0 else '#3182bd' for v in rets.values]

    plt.subplots(figsize=size)
    sns.barplot(
        x=rets.values,
        y=rets.index,
        palette=cores
    )
    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])


def plot_monthly_returns(
    rets: pd.Series,
    title: str='Retornos Mensais',
    show_mean: bool=True,
    show_median: bool=True,
    size: tuple=(18, 6),
    name: str=None,
    is_return: bool=False
) -> None:
    """Função desenvolvida especificamente para plotar os retornos
    mensais, em barplot. Imprime também a média e a mediana dos
    mesmos, se desejado. Salva o plot se 'name' for dado.

    Args:
        rets (pd.Series): retornos (mensais).
        show_mean (bool, optional): se True, também exibe a média.
        show_median (bool, optional): se True, também exibe a mediana.
        size (tuple): tamanho do plot.
        name (str, optional): se != None, salva o plot em 'save_path'.

    Raises:
        NameError: se len(name) == 0.
    """
    colors = map(lambda r: 'indianred' if r < 0 else 'blue', rets)

    fig, ax = plt.subplots(figsize=size)
    rets.plot.bar(
        ax=ax,
        color=list(colors)
    )

    if show_mean:
        ax.axhline(y=rets.mean(), ls=':', color='green', label='Média')
    if show_median:
        ax.axhline(y=rets.median(), ls='-', color='goldenrod', label='Mediana')

    if show_mean or show_median:
        plt.legend()

    plt.title(title)
    plt.ylabel('Percentual')

    if name:
        if len(name) > 0:
            plt.savefig(save_path + str(name) + '.png', dpi=200)
        else:
            raise NameError('Nome da figura precisa ter ao menos um caracter.')

    return ax if is_return else plt.show()


def plot_lines(
    dfs: list,
    titles: list=[None, None, None],
    plot_in: str='sns',
    size: tuple=(19, 6),
    color: str=None,
    name: str=None,
    is_return: bool=False
):
    """Imprime o lineplot de df.

    Args:
        dfs (list): lista de dataframes.
        titles (list, optional): títulos a serem usados no plot:
        plt.title(titles[0]), plt.xlabel(titles[1]) e
        plt.ylabel(titles[2]). Padrão: [None, None, None].
        plot_in (str, optional): biblioteca a ser usada para o plot.
        Padrão: 'sns'.
        size (tuple, optional): tamanho do plot. Padrão: (19, 6).
        color (str, optional): cor do lineplot. Padrão: 'r'.
        legend_loc (str, optional): localização da legenda. Padrão: 'best'.
        fontsize (int, optional): tamanho da fonte da legenda. Padrão: 10.
        name (str, optional): nome do arquivo para salvar o .png.
        Padrão: None.

    Raises:
        NameError: se len(name) == 0.
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

        if not color:
            ax = sns.lineplot(
                data=df_,
                linewidth=2,
                dashes=False
            )
        else:
            ax = sns.lineplot(
                data=df_,
                linewidth=2,
                dashes=False,
                palette=[color]
            )

        plt.title(titles[0])
        plt.xlabel(titles[1])
        plt.ylabel(titles[2])

        if name:
            if len(name) > 0:
                plt.savefig(save_path + str(name) + '.png', dpi=200)
            else:
                raise NameError('Nome da figura precisa ter ao menos um caracter.')

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
                        name=c
                    )
                )

        fig.show()
    else:
        raise NameError('plot_in inválido. Usar "sns" ou "go".')


def plot_heat_sns(
    df: pd.DataFrame,
    title: str='Correlações',
    color: str='coolwarm',
    size: tuple=(12, 10),
    rotate: bool=False
) -> None:
    """Imprime sns.heatmap de df.corr().

    Args:
        df (pd.DataFrame): dataframe.
        title (str, optional): título do plot. Padrão: to 'Correlações'.
        color (str, optional): cmap. Padrão: 'coolwarm'.
        size (tuple, optional): tamanho do plot. Padrâo: (12, 10).
    """
    correlations = df.corr()

    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(correlations, mask=mask, annot=True,
                         cmap=color, fmt='.2f', linewidths=0.05,
                         vmax=1.0, square=True, linecolor='white')

        if rotate:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel(None)


def plot_opt_comparisons(rets: dict, vols: dict, sharpes: dict, colors: dict) -> None:
    """Imprime um go.Bar com os valores de 'rets', 'vols' e 'sharpes'.

    Args:
        rets (dict): dicionário dos retornos das otimizações;
        Ex: {peso_hrp: ..., peso_min_vol: ...}.
        vols (dict): dicionário das volatilidades das otimizações;
        Ex: {vol_hrp: ..., vol_min_vol: ...}.
        sharpes (dict): dicionário dos índices de Sharpe das otimizações;
        Ex: {sharpe_hrp: ..., sharpe_min_vol: ...}.
        colors (dict): dicionário de cores para cada go.Bar:
        Ex: colors = {
                'rets': ret_cores,
                'vols': vol_cores,
                'sharpes': sharpe_cores
            },
        onde ret_cores é um iterator contendo as cores para cada registro,
        e analogamente para vol_cores e sharpe_cores.
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
        title='Resultados Otimizados',
        xaxis=dict(
                title='Tipo de Otimização',
                showgrid=False
            ),
            yaxis=dict(
                title='Valor Registrado',
                showgrid=False
            ),
            plot_bgcolor="#FFF",
            hoverdistance=100
    )

    fig = go.Figure(data=data, layout=cfg_layout)
    fig.show()


def plot_weights(
    df: pd.DataFrame,
    titles: list=['Carteiras Otimizadas', 'Pesos', None],
    plot_in: str='sns',
    size: tuple=(19,6),
    template: str='ggplot2'
) -> None:
    if plot_in == 'sns':
        df.plot(
            kind='barh', figsize=(19, 6),
            title=titles[0]
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
                    orientation='h'
                )
            )

        fig.update_layout(template=template)
        fig.show()
    else:
        raise NameError('plot_in inválido. Usar "sns" ou "go".')


#-----------------------------------------------------------------
if __name__ == '__main__':
    main()
