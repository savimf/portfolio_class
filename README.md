# Portfolio Class

## 1. Objetivo
Automatizar o cálculo de métricas quantitativas e a comparação entre elas para vários portfólios. Funciona em paralelo com o módeulo `quant_tools.py`.

## 2. Atributos
## 2.1 Instância
Um objeto `Portfolio` assume a forma

```python
p = Portfolio(name: str, tickers: list, start: datetime, end: datetime, source: str, crypto: bool)
```
onde

- `name`: nome (este parâmetro fica armazenado num dicionário interno à classe e referencia o `Portfolio`; sendo assim, cada nome é único);
- `tickers`: lista de ativos;
- `start` e `end`: data de início e término, respectivamente, da coleta dos preços de fechamento dos ativos contidos em `tickers`, através de `source`;
- `source`: fonte da coleta dos dados (por enquanto, somente investing.com, `source = 'iv'`, ou Yahoo Finance `source = 'yf'`). Padrão: `'iv'`.;
- `crypto`: se os ativos em `tickers` forem criptomoedas, setar como `True` para baixá-los do Yahoo Finance. Este paramêtro deve-se à formatação da string a ser utilizada na API na hora de realizar o download.

### 2.2 Classe
1. `registered`: dicionário que armazena os nomes dos `Portfolio`s. Dado `p` como definido acima, teríamos

```python
registered = {'name': p}
```
tal que podemos referenciar o `Portfolio` pelo seu nome. Útil na comparação entre os `Portfólio`s.

2. `delta`: tolerância para verificar se a soma dos pesos de um `Portfolio` é igual a 1. Padrão: 0.001. Isto é, quando um `Portfolio` é instanciado, há a verificação (contida na propriedade `Portfolio.weights`)

```python
np.abs(1 - np.sum(weights)) > Portfolio.delta
```
se `True`, `ValueError` será lançado.

3. `overwrite`: se `True`, permite a sobrescrita dos nomes em `Portfolio.registered`. Padrão: `False`.

## 3. Instanciando um Portfolio
### 3.1 Não-Vazio
Para iniciarmos um objeto `Portfolio`, portanto, basta seu nome e uma lista de tickers (ativos). Apesar da data de início e término não serem obrigatórias, quando não as inserimos estamos criando um `Portfolio` com o DataFrame de preços vazios (podem ser preenchidos posteriormente). Isso é útil para situações onde os preços não podem ser baixados dos métodos `yf.Ticker.history()` do Yahoo Finance ou `iv.get_stock_historical_data()` do investing.com (até outras opções serem implementadas). Quando o `Portfolio` é instanciado, seu nome é adicionado à `Portfolio.registered`, os ativos recebem, por padrão, pesos iguais e os preços de fechamento são setados à propriedade `prices`.

Por exemplo, para instanciarmos um `Portfolio` com os ativos ITSA4, SULA11 e SAPR11, entre o período 01/01/2020 e 01/01/2021, do investing.com, fazemos

```python
p1 = Portfolio(
    name='exemplo',
    tickers=['ITSA4', 'SULA11', 'SAPR11'],
    start=datetime(2020, 1, 1),
    end=datetime(2021, 1, 1)
)
```

Visualizamos os preços de fechamento, dos três ativos, com `p1.prices`. Se fizermos, `p1.weights`, obtemos

```python
p1.weights -> array([0.33333333, 0.33333333, 0.33333333])
```

Se quisermos alterá-los, digamos, para 50%, 30% e 20%, fazemos

```python
p1.weights = np.array([.5, .3, .2])
```

Para criptoativos, digamos, BTC e ETH, baixamos do Yahoo Finance com

```python
p2 = Portfolio(
    name='exemplo-cripto',
    tickers=['BTC-USD', 'ETH-USD'],
    start=datetime(2020, 1, 1),
    end=datetime(2021, 1, 1)
    source='yf',
    crypto=True
)
```

### 3.2 Vazio
Os preços de fechamento do IBOVESPA, por exemplo, são adquiridos através do método `search_quotes()` do investing.com (ainda não implementado à classe). Sendo assim, para termos um `Portfolio` do IBOVESPA, podemos baixar os preços separadamente, pelo método acima, criar um `Portfolio` vazio e setar a propriedade `prices` com os preços baixados. No módulo `quant_tools.py` temos o método `market_index()` que realiza o download:

```python
ibvp_past_pr = qt.market_index('bvsp', datetime(2020, 1, 1), datetime(2021, 1, 1))

ibvp_past = Portfolio('IBVP Past', ['IBVP'])
ibvp_past.prices = ibvp_past_pr
ibvp_past.dates = (start_past_dt, end_past_dt)
Portfolio.register(ibvp_past)
```

Note, porém, que também é necessário setar as datas e registrar o objeto. Ao executarmos `Portfolio.registered`, obtemos um dicionário com os nomes dos três `Portfolio`s instanciados, como chaves, e seus endereços de memória como valor.

## 4. Propriedades

1. `name`

2. `tickers`

3. `weights`

4. `dates`

5. `prices`

## 5. Métodos
### 5.1 Mágicos

1. `__len__`: retorna `len(self.tickers)`

2. `__add__`: é possível adiconar dois `Portfolio`s. `p = p1 + p2` gera um novo `Portfolio`, realizando a concatenação de `p1.prices` e `p2.prices`, tal que `p.name = p1.name + p2.name` (pode ser alterado) e `p.tickers` é dado pelas colunas do DataFrame resultante de

```python
pd.concat(
    [p1, p2],
    axis=1,
    join='inner'
)
```
Também é possível realizar um `outer join`, mas a implementação deve ser feita através de `p1.__add__(p2, join='outer')`.

### 5.2 Instância
Seja `p` um `Portfolio`.

1. `p.d_returns()`: retornos diários do portfólio como um todo ou dos ativos que o compõem (ver parâmetros)

2. `p.m_returns()`: análogo a `d_returns()`, mas para retornos mensais

3. `p.total_returns()`: variação total do período (Pf - Pi) / Pi, onde Pf (Pi) é o preço final (inicial), anualizado ou não

4. `p.acm_returns()`: retornos acumulados, do portfólio como um todo ou dos ativos que o compõem

5. `p.portfolio_return()`: média ponderada dos retornos totais, i.e., `p.total_returns().dot(p.weights)`

6. `p.covariance()`: matriz de covariância

7. `p.benchmark([portfolios])`: plota o benchmark de `p` contra os `Portfolio`s em portfolios

8. `p.beta(benchmark)`: beta do portfólio, tendo como benchmark (e.g., IBOVESPA) `benchmark`

9. `p.volatility()`: volatilidades diárias, mensais e anuais, do portfólio como um todo ou dos ativos que o compõem

10. `p.s_index()`: índice de Sharpe, ou Sortino, anualizado

11. `p.var()`: VaR histórico, paramétrico **ou** paramétrico ajustado

12. `p.cvar()`: CVaR histórico, paramétrico **ou** paramétrico ajustado

13. `p.all_vars()`: VaR histórico, paramétrico **e** paramétrico ajustado

14. `p.downside()`: downside (volatilidade dos retornos negativos) do portfólio

15. `p.upside()`: upside (volatilidade dos retornos positivos) do portfólio

16. `p.rol_drawdown()`: drawdown móvel

17. `p.calc_skewness()`: skewness

18. `p.calc_curtose()`: curtose

19. `p.shapiro_test()`: teste de Shapiro para verificar a hipótese de normalidade

20. `p.metrics()`: DataFrame contendo várias métricas do pórtfolio (e.g., todas listadas acima)

21. `p.transfer(new_name: str, new_weights: np.array)`: cria um novo `Portfolio`, realizando uma cópia de `p`, de nome `new_name`e pesos `new_weights`. Útil para compararmos portfólios de mesmos ativos e mesmas datas, mas de pesos diferentes


### 5.3 Classe
Seja `p` um `Portfolio`.

1. `register(p)`: adiciona o `p` ao dicionário `Portfolio.registered`

2. `unregister(name)`: remove o `Portfolio` de nome 'name' do dicionário `Portfolio.registered`

3. `all_rets()`: retornos de todos os `Portfolio`s registrados, anualizado ou não

4. `all_vols()`: volatilidade de todos os `Portfolio`s registrados

5. `all_sindex()`: índices de Sharpe, ou Sortino, de todos os `Portfolio`s registrados

6. `all_weights()`: pesos de todos os `Portfolio`s registrados

7. `all_metrics([portfolios])`: aplica o método `metrics()` a todos os `Portfolio`s em `portfolios`, concatena-os e retorna um DataFrame
