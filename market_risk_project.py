"""
ПРОЕКТ: GARCH-VaR для защитного портфеля GLD/TLT.

Тема:
    Построение и верификация модели рыночного риска для защитного портфеля
    50% GLD + 50% TLT.

Скрипт специально сделан компактным: как в проекте-примере, он загружает данные,
проводит предварительный анализ, строит GARCH(1,1), считает VaR/ES и делает
backtesting. Все результаты сохраняются в папку garch_project_outputs.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
Path(".matplotlib_cache").mkdir(exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from scipy.stats import chi2, norm, t as student_t
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


# ============================================================================
# НАСТРОЙКИ
# ============================================================================

TICKERS = ["GLD", "TLT"]
WEIGHTS = {"GLD": 0.5, "TLT": 0.5}
START_DATE = "2004-11-18"
END_DATE = None
TRAIN_SHARE = 0.80
ROLLING_WINDOW = 252
HIST_WINDOW = 250
OUT = Path("garch_project_outputs")


# ============================================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================


def download_prices() -> tuple[pd.DataFrame, str]:
    """Загружает цены GLD и TLT: сначала Yahoo Finance, затем резервный Stooq."""
    try:
        import yfinance as yf

        data = yf.download(
            TICKERS,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if data.empty:
            raise RuntimeError("Yahoo Finance вернул пустые данные")

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
        else:
            prices = data[["Adj Close"]] if "Adj Close" in data.columns else data[["Close"]]
            prices.columns = TICKERS

        source = "Yahoo Finance via yfinance"
    except Exception as error:
        print(f"[warning] Yahoo Finance не сработал: {error}")
        frames = []
        d1 = START_DATE.replace("-", "")
        d2 = "20991231" if END_DATE is None else END_DATE.replace("-", "")

        for ticker in TICKERS:
            url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d&d1={d1}&d2={d2}"
            df = pd.read_csv(url)
            df["Date"] = pd.to_datetime(df["Date"])
            frames.append(df.set_index("Date")[["Close"]].rename(columns={"Close": ticker}))

        prices = pd.concat(frames, axis=1)
        source = "Stooq daily close prices"

    prices = prices[TICKERS].dropna()
    prices.index = pd.to_datetime(prices.index)
    prices.columns.name = None
    return prices.loc[prices.index >= START_DATE], source


def make_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Считает доходности активов и доходность портфеля 50/50 в процентах."""
    asset_returns = np.log(prices).diff().dropna() * 100
    simple_returns = prices.pct_change().dropna()
    portfolio_simple = sum(WEIGHTS[t] * simple_returns[t] for t in TICKERS)
    portfolio_returns = np.log1p(portfolio_simple) * 100
    portfolio_returns.name = "Portfolio_50_50_GLD_TLT"
    return asset_returns, portfolio_returns


def split_sample(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Делит ряд на обучающую и тестовую выборки без перемешивания."""
    split = int(len(series) * TRAIN_SHARE)
    return series.iloc[:split], series.iloc[split:]


# ============================================================================
# СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================================


def descriptive_stats(returns: pd.Series) -> pd.DataFrame:
    """Формирует таблицу описательной статистики и базовых тестов."""
    jb_stat, jb_p = stats.jarque_bera(returns)
    adf_stat, adf_p, *_ = adfuller(returns)

    rows = [
        ("Средняя дневная доходность", returns.mean()),
        ("Медиана", returns.median()),
        ("Дневная волатильность", returns.std()),
        ("Годовая волатильность", returns.std() * np.sqrt(252)),
        ("Минимум", returns.min()),
        ("Максимум", returns.max()),
        ("Асимметрия", returns.skew()),
        ("Эксцесс", returns.kurtosis() + 3),
        ("Избыточный эксцесс", returns.kurtosis()),
        ("Jarque-Bera statistic", jb_stat),
        ("Jarque-Bera p-value", jb_p),
        ("ADF statistic", adf_stat),
        ("ADF p-value", adf_p),
    ]
    return pd.DataFrame(rows, columns=["Показатель", "Значение"])


def asset_stats(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Считает краткую статистику по GLD и TLT отдельно."""
    table = pd.DataFrame(index=asset_returns.columns)
    table["Средняя доходность"] = asset_returns.mean()
    table["Дневная волатильность"] = asset_returns.std()
    table["Годовая волатильность"] = asset_returns.std() * np.sqrt(252)
    table["Минимум"] = asset_returns.min()
    table["Максимум"] = asset_returns.max()
    table["Эксцесс"] = asset_returns.kurtosis() + 3
    table.index.name = "Актив"
    return table


def ljung_box_table(returns: pd.Series) -> pd.DataFrame:
    """Проверяет автокорреляцию доходностей и квадратов доходностей."""
    rows = []
    for lag in [1, 5, 10, 20]:
        lb_ret = acorr_ljungbox(returns, lags=[lag], return_df=True)
        lb_sq = acorr_ljungbox(returns**2, lags=[lag], return_df=True)
        rows.append(
            {
                "Лаг": lag,
                "p-value доходности": lb_ret["lb_pvalue"].iloc[0],
                "p-value квадраты": lb_sq["lb_pvalue"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# GARCH, VaR, ES
# ============================================================================


def fit_garch(train: pd.Series):
    """Оценивает GARCH(1,1) с t-распределением."""
    model = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    return model.fit(update_freq=0, disp="off")


def model_params(res) -> pd.DataFrame:
    """Собирает параметры модели и показатель персистентности alpha+beta."""
    rows = []
    for name in res.params.index:
        rows.append(
            {
                "Параметр": name,
                "Коэффициент": res.params[name],
                "Ст. ошибка": res.std_err[name],
                "p-value": res.pvalues[name],
            }
        )

    rows.append(
        {
            "Параметр": "alpha[1] + beta[1]",
            "Коэффициент": res.params["alpha[1]"] + res.params["beta[1]"],
            "Ст. ошибка": np.nan,
            "p-value": np.nan,
        }
    )
    return pd.DataFrame(rows)


def std_t_quantile(alpha: float, nu: float) -> float:
    """Квантиль стандартизованного t-распределения."""
    return np.sqrt((nu - 2) / nu) * student_t.ppf(alpha, df=nu)


def std_t_es(alpha: float, nu: float) -> float:
    """Expected Shortfall для стандартизованного t-распределения."""
    q = student_t.ppf(alpha, df=nu)
    pdf = student_t.pdf(q, df=nu)
    raw_es = -pdf * (nu + q**2) / ((nu - 1) * alpha)
    return np.sqrt((nu - 2) / nu) * raw_es


def garch_forecast(res, train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Делает one-step-ahead прогноз VaR и ES на тестовой выборке."""
    mu = res.params["mu"]
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    nu = res.params["nu"]
    prev_var = float(res.conditional_volatility.iloc[-1] ** 2)
    prev_eps = float(train.iloc[-1] - mu)
    rows = []

    for date, ret in test.items():
        var = omega + alpha * prev_eps**2 + beta * prev_var
        sigma = np.sqrt(max(var, 0))

        row = {"Дата": date, "Доходность": ret, "GARCH sigma": sigma}
        for a, label in [(0.05, "95"), (0.01, "99")]:
            row[f"VaR {label}% GARCH-t"] = mu + sigma * std_t_quantile(a, nu)
            row[f"ES {label}% GARCH-t"] = mu + sigma * std_t_es(a, nu)
            row[f"Exception {label}% GARCH-t"] = int(ret < row[f"VaR {label}% GARCH-t"])

        row["VaR 95% GARCH-normal"] = mu + sigma * norm.ppf(0.05)
        row["Exception 95% GARCH-normal"] = int(ret < row["VaR 95% GARCH-normal"])
        rows.append(row)

        # Обновляем рекурсию после появления фактической доходности.
        prev_eps = float(ret - mu)
        prev_var = float(var)

    return pd.DataFrame(rows).set_index("Дата")


def add_historical_var(forecast: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Добавляет исторический VaR как простую альтернативу GARCH."""
    result = forecast.copy()
    for a, label in [(0.05, "95"), (0.01, "99")]:
        hist_var = returns.rolling(HIST_WINDOW).quantile(a).shift(1).reindex(result.index)
        result[f"VaR {label}% Historical"] = hist_var
        result[f"Exception {label}% Historical"] = (result["Доходность"] < hist_var).astype(int)
    return result


# ============================================================================
# BACKTESTING
# ============================================================================


def kupiec_test(exceptions: pd.Series, alpha: float) -> tuple[float, float]:
    """Тест Купика: проверяет, совпадает ли доля исключений с alpha."""
    exc = exceptions.astype(int).to_numpy()
    n = len(exc)
    x = exc.sum()
    if x == 0 or x == n:
        return np.nan, np.nan

    p_hat = x / n
    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - chi2.cdf(lr, 1)


def independence_test(exceptions: pd.Series) -> tuple[float, float]:
    """Тест независимости: проверяет, не идут ли исключения сериями."""
    exc = exceptions.astype(int).to_numpy()
    prev, curr = exc[:-1], exc[1:]
    n00 = ((prev == 0) & (curr == 0)).sum()
    n01 = ((prev == 0) & (curr == 1)).sum()
    n10 = ((prev == 1) & (curr == 0)).sum()
    n11 = ((prev == 1) & (curr == 1)).sum()

    def ll(success: int, fail: int, p: float) -> float:
        p = min(max(p, 1e-12), 1 - 1e-12)
        return success * np.log(p) + fail * np.log(1 - p)

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    lr = -2 * (ll(n01 + n11, n00 + n10, pi) - ll(n01, n00, pi01) - ll(n11, n10, pi11))
    return lr, 1 - chi2.cdf(lr, 1)


def backtesting_table(forecast: pd.DataFrame) -> pd.DataFrame:
    """Собирает итоговую таблицу backtesting для нескольких моделей VaR."""
    specs = [
        ("GARCH(1,1)-t", "95%", "Exception 95% GARCH-t", 0.05),
        ("GARCH(1,1)-t", "99%", "Exception 99% GARCH-t", 0.01),
        ("GARCH-normal", "95%", "Exception 95% GARCH-normal", 0.05),
        ("Historical", "95%", "Exception 95% Historical", 0.05),
        ("Historical", "99%", "Exception 99% Historical", 0.01),
    ]
    rows = []

    for model, level, col, alpha in specs:
        exc = forecast[col].dropna().astype(int)
        lr_uc, p_uc = kupiec_test(exc, alpha)
        lr_ind, p_ind = independence_test(exc)
        rows.append(
            {
                "Модель": model,
                "Уровень": level,
                "Наблюдений": len(exc),
                "Исключений": int(exc.sum()),
                "Ожидаемо": len(exc) * alpha,
                "Доля": exc.mean(),
                "Kupiec p-value": p_uc,
                "Independence p-value": p_ind,
                "Вердикт": "ACCEPT" if p_uc > 0.05 else "REJECT",
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# ГРАФИКИ
# ============================================================================


def save_fig(fig: plt.Figure, name: str) -> None:
    """Сохраняет график в папку результатов."""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_plots(prices: pd.DataFrame, asset_ret: pd.DataFrame, port_ret: pd.Series, res, forecast: pd.DataFrame) -> None:
    """Создает основные графики проекта."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    (prices / prices.iloc[0] * 100).plot(ax=ax[0, 0])
    ax[0, 0].set_title("Нормированные цены GLD и TLT")
    port_ret.plot(ax=ax[0, 1], color="purple", linewidth=0.8)
    ax[0, 1].axhline(0, color="black", linewidth=0.7)
    ax[0, 1].set_title("Доходности портфеля, %")
    ax[1, 0].hist(port_ret, bins=90, density=True, alpha=0.7, color="orange")
    x = np.linspace(port_ret.min(), port_ret.max(), 300)
    ax[1, 0].plot(x, norm.pdf(x, port_ret.mean(), port_ret.std()), color="red")
    ax[1, 0].set_title("Распределение доходностей")
    stats.probplot(port_ret, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title("QQ-график")
    save_fig(fig, "01_overview.png")

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    port_ret.abs().plot(ax=ax[0, 0], linewidth=0.8)
    ax[0, 0].set_title("Абсолютные доходности")
    (port_ret**2).plot(ax=ax[0, 1], color="tomato", linewidth=0.8)
    ax[0, 1].set_title("Квадраты доходностей")
    (port_ret.rolling(ROLLING_WINDOW).std() * np.sqrt(252)).plot(ax=ax[1, 0], color="darkgreen")
    ax[1, 0].set_title("Скользящая годовая волатильность")
    plot_acf(port_ret**2, lags=30, ax=ax[1, 1])
    ax[1, 1].set_title("ACF квадратов доходностей")
    save_fig(fig, "02_volatility.png")

    std_resid = pd.Series(res.std_resid).dropna()
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    std_resid.iloc[-700:].plot(ax=ax[0, 0], linewidth=0.8)
    ax[0, 0].set_title("Стандартизованные остатки")
    ax[0, 1].hist(std_resid, bins=60, density=True, alpha=0.7)
    ax[0, 1].set_title("Распределение остатков")
    stats.probplot(std_resid, dist="norm", plot=ax[1, 0])
    ax[1, 0].set_title("QQ-график остатков")
    plot_acf(std_resid**2, lags=30, ax=ax[1, 1])
    ax[1, 1].set_title("ACF квадратов остатков")
    save_fig(fig, "03_garch_diagnostics.png")

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for axis, level in zip(ax, ["95", "99"]):
        exc = forecast[f"Exception {level}% GARCH-t"] == 1
        axis.plot(forecast.index, forecast["Доходность"], label="Факт", linewidth=0.8)
        axis.plot(forecast.index, forecast[f"VaR {level}% GARCH-t"], color="red", label=f"VaR {level}%")
        axis.scatter(forecast.index[exc], forecast.loc[exc, "Доходность"], color="darkred", s=18, label="Исключения")
        axis.set_title(f"Backtesting VaR {level}%")
        axis.legend()
    save_fig(fig, "04_var_backtesting.png")

    fig, ax = plt.subplots(2, 1, figsize=(15, 9))
    asset_ret.plot(ax=ax[0], linewidth=0.7)
    ax[0].set_title("Доходности GLD и TLT")
    asset_ret["GLD"].rolling(ROLLING_WINDOW).corr(asset_ret["TLT"]).plot(ax=ax[1], color="darkblue")
    ax[1].axhline(0, color="black", linewidth=0.7)
    ax[1].set_title("Скользящая корреляция GLD и TLT")
    save_fig(fig, "05_assets_correlation.png")


# ============================================================================
# СОХРАНЕНИЕ РАСЧЕТОВ
# ============================================================================


def save_outputs(prices, asset_ret, port_ret, stats_table, asset_table, lb_table, params, forecast, backtest):
    """Сохраняет только расчетные таблицы: CSV и один общий Excel-файл."""
    OUT.mkdir(exist_ok=True)
    corr = asset_ret.corr()
    snapshot = forecast.iloc[-1][
        ["GARCH sigma", "VaR 95% GARCH-t", "VaR 99% GARCH-t", "ES 95% GARCH-t", "ES 99% GARCH-t"]
    ].to_frame("Значение")

    tables = {
        "prices": prices,
        "asset_returns_pct": asset_ret,
        "portfolio_returns_pct": port_ret.to_frame("Portfolio return, %"),
        "portfolio_descriptive_statistics": stats_table,
        "asset_descriptive_statistics": asset_table,
        "ljung_box_tests": lb_table,
        "correlation_matrix": corr,
        "garch_parameters": params,
        "risk_forecasts_backtest": forecast,
        "backtesting_summary": backtest,
        "risk_snapshot": snapshot,
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig", index_label="Индекс")
    with pd.ExcelWriter(OUT / "project_tables.xlsx") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index_label="Индекс")


# ============================================================================
# ЗАПУСК
# ============================================================================


def main() -> None:
    """Запускает весь проект: данные, модель, графики и таблицы."""
    OUT.mkdir(exist_ok=True)
    prices, source = download_prices()
    asset_ret, port_ret = make_returns(prices)
    train, test = split_sample(port_ret)

    stats_table = descriptive_stats(port_ret)
    asset_table = asset_stats(asset_ret)
    lb_table = ljung_box_table(port_ret)
    res = fit_garch(train)
    params = model_params(res)
    forecast = add_historical_var(garch_forecast(res, train, test), port_ret)
    backtest = backtesting_table(forecast)

    make_plots(prices, asset_ret, port_ret, res, forecast)
    save_outputs(prices, asset_ret, port_ret, stats_table, asset_table, lb_table, params, forecast, backtest)

    persistence = params.loc[params["Параметр"] == "alpha[1] + beta[1]", "Коэффициент"].iloc[0]
    print("Проект успешно рассчитан.")
    print(f"Период: {prices.index.min().date()} - {prices.index.max().date()}")
    print(f"Наблюдений: {len(port_ret)}; train: {len(train)}; test: {len(test)}")
    print(f"GARCH alpha + beta: {persistence:.4f}")
    print(f"Результаты: {OUT.resolve()}")
    print("\nBacktesting:")
    print(backtest)


if __name__ == "__main__":
    main()
