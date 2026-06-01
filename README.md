# GARCH-VaR для защитного портфеля GLD/TLT

Компактный код проекта по анализу и прогнозированию рыночного риска.

## Тема

**Построение и верификация модели рыночного риска для защитного портфеля 50% GLD + 50% TLT.**

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python market_risk_project.py
```

## Что делает код

- загружает дневные цены GLD и TLT;
- считает доходности активов и портфеля 50/50;
- считает описательную статистику;
- выполняет Jarque-Bera, ADF и Ljung-Box тесты;
- строит GARCH(1,1) с t-распределением;
- считает VaR и Expected Shortfall;
- сравнивает GARCH-t, GARCH-normal и исторический VaR;
- проводит backtesting через тест Купика и тест независимости исключений;
- сохраняет расчетные таблицы и графики в `garch_project_outputs`.
