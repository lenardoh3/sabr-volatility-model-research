# SABR Volatility Model Research for OTC FX Options

This project was completed as an independent research initiative during my internship at Fenics Market Data. It investigates how well the SABR (Stochastic Alpha Beta Rho) stochastic volatility model fits exotic FX options data and compares its hedging performance against the standard Black-Scholes model.

## Project Description

- Adapt the SABR model for use in OTC FX options, accounting for edge cases like negative interest rates.
- Perform SABR parameter calibration across 63 currency pairs and 6 tenors using Hagan’s approximation.
- Use cubic interpolation and principal component analysis (PCA) to test model robustness and dimensionality reduction.
- Backtest hedging performance using delta and delta-vega strategies under SABR and Black-Scholes.

## Results

- SABR calibration provided a strong fit across a wide range of currencies, especially at-the-money.
- PCA showed minimal parameter redundancy—suggesting limited dimensionality reduction for SABR.
- Delta-vega hedging performance was highly correlated (r > 0.99) between SABR and BS; SABR offered slight improvement in ITM options but no significant edge overall.

(Proprietary Data files have been removed)
