#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="ai_trading_system",
    version="1.0.0",
    description="AI-driven trading system with hard turnover constraint",
    author="Trading AI",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.0",
        "schedule>=1.1.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
)