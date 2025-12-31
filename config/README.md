# AI Trading System with Hard Turnover Constraint

## Features
- **Hard Constraint**: Minimum 10% daily portfolio turnover
- **AI-Driven Signals**: Technical + volume-based signal generation
- **Complete Pipeline**: Data collection → Signal generation → Constraint enforcement → Execution
- **Reproducible**: All random seeds fixed, full logging, state persistence
- **Daily Reports**: HTML and text reports with metrics
- **Paper Trading**: Simulated execution with portfolio tracking

## Quick Start

1. **Install dependencies**:
```bash
pip install numpy pandas yfinance schedule python-dotenv PyYAML

2. **Run the system**:
# Single run
python main.py

# Scheduled mode (runs daily at 8:30 AM)
python main.py --mode schedule

# Test mode
python main.py --test

3. **File structure**:
.
├── config.yaml              # Configuration
├── main.py                 # Main trading system
├── requirements.txt        # Dependencies
├── data/                   # Market data cache
├── logs/                   # System logs
├── instructions/           # Daily trading instructions
├── reports/                # Daily reports
├── portfolio/             # Portfolio state
├── models/                # Saved models/states
└── backtests/             # Backtest results