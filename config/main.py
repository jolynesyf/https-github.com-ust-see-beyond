import os
import sys
import json
import yaml
import pickle
import hashlib
import time
import schedule
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import numpy as np
import pandas as pd

# Custom logging system
class Logger:
    """Simple logging system for reproducibility"""
    
    def __init__(self, log_dir="logs", level="INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.level = level
        self.levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        self.setup_log_file()
        
    def setup_log_file(self):
        """Create new log file for this session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trading_{timestamp}.log"
        
    def log(self, level, message, module=""):
        """Log message with timestamp"""
        if self.levels[level] >= self.levels[self.level]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} | {level:8} | {module:20} | {message}"
            
            # Write to file
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")
            
            # Print to console (colored)
            if level == "ERROR" or level == "CRITICAL":
                print(f"\033[91m{log_entry}\033[0m")  # Red
            elif level == "WARNING":
                print(f"\033[93m{log_entry}\033[0m")  # Yellow
            elif level == "INFO":
                print(f"\033[92m{log_entry}\033[0m")  # Green
            else:
                print(log_entry)

    def info(self, message, module=""):
        self.log("INFO", message, module)
    
    def warning(self, message, module=""):
        self.log("WARNING", message, module)
    
    def error(self, message, module=""):
        self.log("ERROR", message, module)
    
    def debug(self, message, module=""):
        self.log("DEBUG", message, module)

# Global logger instance
logger = Logger()

class ReproducibleEnvironment:
    """Ensure complete reproducibility across runs"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_environment()
        self.experiment_id = self.generate_experiment_id()
        self.setup_directories()
        logger.info(f"Environment initialized with ID: {self.experiment_id}")
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration with validation"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}", "Environment")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}", "Environment")
            raise
    
    def setup_environment(self):
        """Set random seeds for reproducibility"""
        import random
        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info(f"Random seeds set to {seed}", "Environment")
        
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{config_hash}"
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['data', 'models', 'reports', 'backtests', 'logs', 'instructions', 'portfolio']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
        logger.info("Directories created", "Environment")
    
    def save_state(self, state_dict: Dict, name: str):
        """Save experiment state for reproducibility"""
        state_path = Path("models") / f"{self.experiment_id}_{name}.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        logger.info(f"Saved state to {state_path}", "Environment")
    
    def load_state(self, name: str) -> Dict:
        """Load experiment state"""
        state_path = Path("models") / f"{self.experiment_id}_{name}.pkl"
        with open(state_path, 'rb') as f:
            return pickle.load(f)

class MarketSignal(Enum):
    STRONG_BUY = 1
    BUY = 2
    NEUTRAL = 3
    SELL = 4
    STRONG_SELL = 5

@dataclass
class TradingInstruction:
    """Trading instruction data class"""
    timestamp: datetime
    direction: str  # "BUY" or "SELL"
    instrument: str
    quantity: int
    expected_price: float
    confidence: float
    signal_source: str
    reasoning: str
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'instrument': self.instrument,
            'quantity': self.quantity,
            'expected_price': self.expected_price,
            'confidence': self.confidence,
            'signal_source': self.signal_source,
            'reasoning': self.reasoning
        }

class DataCollector:
    """Market data collection and processing"""
    
    def __init__(self, env: ReproducibleEnvironment):
        self.env = env
        self.watchlist = self.load_watchlist()
        logger.info(f"DataCollector initialized with {len(self.watchlist)} symbols", "DataCollector")
    
    def load_watchlist(self) -> List[str]:
        """Load trading watchlist"""
        # S&P 500 top 20 most liquid stocks
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
            'META', 'NVDA', 'JPM', 'V', 'JNJ',
            'WMT', 'PG', 'UNH', 'HD', 'BAC',
            'MA', 'DIS', 'ADBE', 'CRM', 'CSCO'
        ]
    
    def collect_market_data(self) -> Dict:
        """Collect market data using yfinance"""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance", "DataCollector")
            return {}
        
        data = {}
        
        for symbol in self.watchlist:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist = ticker.history(period="1mo", interval="1d")
                
                if len(hist) < 20:
                    logger.warning(f"Insufficient data for {symbol}", "DataCollector")
                    continue
                
                # Calculate technical indicators
                close_prices = hist['Close'].values
                current_price = close_prices[-1]
                
                # Simple Moving Average
                sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else current_price
                
                # RSI
                rsi = self.calculate_rsi(close_prices)
                
                # MACD
                macd = self.calculate_macd(close_prices)
                
                # Volume analysis
                volume = hist['Volume'].values[-1] if 'Volume' in hist.columns else 0
                avg_volume = np.mean(hist['Volume'].values[-20:]) if len(hist) >= 20 else volume
                
                data[symbol] = {
                    'price': current_price,
                    'open': hist['Open'].values[-1] if 'Open' in hist.columns else current_price,
                    'high': hist['High'].values[-1] if 'High' in hist.columns else current_price,
                    'low': hist['Low'].values[-1] if 'Low' in hist.columns else current_price,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'technical': {
                        'sma_20': sma_20,
                        'rsi': rsi,
                        'macd': macd,
                        'price_vs_sma': (current_price / sma_20 - 1) * 100
                    },
                    'timestamp': datetime.now()
                }
                
                logger.debug(f"Collected {symbol}: ${current_price:.2f}, RSI: {rsi:.1f}", "DataCollector")
                
            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}", "DataCollector")
        
        logger.info(f"Collected data for {len(data)} symbols", "DataCollector")
        return data
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return 0.0
        
        # EMA calculations
        def ema(data, period):
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            return np.convolve(data, weights, mode='valid')[-1]
        
        fast_ema = ema(prices[-12:], 12)
        slow_ema = ema(prices[-26:], 26)
        
        return fast_ema - slow_ema

class SignalGenerator:
    """Generate trading signals based on multiple factors"""
    
    def __init__(self, env: ReproducibleEnvironment):
        self.env = env
        logger.info("SignalGenerator initialized", "SignalGenerator")
    
    def generate_signals(self, market_data: Dict) -> List[TradingInstruction]:
        """Generate trading signals from market data"""
        instructions = []
        timestamp = datetime.now()
        
        for symbol, data in market_data.items():
            try:
                # Get technical indicators
                technical = data['technical']
                price = data['price']
                volume = data['volume']
                avg_volume = data['avg_volume']
                
                # Calculate signal scores
                technical_score = self.technical_signal(technical, price)
                volume_score = self.volume_signal(volume, avg_volume)
                price_action_score = self.price_action_signal(data)
                
                # Combined score (weighted average)
                combined_score = (
                    technical_score * 0.6 + 
                    volume_score * 0.3 + 
                    price_action_score * 0.1
                )
                
                # Generate signal
                if combined_score > 0.3:
                    direction = "BUY"
                    confidence = min(combined_score, 0.95)
                    reasoning = f"Technical: {technical_score:.2f}, Volume: {volume_score:.2f}"
                elif combined_score < -0.3:
                    direction = "SELL"
                    confidence = min(-combined_score, 0.95)
                    reasoning = f"Technical: {technical_score:.2f}, Volume: {volume_score:.2f}"
                else:
                    continue  # Skip neutral signals
                
                # Calculate position size
                quantity = self.calculate_position_size(price, confidence)
                
                # Create instruction
                instruction = TradingInstruction(
                    timestamp=timestamp,
                    direction=direction,
                    instrument=symbol,
                    quantity=quantity,
                    expected_price=price,
                    confidence=confidence,
                    signal_source="TECHNICAL_VOLUME",
                    reasoning=reasoning
                )
                
                instructions.append(instruction)
                logger.info(f"Signal: {direction} {symbol} @ ${price:.2f} (conf: {confidence:.2%})", "SignalGenerator")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}", "SignalGenerator")
        
        return instructions
    
    def technical_signal(self, technical: Dict, price: float) -> float:
        """Generate technical trading signal"""
        score = 0.0
        
        # RSI signal
        rsi = technical.get('rsi', 50)
        if rsi < self.env.config['signals']['rsi_oversold']:
            score += 0.4  # Oversold = buy signal
        elif rsi > self.env.config['signals']['rsi_overbought']:
            score -= 0.4  # Overbought = sell signal
        
        # Price vs SMA signal
        price_vs_sma = technical.get('price_vs_sma', 0)
        if price_vs_sma < -2:  # Price > 2% below SMA
            score += 0.3
        elif price_vs_sma > 2:  # Price > 2% above SMA
            score -= 0.3
        
        # MACD signal
        macd = technical.get('macd', 0)
        if macd > 0:
            score += 0.2
        else:
            score -= 0.2
        
        return np.tanh(score)  # Normalize to [-1, 1]
    
    def volume_signal(self, volume: float, avg_volume: float) -> float:
        """Generate volume-based signal"""
        if avg_volume == 0:
            return 0.0
        
        volume_ratio = volume / avg_volume
        if volume_ratio > 1.5:  # High volume
            return 0.3
        elif volume_ratio < 0.5:  # Low volume
            return -0.2
        else:
            return 0.0
    
    def price_action_signal(self, data: Dict) -> float:
        """Generate price action signal"""
        score = 0.0
        
        # Basic price action: compare close to open
        if 'open' in data and 'price' in data:
            daily_return = (data['price'] / data['open'] - 1) * 100
            if daily_return > 1:
                score += 0.1
            elif daily_return < -1:
                score -= 0.1
        
        return score
    
    def calculate_position_size(self, price: float, confidence: float) -> int:
        """Calculate position size based on risk management"""
        max_position_value = self.env.config['trading']['initial_capital'] * \
                           self.env.config['constraints']['max_position_size']
        
        # Adjust position size by confidence
        position_value = max_position_value * confidence
        
        # Calculate share quantity
        shares = position_value / price
        
        # Round to nearest 10 shares
        shares = int(shares // 10 * 10)
        
        return max(10, shares)  # Minimum 10 shares

class TurnoverManager:
    """Manage and enforce daily turnover constraint"""
    
    def __init__(self, env: ReproducibleEnvironment):
        self.env = env
        self.daily_turnover_target = env.config['constraints']['daily_turnover_min']
        self.portfolio = self.load_portfolio()
        self.daily_trades = []
        logger.info(f"TurnoverManager initialized (target: {self.daily_turnover_target:.1%})", "TurnoverManager")
    
    def load_portfolio(self) -> Dict:
        """Load current portfolio from file"""
        portfolio_file = Path("portfolio") / "current_portfolio.json"
        if portfolio_file.exists():
            try:
                with open(portfolio_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Could not load portfolio file", "TurnoverManager")
        
        return {}
    
    def save_portfolio(self):
        """Save portfolio to file"""
        portfolio_file = Path("portfolio") / "current_portfolio.json"
        with open(portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def enforce_turnover_constraint(self, instructions: List[TradingInstruction]) -> List[TradingInstruction]:
        """Enforce minimum daily turnover constraint"""
        if not instructions:
            return instructions
        
        # Calculate current portfolio value
        portfolio_value = self.calculate_portfolio_value()
        
        # Calculate proposed trade value
        proposed_value = self.calculate_proposed_trade_value(instructions)
        
        # Calculate current turnover percentage
        if portfolio_value > 0:
            current_turnover = proposed_value / portfolio_value
        else:
            current_turnover = 1.0  # Empty portfolio = 100% turnover
        
        logger.info(f"Portfolio: ${portfolio_value:,.2f}, Proposed: ${proposed_value:,.2f}, Turnover: {current_turnover:.2%}", "TurnoverManager")
        
        # Check if constraint is satisfied
        if current_turnover >= self.daily_turnover_target:
            logger.info(f"Turnover constraint satisfied: {current_turnover:.2%} >= {self.daily_turnover_target:.2%}", "TurnoverManager")
            return instructions
        
        # Need to increase turnover
        logger.warning(f"Turnover constraint NOT satisfied: {current_turnover:.2%} < {self.daily_turnover_target:.2%}", "TurnoverManager")
        
        # Generate additional trades
        required_additional_value = portfolio_value * self.daily_turnover_target - proposed_value
        additional_instructions = self.generate_additional_trades(required_additional_value)
        
        # Combine instructions
        all_instructions = instructions + additional_instructions
        
        # Verify constraint
        final_value = self.calculate_proposed_trade_value(all_instructions)
        final_turnover = final_value / portfolio_value if portfolio_value > 0 else 1.0
        
        if final_turnover >= self.daily_turnover_target:
            logger.info(f"Turnover after adjustment: {final_turnover:.2%}", "TurnoverManager")
            return all_instructions
        else:
            logger.warning(f"Still below target after adjustment: {final_turnover:.2%}", "TurnoverManager")
            return all_instructions  # Return what we have
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0
        for symbol, position in self.portfolio.items():
            total_value += position.get('value', 0)
        return total_value
    
    def calculate_proposed_trade_value(self, instructions: List[TradingInstruction]) -> float:
        """Calculate total value of proposed trades"""
        total_value = 0.0
        for instr in instructions:
            total_value += instr.quantity * instr.expected_price
        return total_value
    
    def generate_additional_trades(self, required_value: float) -> List[TradingInstruction]:
        """Generate additional trades to meet turnover requirement"""
        additional_instructions = []
        
        if not self.portfolio or required_value <= 0:
            return additional_instructions
        
        # Sort positions by age (sell oldest positions)
        sorted_positions = sorted(
            self.portfolio.items(),
            key=lambda x: x[1].get('entry_time', ''),
            reverse=False  # Oldest first
        )
        
        for symbol, position in sorted_positions:
            if required_value <= 0:
                break
            
            position_value = position.get('value', 0)
            
            # Create sell instruction
            instruction = TradingInstruction(
                timestamp=datetime.now(),
                direction="SELL",
                instrument=symbol,
                quantity=position.get('quantity', 0),
                expected_price=position.get('current_price', 0),
                confidence=0.7,
                signal_source="TURNOVER_CONSTRAINT",
                reasoning=f"Position rotation to meet {self.daily_turnover_target:.1%} turnover requirement"
            )
            
            additional_instructions.append(instruction)
            required_value -= position_value
            
            logger.info(f"Added turnover trade: SELL {symbol} for ${position_value:,.2f}", "TurnoverManager")
        
        return additional_instructions
    
    def update_portfolio(self, instructions: List[TradingInstruction]):
        """Update portfolio with executed trades"""
        for instr in instructions:
            symbol = instr.instrument
            
            if instr.direction == "BUY":
                # Add or update position
                self.portfolio[symbol] = {
                    'quantity': instr.quantity,
                    'entry_price': instr.expected_price,
                    'current_price': instr.expected_price,
                    'value': instr.quantity * instr.expected_price,
                    'entry_time': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
                logger.info(f"Portfolio update: Bought {instr.quantity} {symbol} @ ${instr.expected_price:.2f}", "TurnoverManager")
                
            elif instr.direction == "SELL":
                # Remove position
                if symbol in self.portfolio:
                    del self.portfolio[symbol]
                    logger.info(f"Portfolio update: Sold {instr.quantity} {symbol} @ ${instr.expected_price:.2f}", "TurnoverManager")
        
        # Save updated portfolio
        self.save_portfolio()

class InstructionGenerator:
    """Generate and manage daily trading instructions"""
    
    def __init__(self, env: ReproducibleEnvironment):
        self.env = env
        logger.info("InstructionGenerator initialized", "InstructionGenerator")
    
    def generate_daily_instructions(self, instructions: List[TradingInstruction]):
        """Generate daily instruction files"""
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # JSON file
        json_file = Path("instructions") / f"daily_instructions_{timestamp}.json"
        instructions_dict = [instr.to_dict() for instr in instructions]
        
        with open(json_file, 'w') as f:
            json.dump(instructions_dict, f, indent=2)
        
        # CSV file
        csv_file = Path("instructions") / f"daily_instructions_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Direction', 'Symbol', 'Quantity', 'Price', 'Confidence', 'Source', 'Reason'])
            for instr in instructions:
                writer.writerow([
                    instr.timestamp.strftime("%H:%M:%S"),
                    instr.direction,
                    instr.instrument,
                    instr.quantity,
                    f"${instr.expected_price:.2f}",
                    f"{instr.confidence:.1%}",
                    instr.signal_source,
                    instr.reasoning
                ])
        
        # Human-readable summary
        summary_file = Path("instructions") / f"daily_summary_{timestamp}.txt"
        self.generate_summary_file(summary_file, instructions)
        
        logger.info(f"Generated instructions: {json_file}, {csv_file}, {summary_file}", "InstructionGenerator")
    
    def generate_summary_file(self, filename: Path, instructions: List[TradingInstruction]):
        """Generate human-readable summary"""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DAILY TRADING INSTRUCTIONS - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by direction
            buys = [i for i in instructions if i.direction == "BUY"]
            sells = [i for i in instructions if i.direction == "SELL"]
            
            f.write(f"TOTAL INSTRUCTIONS: {len(instructions)}\n")
            f.write(f"BUY SIGNALS: {len(buys)}\n")
            f.write(f"SELL SIGNALS: {len(sells)}\n\n")
            
            f.write("BUY ORDERS:\n")
            f.write("-" * 80 + "\n")
            total_buy_value = 0
            for i, instr in enumerate(buys, 1):
                value = instr.quantity * instr.expected_price
                total_buy_value += value
                f.write(f"{i:2d}. {instr.instrument:6s} | Qty: {instr.quantity:6d} | "
                       f"Price: ${instr.expected_price:8.2f} | "
                       f"Value: ${value:12,.2f} | "
                       f"Conf: {instr.confidence:5.1%} | "
                       f"Reason: {instr.reasoning}\n")
            
            f.write(f"\nTotal Buy Value: ${total_buy_value:,.2f}\n\n")
            
            f.write("SELL ORDERS:\n")
            f.write("-" * 80 + "\n")
            total_sell_value = 0
            for i, instr in enumerate(sells, 1):
                value = instr.quantity * instr.expected_price
                total_sell_value += value
                f.write(f"{i:2d}. {instr.instrument:6s} | Qty: {instr.quantity:6d} | "
                       f"Price: ${instr.expected_price:8.2f} | "
                       f"Value: ${value:12,.2f} | "
                       f"Conf: {instr.confidence:5.1%} | "
                       f"Reason: {instr.reasoning}\n")
            
            f.write(f"\nTotal Sell Value: ${total_sell_value:,.2f}\n\n")
            
            f.write("SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Net Exposure: ${total_buy_value - total_sell_value:,.2f}\n")
            f.write(f"Gross Value: ${total_buy_value + total_sell_value:,.2f}\n")
            if (total_buy_value + total_sell_value) > 0:
                turnover = (total_buy_value + total_sell_value) / (2 * self.env.config['trading']['initial_capital'])
                f.write(f"Estimated Turnover: {turnover:.2%}\n")
                if turnover >= self.env.config['constraints']['daily_turnover_min']:
                    f.write("✓ Turnover constraint SATISFIED\n")
                else:
                    f.write("✗ Turnover constraint NOT satisfied\n")
            
            f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

class ReportGenerator:
    """Generate daily reports and analytics"""
    
    def __init__(self, env: ReproducibleEnvironment):
        self.env = env
        logger.info("ReportGenerator initialized", "ReportGenerator")
    
    def generate_daily_report(self, market_data: Dict, instructions: List[TradingInstruction]):
        """Generate comprehensive daily report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # HTML report
        html_file = Path("reports") / f"daily_report_{timestamp}.html"
        self.generate_html_report(html_file, market_data, instructions)
        
        # Text summary
        text_file = Path("reports") / f"daily_summary_{timestamp}.txt"
        self.generate_text_report(text_file, market_data, instructions)
        
        logger.info(f"Generated reports: {html_file}, {text_file}", "ReportGenerator")
    
    def generate_html_report(self, filename: Path, market_data: Dict, instructions: List[TradingInstruction]):
        """Generate HTML report"""
        # Group instructions by symbol
        symbol_actions = {}
        for instr in instructions:
            symbol_actions[instr.instrument] = instr.direction
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                .buy {{ background-color: #d4edda; }}
                .sell {{ background-color: #f8d7da; }}
                .neutral {{ background-color: #fff3cd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Daily Trading Report</h1>
                <h2>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
                <p>Experiment ID: {self.env.experiment_id}</p>
            </div>
            
            <div class="section">
                <h2>Market Overview</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>RSI</th>
                        <th>Volume</th>
                        <th>SMA-20</th>
                        <th>Action</th>
                    </tr>
        """
        
        # Add market data rows
        for symbol, data in sorted(market_data.items()):
            action = symbol_actions.get(symbol, "HOLD")
            action_class = action.lower()
            
            html_content += f"""
                    <tr class="{action_class}">
                        <td>{symbol}</td>
                        <td>${data['price']:.2f}</td>
                        <td>{data['technical']['rsi']:.1f}</td>
                        <td>{data['volume']:,}</td>
                        <td>${data['technical']['sma_20']:.2f}</td>
                        <td><strong>{action}</strong></td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Trading Instructions</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Action</th>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Value</th>
                        <th>Confidence</th>
                        <th>Reason</th>
                    </tr>
        """
        
        # Add instruction rows
        total_buy_value = 0
        total_sell_value = 0
        
        for instr in instructions:
            value = instr.quantity * instr.expected_price
            if instr.direction == "BUY":
                total_buy_value += value
            else:
                total_sell_value += value
                
            html_content += f"""
                    <tr class="{instr.direction.lower()}">
                        <td>{instr.timestamp.strftime('%H:%M:%S')}</td>
                        <td><strong>{instr.direction}</strong></td>
                        <td>{instr.instrument}</td>
                        <td>{instr.quantity:,}</td>
                        <td>${instr.expected_price:.2f}</td>
                        <td>${value:,.2f}</td>
                        <td>{instr.confidence:.1%}</td>
                        <td>{instr.reasoning}</td>
                    </tr>
            """
        
        html_content += f"""
                </table>
                
                <div style="margin-top: 20px;">
                    <div class="metric">Total Buy Value: <strong>${total_buy_value:,.2f}</strong></div>
                    <div class="metric">Total Sell Value: <strong>${total_sell_value:,.2f}</strong></div>
                    <div class="metric">Net Exposure: <strong>${total_buy_value - total_sell_value:,.2f}</strong></div>
                    <div class="metric">Gross Value: <strong>${total_buy_value + total_sell_value:,.2f}</strong></div>
                </div>
            </div>
            
            <div class="section">
                <h2>System Information</h2>
                <p><strong>Turnover Constraint:</strong> {self.env.config['constraints']['daily_turnover_min']:.1%} minimum daily</p>
                <p><strong>Max Position Size:</strong> {self.env.config['constraints']['max_position_size']:.1%} of portfolio</p>
                <p><strong>Initial Capital:</strong> ${self.env.config['trading']['initial_capital']:,}</p>
                <p><strong>Paper Trading:</strong> {self.env.config['trading']['paper_trading']}</p>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
    
    def generate_text_report(self, filename: Path, market_data: Dict, instructions: List[TradingInstruction]):
        """Generate text report"""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DAILY TRADING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MARKET SNAPSHOT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Symbol':6} {'Price':>10} {'RSI':>6} {'Volume':>12} {'Action':>8}\n")
            f.write("-" * 80 + "\n")
            
            # Group instructions by symbol
            symbol_actions = {}
            for instr in instructions:
                symbol_actions[instr.instrument] = instr.direction
            
            for symbol, data in sorted(market_data.items()):
                action = symbol_actions.get(symbol, "HOLD")
                f.write(f"{symbol:6} ${data['price']:9.2f} {data['technical']['rsi']:6.1f} "
                       f"{data['volume']:12,.0f} {action:>8}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRADING PERFORMANCE METRICS:\n")
            f.write("-" * 80 + "\n")
            
            # Calculate metrics
            total_trades = len(instructions)
            buy_trades = len([i for i in instructions if i.direction == "BUY"])
            sell_trades = len([i for i in instructions if i.direction == "SELL"])
            
            total_buy_value = sum(i.quantity * i.expected_price 
                                for i in instructions if i.direction == "BUY")
            total_sell_value = sum(i.quantity * i.expected_price 
                                 for i in instructions if i.direction == "SELL")
            
            avg_confidence = np.mean([i.confidence for i in instructions]) if instructions else 0
            
            f.write(f"Total Trades: {total_trades}\n")
            f.write(f"Buy Signals: {buy_trades}\n")
            f.write(f"Sell Signals: {sell_trades}\n")
            f.write(f"Average Confidence: {avg_confidence:.1%}\n")
            f.write(f"Total Buy Value: ${total_buy_value:,.2f}\n")
            f.write(f"Total Sell Value: ${total_sell_value:,.2f}\n")
            f.write(f"Net Exposure: ${total_buy_value - total_sell_value:,.2f}\n")
            
            # Turnover calculation
            portfolio_value = self.env.config['trading']['initial_capital']
            turnover = (total_buy_value + total_sell_value) / (2 * portfolio_value)
            f.write(f"Estimated Turnover: {turnover:.2%}\n")
            
            if turnover >= self.env.config['constraints']['daily_turnover_min']:
                f.write("✓ Turnover constraint: SATISFIED\n")
            else:
                f.write("✗ Turnover constraint: NOT SATISFIED\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

class TradingOrchestrator:
    """Main orchestrator for the trading system"""
    
    def __init__(self):
        self.env = ReproducibleEnvironment()
        self.data_collector = DataCollector(self.env)
        self.signal_generator = SignalGenerator(self.env)
        self.turnover_manager = TurnoverManager(self.env)
        self.instruction_generator = InstructionGenerator(self.env)
        self.report_generator = ReportGenerator(self.env)
        self.is_running = False
        
        logger.info("TradingOrchestrator initialized", "Orchestrator")
    
    def daily_pipeline(self):
        """Execute the complete daily trading pipeline"""
        logger.info("=" * 80, "Orchestrator")
        logger.info("STARTING DAILY TRADING PIPELINE", "Orchestrator")
        logger.info("=" * 80, "Orchestrator")
        
        try:
            # Step 1: Collect market data
            logger.info("Step 1: Collecting market data...", "Orchestrator")
            market_data = self.data_collector.collect_market_data()
            
            if not market_data:
                logger.error("No market data collected. Stopping pipeline.", "Orchestrator")
                return
            
            # Step 2: Generate trading signals
            logger.info("Step 2: Generating trading signals...", "Orchestrator")
            proposed_instructions = self.signal_generator.generate_signals(market_data)
            
            if not proposed_instructions:
                logger.warning("No trading signals generated", "Orchestrator")
            
            # Step 3: Apply turnover constraint
            logger.info("Step 3: Applying turnover constraints...", "Orchestrator")
            final_instructions = self.turnover_manager.enforce_turnover_constraint(proposed_instructions)
            
            # Step 4: Generate daily instructions
            logger.info("Step 4: Generating daily instruction files...", "Orchestrator")
            self.instruction_generator.generate_daily_instructions(final_instructions)
            
            # Step 5: Update portfolio (simulated execution)
            logger.info("Step 5: Updating portfolio...", "Orchestrator")
            self.turnover_manager.update_portfolio(final_instructions)
            
            # Step 6: Generate daily report
            logger.info("Step 6: Generating daily report...", "Orchestrator")
            self.report_generator.generate_daily_report(market_data, final_instructions)
            
            logger.info("=" * 80, "Orchestrator")
            logger.info("DAILY PIPELINE COMPLETED SUCCESSFULLY", "Orchestrator")
            logger.info("=" * 80, "Orchestrator")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", "Orchestrator")
            import traceback
            logger.error(traceback.format_exc(), "Orchestrator")
    
    def start_scheduler(self):
        """Start the scheduled trading system"""
        # Schedule daily pipeline at 8:30 AM
        schedule.every().day.at("08:30").do(self.daily_pipeline)
        
        # Also schedule a market close summary at 4:00 PM
        schedule.every().day.at("16:00").do(self.generate_end_of_day_summary)
        
        logger.info("Scheduler started. Next run at 08:30 AM.", "Orchestrator")
        self.is_running = True
        
        # Keep the script running
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Shutting down scheduler...", "Orchestrator")
                self.is_running = False
            except Exception as e:
                logger.error(f"Scheduler error: {e}", "Orchestrator")
    
    def generate_end_of_day_summary(self):
        """Generate end of day summary"""
        logger.info("Generating end of day summary...", "Orchestrator")
        
        # Create simple EOD report
        timestamp = datetime.now().strftime("%Y%m%d")
        eod_file = Path("reports") / f"eod_summary_{timestamp}.txt"
        
        with open(eod_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"END OF DAY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("System Status: COMPLETED\n")
            f.write(f"Last Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment ID: {self.env.experiment_id}\n")
            f.write(f"Total Instructions Generated Today: {self.count_today_instructions()}\n")
            f.write(f"Portfolio Positions: {len(self.turnover_manager.portfolio)}\n")
            
            portfolio_value = self.turnover_manager.calculate_portfolio_value()
            f.write(f"Portfolio Value: ${portfolio_value:,.2f}\n")
            f.write(f"Initial Capital: ${self.env.config['trading']['initial_capital']:,}\n")
            
            if self.env.config['trading']['initial_capital'] > 0:
                pnl = portfolio_value - self.env.config['trading']['initial_capital']
                f.write(f"P&L: ${pnl:,.2f} ({pnl/self.env.config['trading']['initial_capital']:.2%})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Ready for next trading day.\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"EOD summary saved to {eod_file}", "Orchestrator")
    
    def count_today_instructions(self) -> int:
        """Count instructions generated today"""
        today = datetime.now().strftime("%Y%m%d")
        instruction_file = Path("instructions") / f"daily_instructions_{today}.json"
        
        if instruction_file.exists():
            try:
                with open(instruction_file, 'r') as f:
                    instructions = json.load(f)
                return len(instructions)
            except:
                return 0
        return 0
    
    def run_once(self):
        """Run the pipeline once (for testing)"""
        logger.info("Running single pipeline execution...", "Orchestrator")
        self.daily_pipeline()

def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("AI-DRIVEN TRADING SYSTEM WITH TURNOVER CONSTRAINT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Check for required dependencies
    try:
        import yfinance
        logger.info("All dependencies verified", "Main")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}", "Main")
        print("\nPlease install required packages:")
        print("pip install numpy pandas yfinance schedule python-dotenv")
        return
    
    # Create orchestrator
    orchestrator = TradingOrchestrator()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='AI Trading System')
    parser.add_argument('--mode', choices=['once', 'schedule'], default='once',
                       help='Run mode: once (single run) or schedule (continuous)')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in test mode", "Main")
        # Run a quick test
        orchestrator.run_once()
    elif args.mode == 'schedule':
        logger.info("Starting scheduled trading system", "Main")
        orchestrator.start_scheduler()
    else:
        logger.info("Running single pipeline execution", "Main")
        orchestrator.run_once()
    
    logger.info("System shutdown complete", "Main")
    print("\n" + "=" * 80)
    print("System execution completed")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()