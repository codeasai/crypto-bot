import numpy as np
import pandas as pd
from typing import Dict, List
from core.evaluator import ModelEvaluator

class ValidationMetrics:
    def __init__(self, evaluator: ModelEvaluator, validation_data: pd.DataFrame):
        self.evaluator = evaluator
        self.validation_data = validation_data
        
    def calculate_metrics(self) -> Dict:
        """คำนวณ metrics ทั้งหมด"""
        returns = self._calculate_returns()
        trades = self._analyze_trades()
        
        metrics = {
            "validation_metrics": {
                "total_return": float(returns["total_return"]),
                "sharpe_ratio": float(returns["sharpe_ratio"]),
                "max_drawdown": float(returns["max_drawdown"]),
                "win_rate": float(trades["win_rate"]),
                "profit_factor": float(trades["profit_factor"]),
                "avg_trade_profit": float(trades["avg_trade_profit"]),
                "total_trades": int(trades["total_trades"]),
                "winning_trades": int(trades["winning_trades"]),
                "losing_trades": int(trades["losing_trades"])
            },
            "benchmark_comparison": {
                "buy_and_hold_return": float(returns["buy_and_hold"]),
                "outperformance": float(returns["total_return"] - returns["buy_and_hold"])
            },
            "risk_metrics": {
                "volatility": float(returns["volatility"]),
                "sortino_ratio": float(returns["sortino_ratio"]),
                "calmar_ratio": float(returns["calmar_ratio"]),
                "var_95": float(returns["var_95"])
            }
        }
        
        return metrics
    
    def _calculate_returns(self) -> Dict:
        """คำนวณผลตอบแทนและ risk metrics"""
        portfolio_values = self._simulate_portfolio()
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        
        # คำนวณ max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = -np.min(drawdown) * 100
        
        # คำนวณ Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252)
        
        # คำนวณ Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else 0
        
        # คำนวณ VaR
        var_95 = np.percentile(returns, 5) * 100
        
        # คำนวณ buy and hold return
        buy_hold_return = (self.validation_data["close"].iloc[-1] / 
                          self.validation_data["close"].iloc[0] - 1) * 100
        
        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "var_95": var_95,
            "buy_and_hold": buy_hold_return
        }
    
    def _analyze_trades(self) -> Dict:
        """วิเคราะห์ผลการเทรด"""
        trades = self._get_trades()
        
        winning_trades = trades[trades["profit"] > 0]
        losing_trades = trades[trades["profit"] <= 0]
        
        total_trades = len(trades)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        avg_trade_profit = np.mean(trades["profit"]) if total_trades > 0 else 0
        
        gross_profit = np.sum(winning_trades["profit"]) if len(winning_trades) > 0 else 0
        gross_loss = abs(np.sum(losing_trades["profit"])) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_profit": avg_trade_profit,
            "total_trades": total_trades,
            "winning_trades": winning_trades_count,
            "losing_trades": losing_trades_count
        }
    
    def _simulate_portfolio(self) -> np.ndarray:
        """จำลองผลการเทรด"""
        # TODO: Implement portfolio simulation
        return np.array([1000.0])  # Placeholder
    
    def _get_trades(self) -> pd.DataFrame:
        """ดึงข้อมูลการเทรด"""
        # TODO: Implement trade extraction
        return pd.DataFrame({"profit": [0.0]})  # Placeholder 