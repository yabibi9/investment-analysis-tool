
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Any, Tuple

class CompanyAnalyzer:
    """A comprehensive company financial analyzer"""
    
    def __init__(self, ticker: str) -> None:
        """Initialize analyzer with company ticker"""
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        self._load_data()
        
    def _load_data(self) -> None:
        """Load all necessary data from yfinance"""
        try:
            self.info = self.stock.info
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cash_flow = self.stock.cashflow
            self.quarterly_financials = self.stock.quarterly_financials
            self.history = self.stock.history(period='2y')
        except Exception as e:
            raise ValueError(f"Failed to load data for {self.ticker}: {str(e)}")

    def get_basic_info(self) -> Dict:
        """Get comprehensive company information"""
        return {
            'company_info': {
                'name': self.info.get('longName'),
                'sector': self.info.get('sector'),
                'industry': self.info.get('industry'),
                'country': self.info.get('country'),
                'website': self.info.get('website'),
                'employees': self.info.get('fullTimeEmployees')
            },
            'market_data': {
                'market_cap': self.info.get('marketCap'),
                'current_price': self.info.get('currentPrice'),
                'fifty_two_week_high': self.info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': self.info.get('fiftyTwoWeekLow'),
                'volume': self.info.get('volume'),
                'avg_volume': self.info.get('averageVolume')
            },
            'key_stats': {
                'pe_ratio': self.info.get('trailingPE'),
                'forward_pe': self.info.get('forwardPE'),
                'peg_ratio': self.info.get('pegRatio'),
                'beta': self.info.get('beta'),
                'dividend_yield': self.info.get('dividendYield'),
                'market_cap': self.info.get('marketCap')
            }
        }
    
    def calculate_financial_ratios(self) -> Dict:
        """Calculate comprehensive financial ratios"""
        try:
            ratios = {
                'profitability': self._calculate_profitability_ratios(),
                'liquidity': self._calculate_liquidity_ratios(),
                'solvency': self._calculate_solvency_ratios(),
                'efficiency': self._calculate_efficiency_ratios(),
                'growth': self._calculate_growth_ratios()
            }
            return ratios
        except Exception as e:
            return {'error': f"Error calculating ratios: {str(e)}"}

    def _calculate_profitability_ratios(self) -> Dict:
        """Calculate profitability ratios"""
        try:
            revenue = self.financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in self.financials.index else None
            gross_profit = self.financials.loc['Gross Profit'].iloc[0] if 'Gross Profit' in self.financials.index else 0
            operating_income = self.financials.loc['Operating Income'].iloc[0] if 'Operating Income' in self.financials.index else None 
            net_income = self.financials.loc['Net Income'].iloc[0] if 'Net Income' in self.financials.index else None
            
            return {
            'gross_margin': f"{(gross_profit / revenue) * 100:.2f}%" if revenue and gross_profit else "N/A",
            'net_margin': f"{(net_income / revenue) * 100:.2f}%" if revenue and net_income else "N/A",
            'return_on_assets': self._calculate_roa() if self._calculate_roa() else "N/A",
            'return_on_equity': self._calculate_roe() if self._calculate_roe() else "N/A"
            }
        except Exception:
            return {}

    def _calculate_liquidity_ratios(self) -> Dict:
        """Calculate liquidity ratios"""
        try:
            current_assets = self.balance_sheet.loc['Total Current Assets'].iloc[0]
            current_liabilities = self.balance_sheet.loc['Total Current Liabilities'].iloc[0]
            inventory = self.balance_sheet.loc.get('Inventory', pd.Series([0])).iloc[0]
            cash = self.balance_sheet.loc.get('Cash', pd.Series([0])).iloc[0]
            
            return {
                'current_ratio': current_assets / current_liabilities,
                'quick_ratio': (current_assets - inventory) / current_liabilities,
                'cash_ratio': self.balance_sheet.loc['Cash'].iloc[0] / current_liabilities
            }
        except Exception:
            return {'current_ratio': 'N/A',
            'quick_ratio': 'N/A',
            'cash_ratio': 'N/A'}
        

    def _calculate_solvency_ratios(self) -> Dict:
        """Calculate solvency ratios"""
        try:
            total_assets = self.balance_sheet.loc['Total Assets'].iloc[0]
            total_liabilities = self.balance_sheet.loc['Total Liabilities'].iloc[0]
            total_equity = self.balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            
            return {
                'debt_to_equity': total_liabilities / total_equity,
                'debt_to_assets': total_liabilities / total_assets,
                'equity_multiplier': total_assets / total_equity
            }
        except Exception:
            return {}

    def _calculate_efficiency_ratios(self) -> Dict:
        """Calculate efficiency ratios"""
        try:
            revenue = self.financials.loc['Total Revenue'].iloc[0]
            total_assets = self.balance_sheet.loc['Total Assets'].iloc[0]
            receivables = self.balance_sheet.loc.get('Net Receivables', pd.Series([0])).iloc[0]
            inventory = self.balance_sheet.loc.get('Inventory', pd.Series([0])).iloc[0]
            
            return {
                'asset_turnover': revenue / total_assets,
                'receivables_turnover': revenue / receivables if receivables else None,
                'inventory_turnover': (self.financials.loc['Cost Of Revenue'].iloc[0] / inventory) if inventory else None
            }
        except Exception:
            return {}

    def _calculate_growth_ratios(self) -> Dict:
        """Calculate growth ratios"""
        try:
            revenue_growth = self.financials.loc['Total Revenue'].pct_change() * 100
            earnings_growth = self.financials.loc['Net Income'].pct_change() * 100
            
            return {
                'revenue_growth': revenue_growth.mean(),
                'earnings_growth': earnings_growth.mean(),
                'quarterly_revenue_growth': self._calculate_quarterly_growth('Total Revenue'),
                'quarterly_earnings_growth': self._calculate_quarterly_growth('Net Income')
            }
        except Exception:
            return {}

    def _calculate_quarterly_growth(self, metric: str) -> Optional[float]:
        """Calculate quarterly growth for a specific metric"""
        try:
            if metric in self.quarterly_financials.index:
                return (self.quarterly_financials.loc[metric].pct_change() * 100).mean()
            return None
        except Exception:
            return None

    def analyze_market_performance(self) -> Dict:
        """Analyze market performance and technical indicators"""
        try:
            prices = self.history['Close']
            returns = prices.pct_change().dropna()
            
            return {
                'technical_indicators': self._calculate_technical_indicators(prices),
                'volatility_metrics': self._calculate_volatility_metrics(returns),
                'trend_analysis': self._analyze_price_trends(prices)
            }
        except Exception as e:
            return {'error': f"Error analyzing market performance: {str(e)}"}

    def _calculate_technical_indicators(self, prices: pd.Series) -> Dict:
        """Calculate technical indicators"""
        try:
            sma_50 = prices.rolling(window=50).mean().iloc[-1]
            sma_200 = prices.rolling(window=200).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            return {
                'sma_50': sma_50,
                'sma_200': sma_200,
                'price_to_sma50': current_price / sma_50,
                'price_to_sma200': current_price / sma_200,
                'rsi': self._calculate_rsi(prices)
            }
        except Exception:
            return {}

    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict:
        """Calculate volatility metrics"""
        try:
            daily_volatility = returns.std()
            annual_volatility = daily_volatility * np.sqrt(252)
            
            return {
                'daily_volatility': daily_volatility,
                'annual_volatility': annual_volatility,
                'rolling_volatility': self._calculate_rolling_volatility(returns),
                'volatility_trend': self._analyze_volatility_trend(returns)
            }
        except Exception:
            return {}

    def _calculate_rolling_volatility(self, returns: pd.Series, window: int = 30) -> Dict:
        """Calculate rolling volatility"""
        try:
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            return {
                'current': rolling_vol.iloc[-1],
                'mean': rolling_vol.mean(),
                'max': rolling_vol.max(),
                'min': rolling_vol.min()
            }
        except Exception:
            return {}

    def _analyze_volatility_trend(self, returns: pd.Series) -> str:
        """Analyze the trend in volatility"""
        try:
            recent_vol = returns[-30:].std()
            historical_vol = returns[:-30].std()
            
            if recent_vol > historical_vol * 1.2:
                return 'increasing'
            elif recent_vol < historical_vol * 0.8:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'unknown'

    def _analyze_price_trends(self, prices: pd.Series) -> Dict:
        """Analyze price trends"""
        try:
            returns = prices.pct_change()
            
            return {
                'trend': self._determine_trend(prices),
                'momentum': self._calculate_momentum(returns),
                'support_resistance': self._calculate_support_resistance(prices)
            }
        except Exception:
            return {}

    def _determine_trend(self, prices: pd.Series) -> str:
        """Determine the overall price trend"""
        try:
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            current_price = prices.iloc[-1]
            
            if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                return 'uptrend'
            elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                return 'downtrend'
            else:
                return 'sideways'
        except Exception:
            return 'unknown'

    def _calculate_momentum(self, returns: pd.Series) -> Dict:
        """Calculate momentum indicators"""
        try:
            return {
                'daily': returns.iloc[-1],
                'weekly': returns[-5:].mean(),
                'monthly': returns[-20:].mean()
            }
        except Exception:
            return {}

    def _calculate_support_resistance(self, prices: pd.Series) -> Dict:
        """Calculate support and resistance levels"""
        try:
            recent_high = prices[-20:].max()
            recent_low = prices[-20:].min()
            
            return {
                'support': recent_low,
                'resistance': recent_high,
                'midpoint': (recent_high + recent_low) / 2
            }
        except Exception:
            return {}

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs.iloc[-1]))
        except Exception:
            return None

    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate compound annual growth rate"""
        try:
            first_value = series.iloc[-1]  # Earlier period
            last_value = series.iloc[0]    # Later period
            n_periods = len(series) - 1
            
            if first_value > 0 and last_value > 0:
                return (last_value / first_value) ** (1/n_periods) - 1
            return 0.02  # Default growth rate if calculation isn't possible
        except Exception:
            return 0.02  # Default growth rate

    def _calculate_roa(self) -> Optional[float]:
        """Calculate Return on Assets"""
        try:
            net_income = self.financials.loc['Net Income'].iloc[0]
            total_assets = self.balance_sheet.loc['Total Assets'].iloc[0]
            return (net_income / total_assets) * 100
        except Exception:
            return None

    def _calculate_roe(self) -> Optional[float]:
        """Calculate Return on Equity"""
        try:
            net_income = self.financials.loc['Net Income'].iloc[0]
            total_equity = self.balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            return (net_income / total_equity) * 100
        except Exception:
            return None

    def get_dcf_valuation(self) -> Dict:
        """Perform Discounted Cash Flow valuation"""
        try:
            fcf = self.cash_flow.loc['Free Cash Flow']
            growth_rate = self._calculate_growth_rate(fcf)
            wacc = 0.10  # Assumed cost of capital
            
            projected_fcf = []
            for i in range(5):
                projected_fcf.append(fcf.iloc[0] * (1 + growth_rate) ** i)
            
            terminal_value = (projected_fcf[-1] * (1 + 0.02)) / (wacc - 0.02)
            dcf_value = sum([fcf / (1 + wacc) ** (i+1) for i, fcf in enumerate(projected_fcf)])
            dcf_value += terminal_value / (1 + wacc) ** 5
            
            return {
                'dcf_value': dcf_value,
                'per_share_value': dcf_value / self.info.get('sharesOutstanding', 1),
                'assumptions': {
                    'growth_rate': growth_rate,
                    'wacc': wacc,
                    'terminal_growth': 0.02
                }
            }
        except Exception as e:
            return {'error': f"Error calculating DCF valuation: {str(e)}"}
def get_competitor_analysis(self) -> Dict:
    """Compare company metrics with industry peers"""
    try:
        # Get peers from same industry
        industry = self.info.get('industry')
        peers = self._get_peer_companies()
        
        peer_metrics = {}
        for peer in peers[:5]:  # Analyze top 5 peers
            try:
                peer_stock = yf.Ticker(peer)
                peer_metrics[peer] = {
                    'name': peer_stock.info.get('longName'),
                    'market_cap': peer_stock.info.get('marketCap'),
                    'pe_ratio': peer_stock.info.get('trailingPE'),
                    'profit_margin': peer_stock.info.get('profitMargins'),
                    'revenue_growth': peer_stock.info.get('revenueGrowth'),
                    'current_price': peer_stock.info.get('currentPrice')
                }
            except Exception:
                continue
        
        return {
            'peer_comparison': peer_metrics,
            'industry_averages': self._calculate_industry_averages(peer_metrics),
            'relative_position': self._calculate_relative_position(peer_metrics)
        }
    except Exception as e:
        return {'error': f"Error performing competitor analysis: {str(e)}"}

def _get_peer_companies(self) -> List[str]:
    """Get list of peer companies in same industry"""
    try:
        # Basic peer mapping - can be expanded
        industry_peers = {
            'Consumer Electronics': ['AAPL', 'SONY', 'SAMSUNG.KS', 'HPQ', 'DELL'],
            'Software': ['MSFT', 'ORCL', 'CRM', 'ADBE', 'INTU'],
            'Internet Content & Information': ['GOOGL', 'META', 'BIDU', 'TWTR', 'SNAP'],
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM'],
            'Automotive': ['TSLA', 'TM', 'F', 'GM', 'VWAGY'],
            'E-Commerce': ['AMZN', 'BABA', 'JD', 'MELI', 'CPNG']
        }
        
        industry = self.info.get('industry')
        return industry_peers.get(industry, [])
    except Exception:
        return []

def _calculate_industry_averages(self, peer_metrics: Dict) -> Dict:
    """Calculate industry average metrics"""
    try:
        metrics = {
            'avg_market_cap': [],
            'avg_pe_ratio': [],
            'avg_profit_margin': [],
            'avg_revenue_growth': []
        }
        
        for peer, data in peer_metrics.items():
            if data.get('market_cap'):
                metrics['avg_market_cap'].append(data['market_cap'])
            if data.get('pe_ratio'):
                metrics['avg_pe_ratio'].append(data['pe_ratio'])
            if data.get('profit_margin'):
                metrics['avg_profit_margin'].append(data['profit_margin'])
            if data.get('revenue_growth'):
                metrics['avg_revenue_growth'].append(data['revenue_growth'])
        
        return {
            'market_cap': np.mean(metrics['avg_market_cap']) if metrics['avg_market_cap'] else None,
            'pe_ratio': np.mean(metrics['avg_pe_ratio']) if metrics['avg_pe_ratio'] else None,
            'profit_margin': np.mean(metrics['avg_profit_margin']) if metrics['avg_profit_margin'] else None,
            'revenue_growth': np.mean(metrics['avg_revenue_growth']) if metrics['avg_revenue_growth'] else None
        }
    except Exception:
        return {}

def _calculate_relative_position(self, peer_metrics: Dict) -> Dict:
    """Calculate company's relative position in the industry"""
    try:
        company_metrics = {
            'market_cap': self.info.get('marketCap'),
            'pe_ratio': self.info.get('trailingPE'),
            'profit_margin': self.info.get('profitMargins'),
            'revenue_growth': self.info.get('revenueGrowth')
        }
        
        industry_avg = self._calculate_industry_averages(peer_metrics)
        
        return {
            'market_cap_vs_avg': (company_metrics['market_cap'] / industry_avg['market_cap'] - 1) * 100 if industry_avg['market_cap'] else None,
            'pe_ratio_vs_avg': (company_metrics['pe_ratio'] / industry_avg['pe_ratio'] - 1) * 100 if industry_avg['pe_ratio'] else None,
            'profit_margin_vs_avg': (company_metrics['profit_margin'] / industry_avg['profit_margin'] - 1) * 100 if industry_avg['profit_margin'] else None,
            'revenue_growth_vs_avg': (company_metrics['revenue_growth'] / industry_avg['revenue_growth'] - 1) * 100 if industry_avg['revenue_growth'] else None
        }
    except Exception:
        return {}
    
def generate_analysis_report(ticker: str) -> Dict:
    """Generate a comprehensive analysis report"""
    try:
        analyzer = CompanyAnalyzer(ticker)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker.upper(),
            'status': 'success',
            'data': {
                'basic_info': analyzer.get_basic_info(),
                'financial_analysis': analyzer.calculate_financial_ratios(),
                'market_performance': analyzer.analyze_market_performance(),
                'valuation': analyzer.get_dcf_valuation(),
                'competitor_analysis': analyzer.get_competitor_analysis()  # New addition
            }
        }
        
        # Add data quality metrics
        report['data_quality'] = {
            'completeness': _calculate_data_completeness(report['data']),
            'last_updated': datetime.now().isoformat()
        }
        
        return report
    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker.upper(),
            'status': 'error',
            'error': str(e)
        }
    
def _calculate_data_completeness(data: Dict) -> float:
    """Calculate the completeness of the analysis data"""
    total_fields = 0
    filled_fields = 0
    
    def count_fields(d):
        nonlocal total_fields, filled_fields
        for v in d.values():
            if isinstance(v, dict):
                count_fields(v)
            else:
                total_fields += 1
                if v is not None and not (isinstance(v, str) and 'error' in v.lower()):
                    filled_fields += 1
    
    count_fields(data)
    return round(filled_fields / total_fields * 100, 2) if total_fields > 0 else 0

from flask import Flask, jsonify, request
from analyzer import generate_analysis_report  # Direct import
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Ensure CORS for all routes

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        'name': 'Investment Analysis API',
        'version': '2.0',
        'endpoints': {
            '/api/analyze': {
                'method': 'GET',
                'params': {'ticker': 'Stock ticker symbol (e.g., AAPL)'},
                'description': 'Get comprehensive stock analysis'
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Check if the API is running'
            }
        },
        'example': '/api/analyze?ticker=AAPL',
        'timestamp': datetime.now().isoformat()
    })
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['GET'])
def analyze():
    ticker = request.args.get('ticker', default='', type=str)
    if not ticker:
        return jsonify({'error': 'Ticker parameter is missing'}), 400

    try:
        report = generate_analysis_report(ticker)
        return jsonify(report)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
