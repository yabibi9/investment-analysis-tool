
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Any, Tuple
import requests
from bs4 import BeautifulSoup
import logging
import os
import random
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialScraper:
    """Class for scraping financial data from various sources"""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0"
    ]

    @staticmethod
    def get_headers() -> Dict:
        return {
            'User-Agent': random.choice(FinancialScraper.USER_AGENTS),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }

    @staticmethod
    def scrape_sec_edgar(ticker: str) -> dict:
        """Scrape financial data from SEC EDGAR"""
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&type=10-K"
        headers = FinancialScraper.get_headers()
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to access SEC EDGAR for {ticker}")
                return {}

            soup = BeautifulSoup(response.text, 'html.parser')
            filing_link = soup.find('a', string='10-K')
            if filing_link:
                document_url = f"https://www.sec.gov{filing_link['href']}"
                return FinancialScraper.scrape_10k_document(document_url)
            return {}
        except Exception as e:
            logger.error(f"Error accessing SEC EDGAR: {str(e)}")
            return {}

    @staticmethod
    def scrape_10k_document(url: str) -> dict:
        """Scrape specific details from a 10-K document"""
        headers = FinancialScraper.get_headers()
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                data = {}
                # Extract key financial data (can be expanded based on needs)
                return data
            return {}
        except Exception as e:
            logger.error(f"Error scraping 10-K: {str(e)}")
            return {}

class CompanyAnalyzer:
    """A comprehensive company financial analyzer"""

    INDUSTRY_WACC = {
        'Technology': 0.09,
        'Healthcare': 0.08,
        'Financial Services': 0.07,
        'Consumer Goods': 0.10,
        'Utilities': 0.06,
        'Real Estate': 0.08,
        'Energy': 0.11,
        'Materials': 0.09,
        'Industrials': 0.10,
        'Communication Services': 0.08
    }

    def __init__(self, ticker: str) -> None:
        """Initialize analyzer with company ticker"""
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = {}
        self.financials = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.cash_flow = pd.DataFrame()
        self.quarterly_financials = pd.DataFrame()
        self.history = pd.DataFrame()
        self._load_data()

    def _load_data(self) -> None:
        """Load data from multiple sources with fallback options"""
        try:
            self._load_data_from_yahoo_finance()
            
            missing_fields = self._check_missing_fields()
            if missing_fields:
                logger.warning(f"Missing fields for {self.ticker}: {missing_fields}")
                
                # Try Alpha Vantage
                alpha_data = self._get_data_from_alpha_vantage()
                if alpha_data:
                    self.info.update(alpha_data)
                
                # If still missing data, try SEC EDGAR
                if self._check_missing_fields():
                    sec_data = FinancialScraper.scrape_sec_edgar(self.ticker)
                    if sec_data:
                        self.info.update(sec_data)
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Failed to load data for {self.ticker}")

    def _load_data_from_yahoo_finance(self) -> None:
        """Load data from Yahoo Finance"""
        try:
            self.info = self.stock.info
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cash_flow = self.stock.cashflow
            self.quarterly_financials = self.stock.quarterly_financials
            self.history = self.stock.history(period='2y')
        except Exception as e:
            logger.error(f"Yahoo Finance data load failed: {str(e)}")

    def _get_data_from_alpha_vantage(self) -> dict:
        """Fetch data from Alpha Vantage API"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logger.warning("Alpha Vantage API key not found")
            return {}

        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={self.ticker}&apikey={api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'longName': data.get('Name'),
                    'sector': data.get('Sector'),
                    'industry': data.get('Industry'),
                    'marketCap': data.get('MarketCapitalization'),
                    'trailingPE': data.get('TrailingPE'),
                    'forwardPE': data.get('ForwardPE'),
                }
            return {}
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {str(e)}")
            return {}

    def _check_missing_fields(self) -> List[str]:
        """Check for missing essential fields"""
        required_fields = [
            'sector', 'longName', 'marketCap', 'currentPrice',
            'trailingPE', 'forwardPE'
        ]
        return [field for field in required_fields if not self.info.get(field)]

    def _get_industry_wacc(self) -> float:
        """Get industry-specific WACC or default to 10%"""
        sector = self.info.get('sector')
        return self.INDUSTRY_WACC.get(sector, 0.10)

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
            logger.error(f"Error calculating ratios: {str(e)}")
            return {'error': str(e)}

    def _calculate_profitability_ratios(self) -> Dict:
        """Calculate profitability ratios"""
        try:
            # Get data from the most recent period
            if 'Total Revenue' not in self.financials.index:
                return self._default_ratios()

            revenue = float(self.financials.loc['Total Revenue'].iloc[0])
            gross_profit = float(self.financials.loc['Gross Profit'].iloc[0]) if 'Gross Profit' in self.financials.index else None
            operating_income = float(self.financials.loc['Operating Income'].iloc[0]) if 'Operating Income' in self.financials.index else None
            net_income = float(self.financials.loc['Net Income'].iloc[0]) if 'Net Income' in self.financials.index else None
            
            ratios = {}
            
            # Calculate ratios only if we have valid data
            if revenue and revenue > 0:
                if gross_profit is not None:
                    ratios['gross_margin'] = round((gross_profit / revenue) * 100, 2)
                if operating_income is not None:
                    ratios['operating_margin'] = round((operating_income / revenue) * 100, 2)
                if net_income is not None:
                    ratios['net_margin'] = round((net_income / revenue) * 100, 2)
            
            # Calculate ROA and ROE
            roa = self._calculate_roa()
            if roa is not None:
                ratios['return_on_assets'] = round(roa, 2)
                
            roe = self._calculate_roe()
            if roe is not None:
                ratios['return_on_equity'] = round(roe, 2)
            
            # Fill in any missing ratios with 'N/A'
            for key in ['gross_margin', 'operating_margin', 'net_margin', 'return_on_assets', 'return_on_equity']:
                if key not in ratios:
                    ratios[key] = 'N/A'
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating profitability ratios: {str(e)}")
            return self._default_ratios()

    def _default_ratios(self) -> Dict:
        """Return default 'N/A' ratios"""
        return {
            'gross_margin': 'N/A',
            'operating_margin': 'N/A',
            'net_margin': 'N/A',
            'return_on_equity': 'N/A',
            'return_on_assets': 'N/A'
        }

    def _calculate_liquidity_ratios(self) -> Dict:
        """Calculate liquidity ratios"""
        try:
            # Check if required data exists
            if 'Total Current Assets' not in self.balance_sheet.index or \
               'Total Current Liabilities' not in self.balance_sheet.index:
                return {'current_ratio': 'N/A', 'quick_ratio': 'N/A', 'cash_ratio': 'N/A'}

            # Get the data
            current_assets = float(self.balance_sheet.loc['Total Current Assets'].iloc[0])
            current_liabilities = float(self.balance_sheet.loc['Total Current Liabilities'].iloc[0])
            inventory = float(self.balance_sheet.loc['Inventory'].iloc[0]) if 'Inventory' in self.balance_sheet.index else 0
            cash = float(self.balance_sheet.loc['Cash'].iloc[0]) if 'Cash' in self.balance_sheet.index else 0

            # Calculate ratios only if denominators are valid
            ratios = {}
            if current_liabilities and current_liabilities > 0:
                ratios['current_ratio'] = round(current_assets / current_liabilities, 2)
                ratios['quick_ratio'] = round((current_assets - inventory) / current_liabilities, 2)
                ratios['cash_ratio'] = round(cash / current_liabilities, 2)
            else:
                return {'current_ratio': 'N/A', 'quick_ratio': 'N/A', 'cash_ratio': 'N/A'}

            return ratios

        except Exception as e:
            logger.error(f"Error calculating liquidity ratios: {str(e)}")
            return {'current_ratio': 'N/A', 'quick_ratio': 'N/A', 'cash_ratio': 'N/A'}

    def _calculate_solvency_ratios(self) -> Dict:
        """Calculate solvency ratios"""
        try:
            total_assets = float(self.balance_sheet.loc['Total Assets'].iloc[0])
            total_liabilities = float(self.balance_sheet.loc['Total Liabilities'].iloc[0])
            total_equity = float(self.balance_sheet.loc['Total Stockholder Equity'].iloc[0])
            
            if total_equity == 0 or total_assets == 0:
                return {'debt_to_equity': 'N/A', 'debt_to_assets': 'N/A', 'equity_multiplier': 'N/A'}

            return {
                'debt_to_equity': total_liabilities / total_equity,
                'debt_to_assets': total_liabilities / total_assets,
                'equity_multiplier': total_assets / total_equity
            }
        except Exception:
            return {'debt_to_equity': 'N/A', 'debt_to_assets': 'N/A', 'equity_multiplier': 'N/A'}

    def _calculate_efficiency_ratios(self) -> Dict:
        """Calculate efficiency ratios"""
        try:
            revenue = float(self.financials.loc['Total Revenue'].iloc[0])
            total_assets = float(self.balance_sheet.loc['Total Assets'].iloc[0])
            receivables = float(self.balance_sheet.loc.get('Net Receivables', pd.Series([0])).iloc[0])
            inventory = float(self.balance_sheet.loc.get('Inventory', pd.Series([0])).iloc[0])
            
            ratios = {'asset_turnover': revenue / total_assets if total_assets else 'N/A'}
            
            if receivables:
                ratios['receivables_turnover'] = revenue / receivables
            if inventory:
                ratios['inventory_turnover'] = float(self.financials.loc['Cost Of Revenue'].iloc[0]) / inventory
            
            return ratios
        except Exception:
            return {'asset_turnover': 'N/A', 'receivables_turnover': 'N/A', 'inventory_turnover': 'N/A'}

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
            return {
                'revenue_growth': 'N/A',
                'earnings_growth': 'N/A',
                'quarterly_revenue_growth': 'N/A',
                'quarterly_earnings_growth': 'N/A'
            }

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
            logger.error(f"Market performance analysis error: {str(e)}")
            return {'error': str(e)}


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
                'rsi': self._calculate_rsi(prices),
                'macd': self._calculate_macd(prices)
            }
        except Exception as e:
            logger.error(f"Technical indicators error: {str(e)}")
            return {}

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'macd_line': macd.iloc[-1],
                'signal_line': signal.iloc[-1],
                'histogram': macd.iloc[-1] - signal.iloc[-1]
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
            recent_prices = prices[-20:]
            recent_high = recent_prices.max()
            recent_low = recent_prices.min()
            
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

    def get_dcf_valuation(self) -> Dict:
        """Perform Discounted Cash Flow valuation"""
        try:
            fcf = self.cash_flow.loc['Free Cash Flow']
            growth_rate = self._calculate_growth_rate(fcf)
            wacc = self._get_industry_wacc()
            
            projected_fcf = [fcf.iloc[0] * (1 + growth_rate) ** i for i in range(5)]
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
            logger.error(f"DCF valuation error: {str(e)}")
            return {'error': str(e)}

    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate compound annual growth rate"""
        try:
            first_value = series.iloc[-1]  # Earlier period
            last_value = series.iloc[0]    # Later period
            n_periods = len(series) - 1
            
            if first_value > 0 and last_value > 0:
                return (last_value / first_value) ** (1/n_periods) - 1
            return 0.02  # Default growth rate
        except Exception:
            return 0.02

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
                'valuation': analyzer.get_dcf_valuation()
            }
        }
        
        report['data_quality'] = {
            'completeness': _calculate_data_completeness(report['data']),
            'last_updated': datetime.now().isoformat()
        }
        
        return report
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
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
