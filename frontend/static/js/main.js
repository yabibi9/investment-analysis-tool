async function getCompanyData() {
    const ticker = document.getElementById('tickerInput').value.toUpperCase();
    if (!ticker) {
        alert('Please enter a ticker symbol');
        return;
    }

    try {
        const response = await fetch(`https://horrible-bat-694wwvw464g72r6wj-5000.app.github.dev/analyze?ticker=${ticker}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        if (data.status === 'error' || !data.data) {
            throw new Error(data.error || 'Unknown error');
        }

        displayCompanyData(data);
        document.getElementById('errorMessage').classList.add('hidden');
    } catch (error) {
        document.getElementById('companyInfo').classList.add('hidden');
        document.getElementById('errorMessage').classList.remove('hidden');
        console.error('There has been a problem with your fetch operation:', error);
    }
}

function displayCompanyData(data) {
    const company = data.data?.basic_info?.company_info || {};

    document.getElementById('companyName').textContent = company.name || 'Unknown';

    // Basic Info Section
    const basicInfoHtml = `
        <p><strong>Country:</strong> ${company.country || 'N/A'}</p>
        <p><strong>Industry:</strong> ${company.industry || 'N/A'}</p>
        <p><strong>Sector:</strong> ${company.sector || 'N/A'}</p>
        <p><strong>Employees:</strong> ${company.employees !== undefined ? company.employees.toLocaleString() : 'N/A'}</p>
        <p><strong>Website:</strong> <a href="${company.website || '#'}" target="_blank">${company.website || 'N/A'}</a></p>
    `;
    document.getElementById('basicInfo').innerHTML = basicInfoHtml;

    // Key Stats Section
    const keyStats = data.data?.basic_info?.key_stats || {};
    const keyStatsHtml = `
        <p><strong>Beta:</strong> ${keyStats.beta?.toFixed(2) || 'N/A'}</p>
        <p><strong>PE Ratio:</strong> ${keyStats.pe_ratio?.toFixed(2) || 'N/A'}</p>
        <p><strong>Forward PE:</strong> ${keyStats.forward_pe?.toFixed(2) || 'N/A'}</p>
        <p><strong>PEG Ratio:</strong> ${keyStats.peg_ratio?.toFixed(2) || 'N/A'}</p>
        <p><strong>Dividend Yield:</strong> ${keyStats.dividend_yield ? (keyStats.dividend_yield * 100).toFixed(2) + '%' : 'N/A'}</p>
    `;
    document.getElementById('keyStats').innerHTML = keyStatsHtml;

    // Market Data Section
    const marketData = data.data?.basic_info?.market_data || {};
    const marketDataHtml = `
        <p><strong>Current Price:</strong> $${marketData.current_price?.toFixed(2) || 'N/A'}</p>
        <p><strong>52-Week High:</strong> $${marketData.fifty_two_week_high?.toFixed(2) || 'N/A'}</p>
        <p><strong>52-Week Low:</strong> $${marketData.fifty_two_week_low?.toFixed(2) || 'N/A'}</p>
        <p><strong>Market Cap:</strong> $${marketData.market_cap ? (marketData.market_cap / 1e9).toFixed(2) + 'B' : 'N/A'}</p>
        <p><strong>Volume:</strong> ${marketData.volume?.toLocaleString() || 'N/A'}</p>
    `;
    document.getElementById('marketData').innerHTML = marketDataHtml;

    // Growth Analysis Section
    const growth = data.data?.financial_analysis?.growth || {};
    const growthHtml = `
        <p><strong>Revenue Growth (YoY):</strong> ${growth.revenue_growth?.toFixed(2)}%</p>
        <p><strong>Earnings Growth (YoY):</strong> ${growth.earnings_growth?.toFixed(2)}%</p>
        <p><strong>Quarterly Revenue Growth:</strong> ${growth.quarterly_revenue_growth?.toFixed(2)}%</p>
        <p><strong>Quarterly Earnings Growth:</strong> ${growth.quarterly_earnings_growth?.toFixed(2)}%</p>
    `;
    document.getElementById('growthAnalysis').innerHTML = growthHtml;

    // Financial Ratios Section
    const financialRatios = data.data?.financial_analysis || {};
    const profitability = financialRatios.profitability || {};
    const liquidity = financialRatios.liquidity || {};
    const solvency = financialRatios.solvency || {};
    const financialRatiosHtml = `
        <h3>Financial Ratios</h3>
         <h4>Profitability</h4>
        <p><strong>Gross Margin:</strong> ${profitability.gross_margin !== undefined ? profitability.gross_margin + '%' : 'N/A'}</p>
        <p><strong>Net Margin:</strong> ${profitability.net_margin !== undefined ? profitability.net_margin + '%' : 'N/A'}</p>
        <p><strong>ROA:</strong> ${profitability.return_on_assets !== undefined ? profitability.return_on_assets + '%' : 'N/A'}</p>
        <p><strong>ROE:</strong> ${profitability.return_on_equity !== undefined ? profitability.return_on_equity + '%' : 'N/A'}</p>

        <h4>Liquidity</h4>
        <p><strong>Current Ratio:</strong> ${liquidity.current_ratio || 'N/A'}</p>
        <p><strong>Quick Ratio:</strong> ${liquidity.quick_ratio || 'N/A'}</p>
        <p><strong>Cash Ratio:</strong> ${liquidity.cash_ratio || 'N/A'}</p>

        <h4>Solvency</h4>
        <p><strong>Debt to Equity:</strong> ${solvency.debt_to_equity !== undefined ? solvency.debt_to_equity : 'N/A'}</p>
        <p><strong>Debt to Assets:</strong> ${solvency.debt_to_assets !== undefined ? solvency.debt_to_assets : 'N/A'}</p>
    `;
    document.getElementById('financialRatios').innerHTML = financialRatiosHtml;

    // Market Performance Section
    const marketPerformance = data.data?.market_performance || {};
    const technicalIndicators = marketPerformance.technical_indicators || {};
    const trendAnalysis = marketPerformance.trend_analysis || {};
    const volatilityMetrics = marketPerformance.volatility_metrics || {};
    const marketPerformanceHtml = `
        <h4>Technical Indicators</h4>
        <p><strong>50-Day SMA:</strong> $${technicalIndicators.sma_50?.toFixed(2) || 'N/A'}</p>
        <p><strong>200-Day SMA:</strong> $${technicalIndicators.sma_200?.toFixed(2) || 'N/A'}</p>
        <p><strong>RSI:</strong> ${technicalIndicators.rsi?.toFixed(2) || 'N/A'}</p>
        
        <h4>Trend Analysis</h4>
        <p><strong>Current Trend:</strong> ${trendAnalysis.trend || 'N/A'}</p>
        <p><strong>Support Level:</strong> $${trendAnalysis.support_resistance?.support?.toFixed(2) || 'N/A'}</p>
        <p><strong>Resistance Level:</strong> $${trendAnalysis.support_resistance?.resistance?.toFixed(2) || 'N/A'}</p>
        
        <h4>Volatility</h4>
        <p><strong>Annual Volatility:</strong> ${(volatilityMetrics.annual_volatility * 100)?.toFixed(2)}%</p>
        <p><strong>Volatility Trend:</strong> ${volatilityMetrics.volatility_trend || 'N/A'}</p>
    `;
    document.getElementById('marketPerformance').innerHTML = marketPerformanceHtml;

    // Valuation Section
    const valuation = data.data?.valuation || {};
    const valuationHtml = `
        <p><strong>DCF Value:</strong> $${(valuation.dcf_value / 1e9)?.toFixed(2)}B</p>
        <p><strong>Value per Share:</strong> $${valuation.per_share_value?.toFixed(2)}</p>
        <h4>Assumptions</h4>
        <p><strong>Growth Rate:</strong> ${(valuation.assumptions?.growth_rate * 100)?.toFixed(2)}%</p>
        <p><strong>WACC:</strong> ${(valuation.assumptions?.wacc * 100)?.toFixed(2)}%</p>
    `;
    document.getElementById('valuation').innerHTML = valuationHtml;

    document.getElementById('companyInfo').classList.remove('hidden');
}
