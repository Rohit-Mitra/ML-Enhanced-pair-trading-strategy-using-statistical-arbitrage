# ============================================
# NSE PAIRS TRADING - B.TECH MINOR PROJECT
# IMPROVED VERSION - More Pairs Selection
# Author: Siddharth Kothari
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.api import OLS, add_constant
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("NSE PAIRS TRADING SIMULATION - IMPROVED VERSION")
print("="*70)

# -----------------------------
# [1] PARAMETERS
# -----------------------------
print("\n[1] Setting Parameters...")

# IT SECTOR STOCKS (Expanded)
it_stocks = [
    'TCS.NS',        
    'INFY.NS',       
    'WIPRO.NS',      
    'HCLTECH.NS',    
    'TECHM.NS',      
    'LTIM.NS',       
    'MPHASIS.NS',    
    'COFORGE.NS',    
    'PERSISTENT.NS', 
    'OFSS.NS',
    'TATAELXSI.NS',  # Added
    'LTTS.NS',       # Added - L&T Technology Services
    'CYIENT.NS',     # Added
    'ROUTE.NS',      # Added
    'HAPPSTMNDS.NS'  # Added - Happiest Minds
]

# BANKING SECTOR STOCKS (Expanded with different categories)
banking_stocks = [
    # Private Banks
    'HDFCBANK.NS',   
    'ICICIBANK.NS',  
    'KOTAKBANK.NS',  
    'AXISBANK.NS',   
    'INDUSINDBK.NS', 
    'FEDERALBNK.NS', 
    'IDFCFIRSTB.NS',
    'BANDHANBNK.NS', # Added
    'RBLBANK.NS',    # Added
    'YESBANK.NS',    # Added
    # PSU Banks
    'SBIN.NS',       
    'BANKBARODA.NS', 
    'PNB.NS',
    'CANBK.NS',      # Added - Canara Bank
    'UNIONBANK.NS',  # Added
    # NBFCs (similar behavior)
    'BAJFINANCE.NS', # Added
    'BAJAJFINSV.NS', # Added
    'CHOLAFIN.NS',   # Added
    'MUTHOOTFIN.NS', # Added
    'SHRIRAMFIN.NS'  # Added
]

# HEALTHCARE / PHARMA SECTOR STOCKS (Expanded)
healthcare_stocks = [
    # Large Cap Pharma
    'SUNPHARMA.NS',  
    'DRREDDY.NS',    
    'CIPLA.NS',      
    'DIVISLAB.NS',   
    'LUPIN.NS',      
    'AUROPHARMA.NS', 
    'TORNTPHARM.NS',
    'ALKEM.NS',      # Added
    'GLENMARK.NS',   # Added
    'ZYDUSLIFE.NS',  # Added (Zydus Lifesciences)
    'IPCALAB.NS',    # Added
    'NATCOPHARMA.NS',# Added
    'LAURUSLABS.NS', # Added
    'GRANULES.NS',   # Added
    # Hospitals
    'APOLLOHOSP.NS', 
    'FORTIS.NS',
    'MAXHEALTH.NS',  # Added - Max Healthcare
    'MEDANTA.NS',    # Added
    # Diagnostics
    'METROPOLIS.NS', # Added
    'LALPATHLAB.NS', # Added
    # Biotech
    'BIOCON.NS',
    'SYNGENE.NS'     # Added
]

all_sectors = {
    'IT': it_stocks,
    'BANKING': banking_stocks,
    'HEALTHCARE': healthcare_stocks
}

TRAIN_START = '2016-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-01'
TRANSACTION_COST = 0.001  

# TRADING PARAMETERS
ENTRY_ZSCORE = 1.5          
EXIT_ZSCORE = 0.3           
STOP_LOSS_ZSCORE = 3.0      
MAX_HOLDING_DAYS = 30       

# PAIR SELECTION PARAMETERS - TIERED APPROACH
# Tier 1: Strict (Best pairs)
TIER1_PVALUE = 0.03
TIER1_MIN_HALF_LIFE = 5
TIER1_MAX_HALF_LIFE = 40
TIER1_MAX_HURST = 0.45
TIER1_MIN_CORRELATION = 0.75

# Tier 2: Moderate
TIER2_PVALUE = 0.05
TIER2_MIN_HALF_LIFE = 3
TIER2_MAX_HALF_LIFE = 60
TIER2_MAX_HURST = 0.50
TIER2_MIN_CORRELATION = 0.65

# Tier 3: Relaxed (More pairs, slightly lower quality)
TIER3_PVALUE = 0.10
TIER3_MIN_HALF_LIFE = 2
TIER3_MAX_HALF_LIFE = 90
TIER3_MAX_HURST = 0.55
TIER3_MIN_CORRELATION = 0.50

# Minimum pairs per sector target
MIN_PAIRS_PER_SECTOR = 3

# Optimization Grid
WINDOW_OPTIONS = [30, 45, 60, 90]

print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
print(f"Testing Period: {TEST_START} to {TEST_END}")
print(f"Transaction Cost: {TRANSACTION_COST*100}%")
print(f"Entry Z-Score: +/- {ENTRY_ZSCORE}")
print(f"Min Pairs Per Sector Target: {MIN_PAIRS_PER_SECTOR}")
print(f"\nSectors Included:")
for sector, stocks in all_sectors.items():
    print(f"  - {sector}: {len(stocks)} stocks")

# -----------------------------
# [2] DOWNLOAD DATA
# -----------------------------
print("\n[2] Downloading Data...")

sector_data = {}
sector_valid_tickers = {}

for sector_name, stock_list in all_sectors.items():
    print(f"\n--- {sector_name} SECTOR ---")
    
    valid_tickers = []
    price_data = pd.DataFrame()
    
    for ticker in stock_list:
        try:
            print(f"  Downloading {ticker}...", end=" ")
            df = yf.download(ticker, start=TRAIN_START, end=TEST_END, progress=False)
            
            if df.empty:
                print("[X] (no data)")
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if 'Adj Close' in df.columns:
                price_data[ticker] = df['Adj Close']
            elif 'Close' in df.columns:
                price_data[ticker] = df['Close']
            else:
                print("[X] (no price column)")
                continue
            
            # Check for sufficient data
            if len(df) < 500:  # Need at least ~2 years of data
                print(f"[X] (insufficient data: {len(df)} days)")
                continue
                
            valid_tickers.append(ticker)
            print(f"[OK] ({len(df)} days)")
            
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
            print(f"[X] ({error_msg[:30]})")
            continue
    
    price_data.dropna(how='all', inplace=True)
    price_data = price_data.ffill().bfill()
    
    sector_data[sector_name] = price_data
    sector_valid_tickers[sector_name] = valid_tickers
    
    print(f"  [OK] {sector_name}: {len(valid_tickers)}/{len(stock_list)} stocks downloaded")

print("\n" + "-"*50)
print("DOWNLOAD SUMMARY:")
print("-"*50)
total_stocks = 0
for sector_name in all_sectors.keys():
    count = len(sector_valid_tickers[sector_name])
    total_stocks += count
    print(f"  {sector_name}: {count} stocks")
print(f"  TOTAL: {total_stocks} stocks")

# -----------------------------
# [3] HELPER FUNCTIONS
# -----------------------------

def calculate_half_life(spread):
    """Calculate half-life of mean reversion."""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]
    
    try:
        spread_lag_const = add_constant(spread_lag)
        model = OLS(spread_diff, spread_lag_const).fit()
        lambda_param = model.params.iloc[1]
        
        if lambda_param < 0:
            half_life = -np.log(2) / lambda_param
            return half_life
        else:
            return 999
    except:
        return 999

def calculate_hurst_exponent(spread, max_lag=100):
    """Calculate Hurst exponent."""
    lags = range(2, min(max_lag, len(spread)//2))
    tau = []
    
    for lag in lags:
        try:
            tau.append(np.std(np.subtract(spread[lag:].values, spread[:-lag].values)))
        except:
            continue
    
    if len(tau) < 10:
        return 0.5
    
    try:
        reg = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        return reg[0]
    except:
        return 0.5

def check_spread_stationarity(spread):
    """Check if spread is stationary using ADF test."""
    try:
        adf_result = adfuller(spread.dropna(), maxlag=20, autolag='AIC')
        return adf_result[1] < 0.10  # Relaxed from 0.05 to 0.10
    except:
        return False

def calculate_spread_stats(stock1, stock2, price_data, train_start, train_end):
    """Calculate all statistics for a potential pair."""
    try:
        train_prices = price_data.loc[train_start:train_end].dropna()
        
        if len(train_prices) < 200:
            return None
        
        if stock1 not in train_prices.columns or stock2 not in train_prices.columns:
            return None
        
        # Cointegration test
        score, pvalue, _ = coint(train_prices[stock1], train_prices[stock2])
        
        # Calculate hedge ratio
        y = train_prices[stock1]
        x_const = add_constant(train_prices[stock2])
        model = OLS(y, x_const).fit()
        beta = model.params[stock2]
        
        # Calculate spread
        spread = train_prices[stock1] - beta * train_prices[stock2]
        
        # Calculate metrics
        half_life = calculate_half_life(spread)
        hurst = calculate_hurst_exponent(spread)
        is_stationary = check_spread_stationarity(spread)
        correlation = train_prices[stock1].corr(train_prices[stock2])
        
        # Spread volatility (for ranking)
        spread_volatility = spread.std() / spread.mean() if spread.mean() != 0 else 999
        
        return {
            'stock1': stock1,
            'stock2': stock2,
            'pvalue': pvalue,
            'beta': beta,
            'half_life': half_life,
            'hurst': hurst,
            'is_stationary': is_stationary,
            'correlation': correlation,
            'spread_volatility': abs(spread_volatility),
            'coint_score': score
        }
    except:
        return None

def score_pair(stats):
    """
    Score a pair based on multiple criteria.
    Higher score = better pair for trading.
    """
    score = 0
    
    # P-value (lower is better)
    if stats['pvalue'] < 0.01:
        score += 30
    elif stats['pvalue'] < 0.03:
        score += 25
    elif stats['pvalue'] < 0.05:
        score += 20
    elif stats['pvalue'] < 0.10:
        score += 10
    
    # Half-life (5-40 is ideal)
    hl = stats['half_life']
    if 10 <= hl <= 30:
        score += 25
    elif 5 <= hl <= 40:
        score += 20
    elif 3 <= hl <= 60:
        score += 15
    elif 2 <= hl <= 90:
        score += 10
    
    # Hurst exponent (lower is better, < 0.5 is mean reverting)
    hurst = stats['hurst']
    if hurst < 0.4:
        score += 25
    elif hurst < 0.45:
        score += 20
    elif hurst < 0.50:
        score += 15
    elif hurst < 0.55:
        score += 10
    
    # Correlation (higher is better for pairs)
    corr = stats['correlation']
    if corr > 0.85:
        score += 20
    elif corr > 0.75:
        score += 15
    elif corr > 0.65:
        score += 10
    elif corr > 0.50:
        score += 5
    
    # Stationarity bonus
    if stats['is_stationary']:
        score += 10
    
    return score

# -----------------------------
# [4] FIND COINTEGRATED PAIRS (TIERED APPROACH)
# -----------------------------
print("\n[3] Testing for Cointegration (Tiered Approach)...")

all_pairs = []
all_pair_details = []

for sector_name, price_data in sector_data.items():
    print(f"\n{'='*50}")
    print(f"--- {sector_name} SECTOR ---")
    print(f"{'='*50}")
    
    valid_tickers = sector_valid_tickers[sector_name]
    sector_candidates = []
    
    # Calculate stats for all possible pairs
    print(f"  Analyzing {len(valid_tickers) * (len(valid_tickers)-1) // 2} possible pairs...")
    
    for i in range(len(valid_tickers)):
        for j in range(i+1, len(valid_tickers)):
            stock1 = valid_tickers[i]
            stock2 = valid_tickers[j]
            
            stats = calculate_spread_stats(stock1, stock2, price_data, TRAIN_START, TRAIN_END)
            
            if stats is not None:
                stats['sector'] = sector_name
                stats['price_data'] = price_data
                stats['score'] = score_pair(stats)
                sector_candidates.append(stats)
    
    print(f"  Found {len(sector_candidates)} potential pairs")
    
    # Sort by score (descending)
    sector_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Select pairs using tiered approach
    selected_pairs = []
    
    # TIER 1: Strict criteria
    print(f"\n  TIER 1 (Strict):")
    for stats in sector_candidates:
        if len(selected_pairs) >= MIN_PAIRS_PER_SECTOR * 2:  # Get more than minimum
            break
            
        if (stats['pvalue'] < TIER1_PVALUE and
            TIER1_MIN_HALF_LIFE <= stats['half_life'] <= TIER1_MAX_HALF_LIFE and
            stats['hurst'] < TIER1_MAX_HURST and
            stats['correlation'] >= TIER1_MIN_CORRELATION and
            stats['is_stationary']):
            
            selected_pairs.append(stats)
            print(f"    [T1] {stats['stock1']}/{stats['stock2']} "
                  f"(Score:{stats['score']}, p={stats['pvalue']:.3f}, "
                  f"HL={stats['half_life']:.1f}, H={stats['hurst']:.2f})")
    
    tier1_count = len(selected_pairs)
    print(f"    Tier 1 pairs: {tier1_count}")
    
    # TIER 2: Moderate criteria (if needed)
    if len(selected_pairs) < MIN_PAIRS_PER_SECTOR:
        print(f"\n  TIER 2 (Moderate):")
        for stats in sector_candidates:
            if len(selected_pairs) >= MIN_PAIRS_PER_SECTOR * 2:
                break
            
            # Skip if already selected
            if stats in selected_pairs:
                continue
                
            if (stats['pvalue'] < TIER2_PVALUE and
                TIER2_MIN_HALF_LIFE <= stats['half_life'] <= TIER2_MAX_HALF_LIFE and
                stats['hurst'] < TIER2_MAX_HURST and
                stats['correlation'] >= TIER2_MIN_CORRELATION):
                
                selected_pairs.append(stats)
                print(f"    [T2] {stats['stock1']}/{stats['stock2']} "
                      f"(Score:{stats['score']}, p={stats['pvalue']:.3f}, "
                      f"HL={stats['half_life']:.1f}, H={stats['hurst']:.2f})")
        
        tier2_count = len(selected_pairs) - tier1_count
        print(f"    Tier 2 pairs: {tier2_count}")
    
    # TIER 3: Relaxed criteria (if still needed)
    if len(selected_pairs) < MIN_PAIRS_PER_SECTOR:
        print(f"\n  TIER 3 (Relaxed):")
        for stats in sector_candidates:
            if len(selected_pairs) >= MIN_PAIRS_PER_SECTOR * 2:
                break
            
            if stats in selected_pairs:
                continue
                
            if (stats['pvalue'] < TIER3_PVALUE and
                TIER3_MIN_HALF_LIFE <= stats['half_life'] <= TIER3_MAX_HALF_LIFE and
                stats['hurst'] < TIER3_MAX_HURST and
                stats['correlation'] >= TIER3_MIN_CORRELATION):
                
                selected_pairs.append(stats)
                print(f"    [T3] {stats['stock1']}/{stats['stock2']} "
                      f"(Score:{stats['score']}, p={stats['pvalue']:.3f}, "
                      f"HL={stats['half_life']:.1f}, H={stats['hurst']:.2f})")
        
        tier3_count = len(selected_pairs) - tier1_count - tier2_count
        print(f"    Tier 3 pairs: {tier3_count}")
    
    # TIER 4: Top scored pairs regardless of strict criteria (last resort)
    if len(selected_pairs) < MIN_PAIRS_PER_SECTOR:
        print(f"\n  TIER 4 (Top Scored - Last Resort):")
        for stats in sector_candidates:
            if len(selected_pairs) >= MIN_PAIRS_PER_SECTOR:
                break
            
            if stats in selected_pairs:
                continue
            
            # Only basic filter: p-value < 0.15 and reasonable half-life
            if stats['pvalue'] < 0.15 and stats['half_life'] < 120:
                selected_pairs.append(stats)
                print(f"    [T4] {stats['stock1']}/{stats['stock2']} "
                      f"(Score:{stats['score']}, p={stats['pvalue']:.3f}, "
                      f"HL={stats['half_life']:.1f}, H={stats['hurst']:.2f})")
    
    print(f"\n  [OK] {sector_name} total selected: {len(selected_pairs)} pairs")
    
    # Add to global lists
    for stats in selected_pairs:
        pair_info = {
            'pair': (stats['stock1'], stats['stock2']),
            'sector': sector_name,
            'price_data': price_data,
            'beta': stats['beta'],
            'half_life': stats['half_life'],
            'hurst': stats['hurst'],
            'correlation': stats['correlation'],
            'score': stats['score'],
            'pvalue': stats['pvalue']
        }
        all_pairs.append(pair_info)
        
        all_pair_details.append({
            'Sector': sector_name,
            'Stock1': stats['stock1'],
            'Stock2': stats['stock2'],
            'P-Value': stats['pvalue'],
            'Beta': stats['beta'],
            'Half_Life': stats['half_life'],
            'Hurst': stats['hurst'],
            'Correlation': stats['correlation'],
            'Score': stats['score'],
            'Stationary': stats['is_stationary']
        })

# Sort all pairs by score
all_pairs.sort(key=lambda x: x['score'], reverse=True)

print(f"\n{'='*50}")
print(f"PAIR SELECTION SUMMARY")
print(f"{'='*50}")

# Summary by sector
for sector in all_sectors.keys():
    sector_count = len([p for p in all_pairs if p['sector'] == sector])
    print(f"  {sector}: {sector_count} pairs")

print(f"  TOTAL: {len(all_pairs)} pairs")

if len(all_pairs) == 0:
    print("\n[X] No pairs found even with relaxed criteria. Exiting.")
    exit()

# Save pair details
pairs_df = pd.DataFrame(all_pair_details)
pairs_df = pairs_df.sort_values(['Sector', 'Score'], ascending=[True, False])
pairs_df.to_csv('cointegrated_pairs.csv', index=False)
print("\n[OK] Cointegrated pairs saved to: cointegrated_pairs.csv")

# Display top pairs
print("\n[INFO] TOP 10 PAIRS BY SCORE:")
print("-" * 70)
top_pairs_df = pairs_df.head(10)[['Sector', 'Stock1', 'Stock2', 'Score', 'P-Value', 'Half_Life', 'Hurst']]
print(top_pairs_df.to_string(index=False))

# -----------------------------
# [5] IMPROVED ML FEATURE CREATION
# -----------------------------
def create_ml_data_improved(y_price, x_price, beta, window):
    """Creates IMPROVED ML features for pairs trading."""
    
    spread = y_price - beta * x_price 
    
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    df = pd.DataFrame(index=spread.index)
    df['spread'] = spread
    df['zscore'] = (spread - rolling_mean) / rolling_std
    
    # Z-score features
    df['zscore_lag1'] = df['zscore'].shift(1)
    df['zscore_lag2'] = df['zscore'].shift(2)
    df['zscore_lag3'] = df['zscore'].shift(3)
    df['zscore_lag5'] = df['zscore'].shift(5)
    
    # Z-score momentum
    df['zscore_change'] = df['zscore'].diff().shift(1)
    df['zscore_change_5d'] = (df['zscore'] - df['zscore'].shift(5)).shift(1)
    
    # Spread features
    df['spread_diff_lag1'] = spread.diff().shift(1)
    df['spread_diff_lag5'] = spread.diff(5).shift(1)
    
    # Volatility features
    df['spread_volatility'] = spread.rolling(window=20).std().shift(1)
    df['zscore_volatility'] = df['zscore'].rolling(window=20).std().shift(1)
    
    # Moving average crossover
    df['zscore_ma5'] = df['zscore'].rolling(5).mean().shift(1)
    df['zscore_ma10'] = df['zscore'].rolling(10).mean().shift(1)
    df['ma_crossover'] = (df['zscore_ma5'] - df['zscore_ma10']).shift(1)
    
    # Mean reversion signal
    df['reversion_signal'] = np.where(
        df['zscore_lag1'] > 0,
        -df['zscore_change'],
        df['zscore_change']
    )
    
    # RSI-like indicator
    delta = df['zscore'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['zscore_rsi'] = (100 - (100 / (1 + rs))).shift(1)
    
    # Target: Mean reversion success
    future_spread_change = df['spread'].shift(-5) - df['spread']
    current_zscore = df['zscore']
    
    df['target'] = np.where(
        (current_zscore > 0.5) & (future_spread_change < 0), 1,
        np.where(
            (current_zscore < -0.5) & (future_spread_change > 0), 1,
            0
        )
    )
    
    df.dropna(inplace=True)
    return df

# -----------------------------
# [6] IMPROVED SIMULATION FUNCTION
# -----------------------------
def run_simulation_improved(data, model, hedge_ratio_beta, y_stock, x_stock, price_data_sector, scaler=None):
    """IMPROVED trading simulation with risk management."""
    
    feature_cols = ['zscore_lag1', 'zscore_lag2', 'zscore_lag3', 'zscore_lag5',
                    'zscore_change', 'zscore_change_5d', 'spread_diff_lag1', 
                    'spread_diff_lag5', 'spread_volatility', 'zscore_volatility',
                    'zscore_ma5', 'zscore_ma10', 'ma_crossover', 'reversion_signal', 'zscore_rsi']
    
    if scaler is not None:
        features_scaled = scaler.transform(data[feature_cols])
        data['pred_signal'] = model.predict(features_scaled)
        data['pred_proba'] = model.predict_proba(features_scaled)[:, 1]
    else:
        data['pred_signal'] = model.predict(data[feature_cols])
        try:
            data['pred_proba'] = model.predict_proba(data[feature_cols])[:, 1]
        except:
            data['pred_proba'] = 0.5
    
    positions = []
    pnl_list = []
    current_position = 0
    days_in_position = 0
    
    for i in range(len(data)):
        row = data.iloc[i]
        z = row['zscore_lag1'] if not np.isnan(row['zscore_lag1']) else 0
        pred = row['pred_signal']
        pred_proba = row['pred_proba']
        current_spread = row['spread']
        
        if i > 0:
            spread_change = current_spread - data.iloc[i-1]['spread']
            daily_pnl = current_position * spread_change
        else:
            daily_pnl = 0
        
        # EXIT CONDITIONS
        if current_position != 0:
            days_in_position += 1
            
            stop_loss_triggered = False
            if current_position == 1 and z < -STOP_LOSS_ZSCORE:
                stop_loss_triggered = True
            elif current_position == -1 and z > STOP_LOSS_ZSCORE:
                stop_loss_triggered = True
            
            take_profit = False
            if current_position == 1 and z > -EXIT_ZSCORE:
                take_profit = True
            elif current_position == -1 and z < EXIT_ZSCORE:
                take_profit = True
            
            max_hold_reached = days_in_position >= MAX_HOLDING_DAYS
            
            if stop_loss_triggered or take_profit or max_hold_reached:
                current_position = 0
                days_in_position = 0
        
        # ENTRY CONDITIONS
        if current_position == 0:
            if z < -ENTRY_ZSCORE and pred == 1 and pred_proba > 0.52:
                current_position = 1
                days_in_position = 0
            elif z > ENTRY_ZSCORE and pred == 1 and pred_proba > 0.52:
                current_position = -1
                days_in_position = 0
        
        positions.append(current_position)
        pnl_list.append(daily_pnl)
    
    data['position'] = positions
    data['pnl_gross'] = pnl_list
    
    position_series = pd.Series(positions, index=data.index)
    position_changes = position_series.diff().fillna(0).abs()
    
    trade_value = (price_data_sector[y_stock].loc[data.index] + 
                   abs(hedge_ratio_beta) * price_data_sector[x_stock].loc[data.index])
    
    data['txn_cost'] = position_changes * trade_value * TRANSACTION_COST
    data['pnl_net'] = data['pnl_gross'] - data['txn_cost']
    data['cumulative_pnl'] = data['pnl_net'].cumsum()
    
    return data, data['pnl_net'].sum()

# -----------------------------
# [7] MAIN LOOP: TRAIN, OPTIMIZE, TEST
# -----------------------------
print("\n[4] Training ML Models & Running Simulations...")

trades_list = []
summary_list = []

feature_cols = ['zscore_lag1', 'zscore_lag2', 'zscore_lag3', 'zscore_lag5',
                'zscore_change', 'zscore_change_5d', 'spread_diff_lag1', 
                'spread_diff_lag5', 'spread_volatility', 'zscore_volatility',
                'zscore_ma5', 'zscore_ma10', 'ma_crossover', 'reversion_signal', 'zscore_rsi']

for pair_idx, pair_info in enumerate(all_pairs, 1):
    
    y_stock, x_stock = pair_info['pair']
    sector_name = pair_info['sector']
    price_data = pair_info['price_data']
    pre_calc_beta = pair_info['beta']
    half_life = pair_info['half_life']
    pair_score = pair_info['score']
    
    print(f"\n--- Pair {pair_idx}/{len(all_pairs)}: {y_stock}/{x_stock} [{sector_name}] ---")
    print(f"Score: {pair_score}, Beta: {pre_calc_beta:.4f}, Half-life: {half_life:.1f} days")
    
    hedge_ratio_beta = pre_calc_beta
    
    # OPTIMIZATION
    best_score = -float('inf')
    best_window = int(min(max(half_life * 1.5, 30), 60))
    
    print("Optimizing with cross-validation...")
    
    for window in WINDOW_OPTIONS:
        try:
            full_data = create_ml_data_improved(price_data[y_stock], price_data[x_stock], 
                                                hedge_ratio_beta, window)
            train_data_opt = full_data.loc[TRAIN_START:TRAIN_END].copy()
            
            if len(train_data_opt) < 100:
                continue
            
            X = train_data_opt[feature_cols]
            y = train_data_opt['target']
            
            if len(np.unique(y)) < 2:
                continue
            
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_cv)
                X_val_scaled = scaler.transform(X_val_cv)
                
                model_cv = GradientBoostingClassifier(
                    n_estimators=100, 
                    max_depth=4, 
                    learning_rate=0.1,
                    random_state=42
                )
                model_cv.fit(X_train_scaled, y_train_cv)
                
                val_pred = model_cv.predict(X_val_scaled)
                cv_scores.append(accuracy_score(y_val_cv, val_pred))
            
            avg_score = np.mean(cv_scores)
            print(f"  Window {window}: CV Accuracy = {avg_score:.2%}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_window = window
                
        except Exception as e:
            continue
    
    print(f"[OK] Optimal Window: {best_window} days (CV Score: {best_score:.2%})")
    
    try:
        # FINAL MODEL
        final_full_data = create_ml_data_improved(price_data[y_stock], price_data[x_stock], 
                                                   hedge_ratio_beta, best_window)
        final_train_data = final_full_data.loc[TRAIN_START:TRAIN_END].copy()
        final_test_data = final_full_data.loc[TEST_START:TEST_END].copy()
        
        if len(final_test_data) < 20:
            print("[X] Insufficient test data, skipping...")
            continue
        
        if len(final_train_data) < 100:
            print("[X] Insufficient training data, skipping...")
            continue
        
        X_train = final_train_data[feature_cols]
        y_train = final_train_data['target']
        
        if len(np.unique(y_train)) < 2:
            print("[X] Insufficient class variety, skipping...")
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        final_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        final_model.fit(X_train_scaled, y_train)
        
        train_pred = final_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"[OK] Training Accuracy: {train_accuracy:.2%}")
        
        # SIMULATION
        final_test_data, total_pnl = run_simulation_improved(
            final_test_data.copy(), 
            final_model, 
            hedge_ratio_beta, 
            y_stock, 
            x_stock, 
            price_data,
            scaler
        )
        
        # METRICS
        num_trades = (final_test_data['position'].diff().fillna(0) != 0).sum()
        total_cost = final_test_data['txn_cost'].sum()
        
        daily_returns = final_test_data['pnl_net']
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        cumulative = final_test_data['cumulative_pnl']
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        trade_pnls = daily_returns[daily_returns != 0]
        if len(trade_pnls) > 0:
            win_rate = (trade_pnls > 0).sum() / len(trade_pnls)
        else:
            win_rate = 0
        
        print(f"[OK] Test PnL: Rs.{total_pnl:.2f}")
        print(f"[OK] Total Trades: {num_trades}")
        print(f"[OK] Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"[OK] Win Rate: {win_rate:.2%}")
        
        # SAVE DATA
        trades_list.append(pd.DataFrame({
            'Sector': sector_name,
            'Pair': f'{y_stock}/{x_stock}',
            'Date': final_test_data.index,
            'Position': final_test_data['position'],
            'Hedged_Spread': final_test_data['spread'],
            'ZScore': final_test_data['zscore'],
            'Pred_Signal': final_test_data['pred_signal'],
            'Pred_Proba': final_test_data['pred_proba'],
            'PnL_Gross': final_test_data['pnl_gross'],
            'PnL_Net': final_test_data['pnl_net'],
            'Cumulative_PnL': final_test_data['cumulative_pnl']
        }))
        
        summary_list.append({
            'Sector': sector_name,
            'Pair': f'{y_stock}/{x_stock}',
            'Pair_Score': pair_score,
            'Beta': hedge_ratio_beta,
            'Half_Life': half_life,
            'Optimal_Window': best_window,
            'CV_Score': best_score,
            'Training_Accuracy': train_accuracy,
            'Total_Trades': num_trades,
            'Total_PnL_Gross': final_test_data['pnl_gross'].sum(),
            'Total_PnL_Net': total_pnl,
            'Total_Cost': total_cost,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate
        })
        
    except Exception as e:
        error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
        print(f"[X] Error: {error_msg[:50]}")
        continue

# -----------------------------
# [8] SAVE RESULTS
# -----------------------------
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

if trades_list:
    trades_df = pd.concat(trades_list)
    trades_df.to_csv('nse_pairs_trades_detailed.csv', index=False)
    print("\n[OK] Detailed trades saved: nse_pairs_trades_detailed.csv")

if summary_list:
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('Total_PnL_Net', ascending=False)
    summary_df.to_csv('nse_pairs_summary.csv', index=False)
    
    print("\n[INFO] PERFORMANCE SUMMARY")
    print("-" * 70)
    display_cols = ['Sector', 'Pair', 'Pair_Score', 'Total_PnL_Net', 'Sharpe_Ratio', 'Win_Rate', 'Total_Trades']
    print(summary_df[display_cols].to_string(index=False))
    
    print("\n[INFO] SECTOR-WISE PERFORMANCE")
    print("-" * 70)
    sector_summary = summary_df.groupby('Sector').agg({
        'Pair': 'count',
        'Total_PnL_Net': ['sum', 'mean'],
        'Sharpe_Ratio': 'mean',
        'Win_Rate': 'mean',
        'Total_Trades': 'sum'
    }).round(2)
    sector_summary.columns = ['Pairs', 'Total_PnL', 'Avg_PnL', 'Avg_Sharpe', 'Avg_WinRate', 'Trades']
    print(sector_summary.to_string())
    
    print("\n[INFO] AGGREGATE STATISTICS")
    print("-" * 70)
    print(f"Total Pairs Traded: {len(summary_df)}")
    print(f"Total Net PnL: Rs.{summary_df['Total_PnL_Net'].sum():,.2f}")
    print(f"Average PnL per Pair: Rs.{summary_df['Total_PnL_Net'].mean():,.2f}")
    
    if len(summary_df) > 0:
        print(f"Best Pair: {summary_df.iloc[0]['Pair']} [{summary_df.iloc[0]['Sector']}] "
              f"(Rs.{summary_df.iloc[0]['Total_PnL_Net']:,.2f})")
        print(f"Worst Pair: {summary_df.iloc[-1]['Pair']} [{summary_df.iloc[-1]['Sector']}] "
              f"(Rs.{summary_df.iloc[-1]['Total_PnL_Net']:,.2f})")
    
    print(f"Average Sharpe Ratio: {summary_df['Sharpe_Ratio'].mean():.2f}")
    print(f"Average Win Rate: {summary_df['Win_Rate'].mean():.2%}")
    profitable_count = (summary_df['Total_PnL_Net'] > 0).sum()
    print(f"Profitable Pairs: {profitable_count}/{len(summary_df)} "
          f"({profitable_count / len(summary_df) * 100:.1f}%)")
    
    print("\n[INFO] BEST PAIR PER SECTOR")
    print("-" * 70)
    for sector in all_sectors.keys():
        sector_df = summary_df[summary_df['Sector'] == sector]
        if len(sector_df) > 0:
            best = sector_df.iloc[0]
            print(f"  {sector}: {best['Pair']} (PnL: Rs.{best['Total_PnL_Net']:,.2f}, "
                  f"Sharpe: {best['Sharpe_Ratio']:.2f})")
        else:
            print(f"  {sector}: No pairs traded")
    
    print("\n[OK] Summary saved: nse_pairs_summary.csv")
else:
    print("\n[X] No successful trades to report")

print("\n" + "="*70)
print("SIMULATION COMPLETE!")
print("="*70)