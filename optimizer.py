import os
import gc
import time
import random
import logging
import warnings
import pandas as pd
import numpy as np
import psycopg2
import requests
from multiprocessing import Pool
from deap import base, creator, tools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Variable for Memory Space across process boundaries
GLOBAL_DATA = None

# ---------------------------------------------------------------------------
# 1. Database Helper & Schema Assurance Functions
# ---------------------------------------------------------------------------
def get_db_connection():
    """Establishes a connection to PostgreSQL/Supabase using DATABASE_URL."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logging.error("DATABASE_URL environment variable is not set.")
        return None
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        return None

def ensure_tables_exist():
    """Ensures strategy_parameters table exists and contains all required hyperparameter columns."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # Create candles table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE,
                    symbol VARCHAR(20) NOT NULL,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume NUMERIC
                );
            """)

            # Create base strategy_parameters table if absent
            cur.execute("""
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) UNIQUE NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Schema migration: Ensure all required parameter columns exist
            columns_to_add = [
                ("tema_period", "INT NOT NULL DEFAULT 200"),
                ("rsi_period", "INT NOT NULL DEFAULT 14"),
                ("zone_tolerance", "NUMERIC NOT NULL DEFAULT 0.015"),
                ("min_sentiment", "NUMERIC NOT NULL DEFAULT 0.0"),
                ("risk_pct", "NUMERIC NOT NULL DEFAULT 1.0"),
                ("min_rr", "NUMERIC NOT NULL DEFAULT 2.0"),
                ("fitness_score", "NUMERIC DEFAULT 0.0")
            ]

            for col_name, col_type in columns_to_add:
                cur.execute(f"""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='strategy_parameters' AND column_name='{col_name}'
                        ) THEN
                            ALTER TABLE strategy_parameters ADD COLUMN {col_name} {col_type};
                        END IF;
                    END $$;
                """)

            conn.commit()
        conn.close()
        logging.info("Database schema verification/migration completed.")
    except Exception as e:
        logging.error(f"Failed to verify or update database schema: {e}")

def fetch_binance_klines_direct(symbol="BTCUSDT", interval="1h", limit=3000):
    """Fallback direct REST fetch from Binance public API."""
    clean_symbol = symbol.replace("/", "").upper()
    url = f"https://api.binance.com/api/v3/klines?symbol={clean_symbol}&interval={interval}&limit={limit}"
    
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    else:
        raise Exception(f"Binance Direct API Returned Status {response.status_code}")

def fetch_historical_candles(symbol="BTCUSDT", limit=3000):
    """Fetch candle data using DB -> Direct API -> Synthetic pipeline."""
    ensure_tables_exist()
    clean_symbol = symbol.replace("/", "").upper()
    
    conn = get_db_connection()
    if conn:
        try:
            query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol = %s ORDER BY timestamp DESC LIMIT %s;"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df = pd.read_sql_query(query, conn, params=(clean_symbol, limit))
            conn.close()
            if not df.empty and len(df) > 100:
                df = df.iloc[::-1].reset_index(drop=True)
                return df
        except Exception as e:
            logging.warning(f"Could not load candles from SQL DB: {e}.")
            if conn:
                conn.close()

    try:
        logging.info(f"Fetching live historical klines directly for {clean_symbol}...")
        df = fetch_binance_klines_direct(symbol=clean_symbol, interval="1h", limit=limit)
        if not df.empty:
            return df
    except Exception as e:
        logging.warning(f"Failed direct klines API fetch for {clean_symbol}: {e}")

    logging.info("Falling back to generated sample market data...")
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.02, limit)
    close_prices = 50000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=limit, freq="1h"),
        "open": close_prices * 0.999,
        "high": close_prices * 1.002,
        "low": close_prices * 0.998,
        "close": close_prices,
        "volume": np.random.randint(100, 1000, size=limit)
    })
    return df

def save_optimized_parameters(symbol, best_params, fitness_score):
    """Saves winning strategy parameters directly to Supabase/PostgreSQL."""
    ensure_tables_exist()
    conn = get_db_connection()
    if not conn:
        logging.warning("Skipping database update: No DB connection.")
        return
    
    try:
        clean_symbol = symbol.replace("/", "").upper()
        cursor = conn.cursor()
        
        tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr = best_params
        
        query = """
            INSERT INTO strategy_parameters (
                symbol, tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr, fitness_score, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (symbol) 
            DO UPDATE SET tema_period = EXCLUDED.tema_period,
                          rsi_period = EXCLUDED.rsi_period,
                          zone_tolerance = EXCLUDED.zone_tolerance,
                          min_sentiment = EXCLUDED.min_sentiment,
                          risk_pct = EXCLUDED.risk_pct,
                          min_rr = EXCLUDED.min_rr,
                          fitness_score = EXCLUDED.fitness_score,
                          updated_at = NOW();
        """
        cursor.execute(query, (
            clean_symbol, int(tema_period), int(rsi_period), 
            float(zone_tolerance), float(min_sentiment), 
            float(risk_pct), float(min_rr), float(fitness_score)
        ))
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Successfully saved updated parameters for {clean_symbol} to Supabase.")
    except Exception as e:
        logging.error(f"Failed to save parameters to Supabase: {e}")

# ---------------------------------------------------------------------------
# 2. DEAP Initialization & Evaluation Pipeline
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def init_worker(data):
    """Initializer function to pass dataframe across process memory boundaries."""
    global GLOBAL_DATA
    GLOBAL_DATA = data

def compute_ema(series, period):
    """Computes Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def compute_tema(series, period):
    """Computes Triple Exponential Moving Average (TEMA)."""
    ema1 = compute_ema(series, period)
    ema2 = compute_ema(ema1, period)
    ema3 = compute_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def compute_rsi(series, period=14):
    """Computes RSI indicator vectorially."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def evaluate_strategy(individual):
    """
    Backtesting evaluation function for all candidate strategy parameters:
    [tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr]
    """
    global GLOBAL_DATA
    if GLOBAL_DATA is None or GLOBAL_DATA.empty:
        return (-999.0,)

    tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr = individual

    df = GLOBAL_DATA
    df_close = df["close"]
    
    # Calculate indicators
    tema_series = compute_tema(df_close, period=int(tema_period))
    rsi_series = compute_rsi(df_close, period=int(rsi_period))

    # Trend Determination relative to TEMA and Zone Tolerance band
    upper_zone = tema_series * (1.0 + zone_tolerance)
    lower_zone = tema_series * (1.0 - zone_tolerance)

    # Long signals when price breaks above TEMA band with RSI confirmation (>50)
    # Short signals when price drops below TEMA band with RSI confirmation (<50)
    long_signal = (df_close > upper_zone) & (rsi_series > 50)
    short_signal = (df_close < lower_zone) & (rsi_series < 50)

    combined_signal = np.where(long_signal, 1.0, np.where(short_signal, -1.0, 0.0))

    # Minimum Trade Execution Constraint
    trades_count = np.count_nonzero(np.diff(combined_signal))
    if trades_count < 15:
        return (-999.0,)

    # Calculate strategy returns scaled by risk_pct and minimum R:R expectation
    returns = df_close.pct_change().fillna(0)
    effective_leverage = (risk_pct / 100.0) * (min_rr / 2.0)
    strategy_returns = returns * pd.Series(combined_signal, index=returns.index).shift(1).fillna(0) * effective_leverage

    mean_ret = strategy_returns.mean()
    std_dev = strategy_returns.std()
    
    if std_dev == 0 or np.isnan(std_dev) or np.count_nonzero(combined_signal) == 0:
        return (-999.0,)

    # Annualized Sharpe ratio for hourly candles
    sharpe_ratio = (mean_ret / std_dev) * np.sqrt(252 * 24)
    
    if np.isnan(sharpe_ratio):
        return (-999.0,)

    return (float(sharpe_ratio),)

# Individual Generator Setup
def create_random_individual():
    tema_period = random.randint(50, 300)
    rsi_period = random.randint(7, 30)
    zone_tolerance = round(random.uniform(0.005, 0.030), 4)
    min_sentiment = round(random.uniform(-0.5, 0.5), 2)
    risk_pct = round(random.uniform(0.5, 3.0), 2)
    min_rr = round(random.uniform(1.2, 4.0), 2)
    return creator.Individual([tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr])

def mutate_individual(individual, indpb=0.25):
    """Custom mutation function supporting float and integer parameter types."""
    if random.random() < indpb:
        individual[0] = random.randint(50, 300)      # tema_period
    if random.random() < indpb:
        individual[1] = random.randint(7, 30)        # rsi_period
    if random.random() < indpb:
        individual[2] = round(random.uniform(0.005, 0.030), 4) # zone_tolerance
    if random.random() < indpb:
        individual[3] = round(random.uniform(-0.5, 0.5), 2)   # min_sentiment
    if random.random() < indpb:
        individual[4] = round(random.uniform(0.5, 3.0), 2)    # risk_pct
    if random.random() < indpb:
        individual[5] = round(random.uniform(1.2, 4.0), 2)    # min_rr
    return (individual,)

toolbox = base.Toolbox()
toolbox.register("individual", create_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_strategy)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------------------------
# 3. Main Optimization Engine & Multi-Asset Scheduling Loop
# ---------------------------------------------------------------------------
def run_optimization(symbol="BTCUSDT"):
    global GLOBAL_DATA

    logging.info(f"--- Fetching historical data for {symbol} ---")
    data_df = fetch_historical_candles(symbol=symbol, limit=3000)
    GLOBAL_DATA = data_df

    POP_SIZE = 20
    NGEN = 10
    CXPB, MUTPB = 0.6, 0.35

    logging.info(f"Starting Multi-Parameter DEAP GA Optimization on {symbol} (Pop: {POP_SIZE}, Gen: {NGEN})")

    pool = Pool(processes=2, initializer=init_worker, initargs=(data_df,))
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(maxsize=1)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    for gen in range(1, NGEN + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        best_in_gen = hof[0].fitness.values[0]
        logging.info(f"[{symbol}] Generation {gen}/{NGEN} Complete | Best Fitness Score: {best_in_gen:.4f}")

        del offspring
        del invalid_ind
        gc.collect()

    pool.close()
    pool.join()
    toolbox.unregister("map")

    best_params = hof[0]
    best_score = hof[0].fitness.values[0]
    logging.info(f"--- Optimization Finished for {symbol} ---")
    logging.info(f"Best Parameters: TEMA={best_params[0]}, RSI={best_params[1]}, "
                 f"ZoneTol={best_params[2]}, MinSentiment={best_params[3]}, "
                 f"RiskPct={best_params[4]}%, MinRR={best_params[5]}")
    logging.info(f"Best Fitness: {best_score:.4f}")

    save_optimized_parameters(symbol, best_params, best_score)
    
    GLOBAL_DATA = None
    gc.collect()

    return best_params

def start_reoptimization_loop(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], interval_hours=24):
    """Runs continuous re-optimization cycle across target pairs every interval_hours."""
    sleep_seconds = interval_hours * 3600
    logging.info(f"Starting continuous multi-asset re-optimization service. Loop interval: {interval_hours} hours.")
    
    while True:
        try:
            for symbol in symbols:
                logging.info(f"Executing scheduled parameter update for {symbol}...")
                run_optimization(symbol=symbol)
        except Exception as e:
            logging.error(f"Error encountered during optimization cycle: {e}")
            
        logging.info(f"Optimization run completed for all assets. Waiting {interval_hours} hours until next run...")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    start_reoptimization_loop(symbols=["BNBUSDT", "ETHUSDT", "SOLUSDT"], interval_hours=24)