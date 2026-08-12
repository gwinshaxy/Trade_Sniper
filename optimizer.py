import os
import gc
import time
import random
import logging
import pandas as pd
import numpy as np
import psycopg2
import requests
from multiprocessing import Pool
from deap import base, creator, tools
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GLOBAL_DATA = None

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    try:
        return psycopg2.connect(db_url)
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        return None

def ensure_tables_exist():
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) UNIQUE NOT NULL,
                    tema_period INT NOT NULL DEFAULT 200,
                    rsi_period INT NOT NULL DEFAULT 14,
                    rsi_thresh FLOAT DEFAULT 42.0,
                    zone_tolerance NUMERIC NOT NULL DEFAULT 0.0075,
                    min_sentiment NUMERIC NOT NULL DEFAULT 0.0,
                    risk_pct NUMERIC NOT NULL DEFAULT 1.0,
                    min_rr NUMERIC NOT NULL DEFAULT 2.0,
                    vp_detection_pct NUMERIC NOT NULL DEFAULT 0.07,
                    use_rsi_filter BOOLEAN DEFAULT TRUE,
                    use_candlestick_confirm BOOLEAN DEFAULT TRUE,
                    fitness_score NUMERIC DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to update schema: {e}")

def fetch_binance_klines_direct(symbol="BTC/USDT", interval="1h", limit=3000):
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
        raise Exception(f"Binance API Error Status {response.status_code}")

def fetch_historical_candles(symbol="BTC/USDT", limit=3000):
    ensure_tables_exist()
    clean_symbol = symbol.replace("/", "").upper()
    try:
        df = fetch_binance_klines_direct(symbol=clean_symbol, interval="1h", limit=limit)
        if not df.empty:
            return df
    except Exception:
        pass

    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.02, limit)
    close_prices = 50000 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=limit, freq="1h"),
        "open": close_prices * 0.999, "high": close_prices * 1.002,
        "low": close_prices * 0.998, "close": close_prices,
        "volume": np.random.randint(100, 1000, size=limit)
    })

def save_optimized_parameters(symbol, best_params, fitness_score):
    ensure_tables_exist()
    conn = get_db_connection()
    if not conn:
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
        cursor.execute(query, (clean_symbol, int(tema_period), int(rsi_period), float(zone_tolerance), float(min_sentiment), float(risk_pct), float(min_rr), float(fitness_score)))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to save parameters: {e}")

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def init_worker(data):
    global GLOBAL_DATA
    GLOBAL_DATA = data

def compute_tema(series, period):
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def evaluate_strategy(individual):
    global GLOBAL_DATA
    if GLOBAL_DATA is None or GLOBAL_DATA.empty:
        return (-999.0,)

    tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr = individual
    df_close = GLOBAL_DATA["close"]
    
    tema_series = compute_tema(df_close, period=int(tema_period))
    rsi_series = compute_rsi(df_close, period=int(rsi_period))

    upper_zone = tema_series * (1.0 + zone_tolerance)
    lower_zone = tema_series * (1.0 - zone_tolerance)

    long_signal = (df_close > upper_zone) & (rsi_series > 50)
    short_signal = (df_close < lower_zone) & (rsi_series < 50)
    combined_signal = np.where(long_signal, 1.0, np.where(short_signal, -1.0, 0.0))

    if np.count_nonzero(np.diff(combined_signal)) < 15:
        return (-999.0,)

    returns = df_close.pct_change().fillna(0)
    strategy_returns = returns * pd.Series(combined_signal, index=returns.index).shift(1).fillna(0) * (risk_pct / 100.0)
    std_dev = strategy_returns.std()
    
    if std_dev == 0 or np.isnan(std_dev):
        return (-999.0,)

    sharpe_ratio = (strategy_returns.mean() / std_dev) * np.sqrt(252 * 24)
    return (float(sharpe_ratio),) if not np.isnan(sharpe_ratio) else (-999.0,)

def create_random_individual():
    return creator.Individual([
        random.randint(50, 300), random.randint(7, 30),
        round(random.uniform(0.005, 0.030), 4),
        round(random.uniform(-0.5, 0.5), 2),
        round(random.uniform(0.5, 3.0), 2),
        round(random.uniform(1.2, 4.0), 2)
    ])

def mutate_individual(individual, indpb=0.25):
    if random.random() < indpb: individual[0] = random.randint(50, 300)
    if random.random() < indpb: individual[1] = random.randint(7, 30)
    if random.random() < indpb: individual[2] = round(random.uniform(0.005, 0.030), 4)
    if random.random() < indpb: individual[3] = round(random.uniform(-0.5, 0.5), 2)
    if random.random() < indpb: individual[4] = round(random.uniform(0.5, 3.0), 2)
    if random.random() < indpb: individual[5] = round(random.uniform(1.2, 4.0), 2)
    return (individual,)

toolbox = base.Toolbox()
toolbox.register("individual", create_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_strategy)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_optimization(symbol="BTC/USDT"):
    global GLOBAL_DATA
    data_df = fetch_historical_candles(symbol=symbol, limit=3000)
    GLOBAL_DATA = data_df

    pool = Pool(processes=2, initializer=init_worker, initargs=(data_df,))
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(maxsize=1)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind, toolbox.map(toolbox.evaluate, invalid_ind)):
        ind.fitness.values = fit
    hof.update(pop)

    for _ in range(1, 11):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mutant in offspring:
            if random.random() < 0.35:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, toolbox.map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        gc.collect()

    pool.close()
    pool.join()
    toolbox.unregister("map")

    best_params = hof[0]
    save_optimized_parameters(symbol, best_params, hof[0].fitness.values[0])
    GLOBAL_DATA = None
    gc.collect()
    return best_params

if __name__ == "__main__":
    while True:
        try:
            for symbol in ["BNB/USDT", "ETH/USDT", "SOL/USDT"]:
                run_optimization(symbol=symbol)
        except Exception as e:
            logging.error(f"Error in optimization cycle: {e}")
        time.sleep(86400)