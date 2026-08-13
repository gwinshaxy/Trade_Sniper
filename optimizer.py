import os
import gc
import time
import random
import logging
import pandas as pd
import numpy as np
import psycopg2
from multiprocessing import Pool
from deap import base, creator, tools
from dotenv import load_dotenv

from common import get_db_connection, ensure_schema_updated
import strategy

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GLOBAL_DATA = None

def save_optimized_parameters(symbol, best_params, fitness_score):
    """Saves or updates optimized strategy parameters in the database."""
    ensure_schema_updated()
    conn = get_db_connection()
    if not conn:
        return
    try:
        clean_symbol = symbol.replace("/", "").strip().upper()
        cursor = conn.cursor()
        (
            tema_period, rsi_period, rsi_thresh, adx_period, adx_threshold, 
            max_sl_pct, zone_tolerance, min_sentiment, risk_pct, min_rr
        ) = best_params

        query = """
            INSERT INTO strategy_parameters (
                symbol, tema_period, rsi_period, rsi_thresh, adx_period, adx_threshold, 
                max_sl_pct, zone_tolerance, min_sentiment, risk_pct, min_rr, fitness_score, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (symbol) 
            DO UPDATE SET tema_period = EXCLUDED.tema_period,
                          rsi_period = EXCLUDED.rsi_period,
                          rsi_thresh = EXCLUDED.rsi_thresh,
                          adx_period = EXCLUDED.adx_period,
                          adx_threshold = EXCLUDED.adx_threshold,
                          max_sl_pct = EXCLUDED.max_sl_pct,
                          zone_tolerance = EXCLUDED.zone_tolerance,
                          min_sentiment = EXCLUDED.min_sentiment,
                          risk_pct = EXCLUDED.risk_pct,
                          min_rr = EXCLUDED.min_rr,
                          fitness_score = EXCLUDED.fitness_score,
                          updated_at = NOW();
        """
        cursor.execute(query, (
            clean_symbol, int(tema_period), int(rsi_period), float(rsi_thresh), int(adx_period), float(adx_threshold),
            float(max_sl_pct), float(zone_tolerance), float(min_sentiment), float(risk_pct), float(min_rr), float(fitness_score)
        ))
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Successfully saved DEAP optimization parameters for {clean_symbol} (Fitness: {fitness_score:.4f})")
    except Exception as e:
        logging.error(f"Failed to save parameters for {symbol}: {e}")

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def init_worker(data):
    global GLOBAL_DATA
    GLOBAL_DATA = data

def evaluate_strategy(individual):
    global GLOBAL_DATA
    if GLOBAL_DATA is None or GLOBAL_DATA.empty or len(GLOBAL_DATA) < 300:
        return (-999.0,)

    (
        tema_period, rsi_period, rsi_thresh, adx_period, adx_threshold, 
        max_sl_pct, zone_tolerance, min_sentiment, risk_pct, min_rr
    ) = individual

    df_close = GLOBAL_DATA["close"]
    
    tema_series = strategy.calc_tema(df_close, period=int(tema_period))
    rsi_series = strategy.calc_rsi(df_close, period=int(rsi_period))
    adx_series = strategy.calc_adx(GLOBAL_DATA, period=int(adx_period))

    upper_zone = tema_series * (1.0 + zone_tolerance)
    lower_zone = tema_series * (1.0 - zone_tolerance)

    long_signal = (df_close > upper_zone) & (rsi_series > rsi_thresh) & (adx_series > adx_threshold)
    short_signal = (df_close < lower_zone) & (rsi_series < (100.0 - rsi_thresh)) & (adx_series > adx_threshold)
    combined_signal = np.where(long_signal, 1.0, np.where(short_signal, -1.0, 0.0))

    # Reject strategies with insufficient trading frequency
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
        random.randint(50, 300),              # tema_period
        random.randint(7, 30),                # rsi_period
        round(random.uniform(30.0, 50.0), 1), # rsi_thresh
        random.randint(7, 30),                # adx_period
        round(random.uniform(15.0, 35.0), 1), # adx_threshold
        round(random.uniform(0.01, 0.05), 3), # max_sl_pct
        round(random.uniform(0.005, 0.030), 4),# zone_tolerance
        round(random.uniform(-0.5, 0.5), 2),  # min_sentiment
        round(random.uniform(0.5, 3.0), 2),    # risk_pct
        round(random.uniform(1.2, 4.0), 2)     # min_rr
    ])

def mutate_individual(individual, indpb=0.25):
    if random.random() < indpb: individual[0] = random.randint(50, 300)
    if random.random() < indpb: individual[1] = random.randint(7, 30)
    if random.random() < indpb: individual[2] = round(random.uniform(30.0, 50.0), 1)
    if random.random() < indpb: individual[3] = random.randint(7, 30)
    if random.random() < indpb: individual[4] = round(random.uniform(15.0, 35.0), 1)
    if random.random() < indpb: individual[5] = round(random.uniform(0.01, 0.05), 3)
    if random.random() < indpb: individual[6] = round(random.uniform(0.005, 0.030), 4)
    if random.random() < indpb: individual[7] = round(random.uniform(-0.5, 0.5), 2)
    if random.random() < indpb: individual[8] = round(random.uniform(0.5, 3.0), 2)
    if random.random() < indpb: individual[9] = round(random.uniform(1.2, 4.0), 2)
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
    data_df = strategy.fetch_klines(symbol=symbol, interval="1h", limit=3000)
    if data_df.empty:
        logging.warning(f"Could not fetch historical klines for {symbol}. Skipping optimization.")
        return None

    GLOBAL_DATA = data_df

    pool = Pool(processes=2, initializer=init_worker, initargs=(data_df,))
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(maxsize=1)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind, toolbox.map(toolbox.evaluate, invalid_ind)):
        ind.fitness.values = fit
    hof.update(pop)

    for gen in range(1, 11):
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

    if len(hof) > 0:
        best_params = hof[0]
        save_optimized_parameters(symbol, best_params, hof[0].fitness.values[0])
        GLOBAL_DATA = None
        gc.collect()
        return best_params

    GLOBAL_DATA = None
    gc.collect()
    return None

if __name__ == "__main__":
    ensure_schema_updated()
    symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
    symbols = [s.strip().upper() for s in symbols_env.split(",")]

    while True:
        try:
            for symbol in symbols:
                logging.info(f"Starting DEAP optimization run for {symbol}...")
                run_optimization(symbol=symbol)
        except Exception as e:
            logging.error(f"Error in optimization cycle: {e}")
        time.sleep(86400)