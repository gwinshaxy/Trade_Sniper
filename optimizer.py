import os
import gc
import time
import random
import logging
import argparse
import pandas as pd
import numpy as np
from deap import base, creator, tools
from dotenv import load_dotenv

from common import (
    get_db_connection, 
    release_db_connection, 
    ensure_schema_updated, 
    logger, 
    normalize_symbol
)
import strategy

load_dotenv()

GLOBAL_DATA = None

# Baseline exchange friction & Gate Thresholds
FEE_RATE = 0.00075            # 0.075% trading fee per turn
SLIPPAGE_PCT = 0.0005         # 0.05% standard slippage per turn
SLIPPAGE_PCT_STRESS = 0.0010   # 0.10% stress-test slippage per turn
MIN_REQUIRED_FITNESS = 0.18   # Minimum Walk-Forward Fitness required for live persistence

# Lightweight Optimization Parameters (Tuned for Low RAM Constraints)
POPULATION_SIZE = 40
NGEN = 20

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def save_optimized_parameters(symbol: str, best_params: list, fitness_score: float):
    if fitness_score < MIN_REQUIRED_FITNESS:
        logger.warning(
            f"Skipping save for {symbol}: Fitness score ({fitness_score:.4f}) "
            f"did not meet minimum required threshold of {MIN_REQUIRED_FITNESS:.2f}."
        )
        return

    conn = get_db_connection()
    if not conn:
        return
    try:
        formatted_symbol = symbol.strip().upper()
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
            formatted_symbol, int(tema_period), int(rsi_period), float(rsi_thresh), int(adx_period), float(adx_threshold),
            float(max_sl_pct), float(zone_tolerance), float(min_sentiment), float(risk_pct), float(min_rr), float(fitness_score)
        ))
        conn.commit()
        cursor.close()
        logger.info(f"Successfully saved validated parameters for {formatted_symbol} (Walk-Forward Fitness: {fitness_score:.4f})")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to save parameters for {symbol}: {e}")
    finally:
        if conn:
            release_db_connection(conn)


def confirm_market_structure(df_close: pd.Series, lookback: int = 5) -> tuple:
    """Soft Structure Confirmation: Evaluates trend momentum without locking out trades."""
    roll_max = df_close.rolling(window=lookback).max()
    roll_min = df_close.rolling(window=lookback).min()

    struct_long = (roll_max >= roll_max.shift(1)) | (roll_min > roll_min.shift(1))
    struct_short = (roll_max < roll_max.shift(1)) | (roll_min <= roll_min.shift(1))

    return struct_long.fillna(True).to_numpy(), struct_short.fillna(True).to_numpy()


def run_backtest_with_friction(
    df: pd.DataFrame, individual: list, min_trades: int = 1, stress_slippage: float = None
) -> float:
    if df is None or df.empty or len(df) < 50:
        return -999.0

    (
        tema_period, rsi_period, rsi_thresh, adx_period, adx_threshold, 
        max_sl_pct, zone_tolerance, min_sentiment, risk_pct, min_rr
    ) = individual

    df_close = df["close"]
    
    # Calculate indicators
    tema_series = strategy.calc_tema(df_close, period=int(tema_period))
    rsi_series = strategy.calc_rsi(df_close, period=int(rsi_period))
    adx_series = strategy.calc_adx(df, period=int(adx_period))

    struct_long, struct_short = confirm_market_structure(df_close, lookback=max(3, int(rsi_period // 2)))

    # Convert to pure NumPy arrays
    close_arr = df_close.to_numpy()
    tema_arr = tema_series.to_numpy()
    rsi_arr = rsi_series.to_numpy()
    adx_arr = adx_series.to_numpy()

    upper_zone = tema_arr * (1.0 + zone_tolerance)
    lower_zone = tema_arr * (1.0 - zone_tolerance)

    long_signal = (close_arr > upper_zone) & (rsi_arr > rsi_thresh) & (adx_arr > adx_threshold) & struct_long
    short_signal = (close_arr < lower_zone) & (rsi_arr < (100.0 - rsi_thresh)) & (adx_arr > adx_threshold) & struct_short
    
    combined_signal = np.where(long_signal, 1.0, np.where(short_signal, -1.0, 0.0))

    signal_changes = np.diff(combined_signal)
    trades_count = np.count_nonzero(signal_changes)
    
    # Dynamic fold-aware trade minimum (~1 trade per 50 candles in slice)
    dynamic_min_trades = max(min_trades, int(len(df) / 50))
    if trades_count < dynamic_min_trades:
        return -999.0

    # NumPy return & friction calculations
    raw_returns = np.zeros_like(close_arr)
    raw_returns[1:] = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]
    
    position = np.zeros_like(combined_signal)
    position[1:] = combined_signal[:-1]
    
    strategy_returns = raw_returns * position
    turnovers = np.abs(np.diff(position, prepend=0))
    
    # Use stress slippage if provided, otherwise default slippage
    active_slippage = stress_slippage if stress_slippage is not None else SLIPPAGE_PCT
    friction = turnovers * (FEE_RATE + active_slippage)
    
    net_strategy_returns = strategy_returns - friction

    std_dev = np.std(net_strategy_returns)
    if std_dev == 0 or np.isnan(std_dev):
        return -999.0

    sharpe_ratio = (np.mean(net_strategy_returns) / std_dev) * np.sqrt(252 * 24)
    if np.isnan(sharpe_ratio):
        return -999.0

    # Turnover penalty
    turnover_rate = np.sum(turnovers) / len(df)
    regularization_penalty = max(0.0, turnover_rate * 0.2)
    adjusted_sharpe = sharpe_ratio - regularization_penalty

    # Execution count filter
    if trades_count < 10:
        adjusted_sharpe *= (trades_count / 10.0)

    return float(adjusted_sharpe)


def run_walk_forward_backtest(df: pd.DataFrame, individual: list, stress_slippage: float = None) -> float:
    """Robust Rolling Walk-Forward Validation across dynamic fold structures."""
    if df is None or len(df) < 350:
        return -999.0

    for n_folds in [3, 2]:
        fold_size = len(df) // n_folds
        fold_scores = []

        for fold in range(n_folds):
            fold_df = df.iloc[fold * fold_size : (fold + 1) * fold_size].reset_index(drop=True)
            
            split_idx = int(len(fold_df) * 0.70)
            train_sub = fold_df.iloc[:split_idx]
            test_sub = fold_df.iloc[split_idx:]

            train_score = run_backtest_with_friction(train_sub, individual, min_trades=1, stress_slippage=stress_slippage)
            test_score = run_backtest_with_friction(test_sub, individual, min_trades=1, stress_slippage=stress_slippage)

            if train_score > -999.0 and test_score > -999.0:
                fold_scores.append(test_score)

        if len(fold_scores) >= 1:
            return float(np.mean(fold_scores))

    return -999.0


def evaluate_strategy_train(individual: list) -> tuple:
    global GLOBAL_DATA
    fitness = run_walk_forward_backtest(GLOBAL_DATA, individual)
    return (fitness,)


def check_parameter_stability(df: pd.DataFrame, individual: list, perturbations: int = 3) -> float:
    scores = []
    base_score = run_walk_forward_backtest(df, individual)
    if base_score <= -999.0:
        return -999.0
    scores.append(base_score)

    for _ in range(perturbations):
        neighbor = list(individual)
        neighbor[0] = int(np.clip(neighbor[0] + random.choice([-2, 2]), 10, 200))
        neighbor[2] = round(float(np.clip(neighbor[2] + random.choice([-1.0, 1.0]), 10.0, 48.0)), 1)
        
        neighbor_score = run_walk_forward_backtest(df, neighbor)
        if neighbor_score > -999.0:
            scores.append(neighbor_score)

    return float(np.mean(scores)) if len(scores) > 0 else -999.0


def create_random_individual():
    return creator.Individual([
        random.randint(10, 200),                 # tema_period
        random.randint(5, 25),                    # rsi_period
        round(random.uniform(10.0, 48.0), 1),     # rsi_thresh
        random.randint(5, 25),                    # adx_period
        round(random.uniform(3.0, 28.0), 1),      # adx_threshold
        round(random.uniform(0.008, 0.03), 4),    # max_sl_pct
        round(random.uniform(0.00005, 0.02), 5),  # zone_tolerance
        round(random.uniform(-0.5, 0.5), 2),      # min_sentiment
        round(random.uniform(0.5, 1.5), 2),       # risk_pct
        round(random.uniform(1.2, 3.0), 2)        # min_rr
    ])


def mutate_individual(individual, indpb=0.20):
    if random.random() < indpb:
        individual[0] = int(np.clip(individual[0] + int(random.gauss(0, 8)), 10, 200))
    if random.random() < indpb:
        individual[1] = int(np.clip(individual[1] + int(random.gauss(0, 2)), 5, 25))
    if random.random() < indpb:
        individual[2] = round(float(np.clip(individual[2] + random.gauss(0, 1.2), 10.0, 48.0)), 1)
    if random.random() < indpb:
        individual[3] = int(np.clip(individual[3] + int(random.gauss(0, 2)), 5, 25))
    if random.random() < indpb:
        individual[4] = round(float(np.clip(individual[4] + random.gauss(0, 1.5), 3.0, 28.0)), 1)
    if random.random() < indpb:
        individual[5] = round(float(np.clip(individual[5] + random.gauss(0, 0.005), 0.008, 0.03)), 4)
    if random.random() < indpb:
        individual[6] = round(float(np.clip(individual[6] + random.gauss(0, 0.002), 0.00005, 0.02)), 5)
    if random.random() < indpb:
        individual[7] = round(float(np.clip(individual[7] + random.gauss(0, 0.1), -0.5, 0.5)), 2)
    if random.random() < indpb:
        individual[8] = round(float(np.clip(individual[8] + random.gauss(0, 0.2), 0.5, 1.5)), 2)
    if random.random() < indpb:
        individual[9] = round(float(np.clip(individual[9] + random.gauss(0, 0.3), 1.2, 3.0)), 2)
    return (individual,)


toolbox = base.Toolbox()
toolbox.register("individual", create_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_strategy_train)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("map", map)


def fetch_klines_with_retry(symbol: str, interval: str = "1h", limit: int = 5000, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        df = strategy.fetch_klines(symbol=symbol, interval=interval, limit=limit)
        if df is not None and not df.empty and len(df) >= 350:
            return df
        logger.info(f"Retrying kline fetch for {symbol} (Attempt {attempt}/{max_retries})...")
        time.sleep(2 * attempt)
    return pd.DataFrame()


def run_optimization(symbol="XRP/USDT"):
    global GLOBAL_DATA
    data_df = fetch_klines_with_retry(symbol=symbol, interval="1h", limit=5000)
    if data_df.empty or len(data_df) < 350:
        logger.warning(f"Insufficient historical klines for {symbol}. Skipping optimization.")
        return None

    GLOBAL_DATA = data_df

    try:
        pop = toolbox.population(n=POPULATION_SIZE)
        hof = tools.HallOfFame(maxsize=10)

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, toolbox.map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit
        hof.update(pop)

        for gen in range(1, NGEN + 1):
            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.65:
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

        validated_candidate = None
        best_final_score = -999.0

        for candidate in hof:
            wf_score = candidate.fitness.values[0]
            if wf_score < MIN_REQUIRED_FITNESS:
                continue

            plateau_score = check_parameter_stability(data_df, candidate)
            if plateau_score < MIN_REQUIRED_FITNESS:
                continue

            stress_score = run_walk_forward_backtest(data_df, candidate, stress_slippage=SLIPPAGE_PCT_STRESS)
            if stress_score <= -0.5:
                logger.info(f"Candidate for {symbol} failed 0.10% stress slippage check (Stress Sharpe: {stress_score:.4f}). Skipping.")
                continue

            validated_candidate = candidate
            best_final_score = plateau_score
            break

        if validated_candidate is not None:
            save_optimized_parameters(symbol, validated_candidate, best_final_score)
            return validated_candidate

        logger.warning(f"No candidates passed walk-forward cross-validation, stability, and stress-test for {symbol}.")
        return None

    finally:
        GLOBAL_DATA = None
        del data_df
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Optimizer Engine")
    parser.add_argument("--once", action="store_true", help="Run optimization cycle once for all symbols and exit")
    args = parser.parse_args()

    ensure_schema_updated()
    symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL") or os.getenv("TRADING_SYMBOLS") or "XRP/USDT,BNB/USDT,SOL/USDT,LINK/USDT"
    symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]

    if args.once:
        logger.info("Executing single optimization pass (--once specified)...")
        for symbol in symbols:
            logger.info(f"Starting DEAP Walk-Forward optimization run for {symbol}...")
            run_optimization(symbol=symbol)
            time.sleep(2)
        logger.info("Single-pass optimization complete. Exiting clean.")
    else:
        while True:
            try:
                for symbol in symbols:
                    logger.info(f"Starting DEAP Walk-Forward optimization run for {symbol}...")
                    run_optimization(symbol=symbol)
                    time.sleep(10)
                    
                logger.info("Optimization cycle complete. Sleeping for 24 hours...")
                time.sleep(86400)
            except Exception as e:
                logger.error(f"Error in optimization cycle: {e}")
                time.sleep(3600)