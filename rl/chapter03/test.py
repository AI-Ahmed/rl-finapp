import logging
from chapter03.utils import *
from chapter03.plt_utils import *

from loguru import logger

import numpy as np

logger.add("../ch03_logs.log",
           level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")

# Get or create a logger using logging.getLogger
logging_logger = logging.getLogger(__name__)

# Attach the loguru handler to the existing logger
logging_logger.addHandler(logger._core.handlers[0])


if __name__ == "__main__":
    # testing variables
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    logger.debug("Running Process1")
    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces
    )

    logger.info("Markov Property of Process1")
    print(process1_traces)
    print(f"Dimension: {process1_traces.shape}")

# ------------------------------------------------------------
    logger.debug("Running process2")
    process2_traces: np.ndarray = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=num_traces
    )
    logger.info("Markov Property of Process2")
#    print(process2_traces)
    print(f"Dimension: {process2_traces.shape}")

# ------------------------------------------------------------
    logger.debug("Running process3")
    process3_traces: np.ndarray = process3_price_traces(
            start_price=start_price,
            alpha3=alpha3,
            time_steps=time_steps,
            num_traces=num_traces)

    logger.info("Markov Property of Process3")
#    print(process3_traces)
    print(f"Dimension: {process3_traces.shape}")

# ------------------------------------------------------------
    logger.info("Stock Price Markov Process3")

    stock_price_process3_traces: np.ndarray = mkprocess3_price_traces(
            start_price=start_price,
            alpha3=alpha3,
            time_steps=time_steps,
            num_traces=num_traces)

    logger.info("Markov Property of Process3")
    print(f"Dimension: {process3_traces.shape}")
# ------------------------------------------------------------
    trace1 = process1_traces[0]
    trace2 = process2_traces[0]
    trace3 = process3_traces[0]
    trace4 = stock_price_process3_traces[0]

    logger.debug("Plotting the curves")
    plot_single_trace_all_processes(trace2, trace3, trace4)

    plot_distribution_at_time_all_processes(
        process2_traces,
        process3_traces,
        stock_price_process3_traces
    )
