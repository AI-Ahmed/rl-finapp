from utils import *
from loguru import logger

import numpy as np


if __name__ == "__main__":
    # testing variables
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    logger.info("Running Process1")
    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces
    )

    logger.info("Markow Property of Process1")
    print(process1_traces)
    print(f"Dimension: {process1_traces.shape}")
    print("\n")

    logger.info("Running process2")
    process2_traces: np.ndarray = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=num_traces
    )
    logger.info("Markow Property of Process2")
    print(process2_traces)
    print(f"Dimension: {process1_traces.shape}")
    print("\n")    
