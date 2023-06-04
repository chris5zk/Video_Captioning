# -*- coding: utf-8 -*-
"""
Created on Sat June 3 12:33:17 2023
@title: training log
"""

import os
import logging
from datetime import datetime


def create_logger(cfg):
    tfm = "%y%m%d-%H%M%S"
    timestamp = datetime.now().strftime(tfm)
    log_path = cfg.log_root + timestamp.split('-')[0] + '/'
    full_path = log_path + f"{timestamp.split('-')[1]}_train.logfile"
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(filename=full_path, format="[%(asctime)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, log_path, timestamp


