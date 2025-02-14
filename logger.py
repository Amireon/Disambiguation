import logging
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="./logs/", log_file_prefix="experiment_log"):
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 定义日志文件名
        log_file = os.path.join(log_dir, f"{log_file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置日志
        self.logger = logging.getLogger("ExperimentLogger")
        self.logger.setLevel(logging.INFO)
        
        # 日志文件Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 控制台输出Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加Handler到Logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("Logger initialized.")
    
    def log_info(self, message):
        self.logger.info(message)
    
    def log_warning(self, message):
        self.logger.warning(message)
    
    def log_error(self, message):
        self.logger.error(message)
    
    def log_experiment_params(self, params: dict):
        """记录实验的参数信息"""
        self.logger.info("Experiment Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_results(self, task_name, metric_values: dict):
        """记录实验结果"""
        self.logger.info(f"Results for {task_name}:")
        for metric, value in metric_values.items():
            self.logger.info(f"  {metric}: {value}")