from datetime import datetime
import random
import numpy as np
import pandas as pd

from shipmodel import FaultType
from shipmodel import ShipModel


class SimulationRunner:
    def __init__(self, fault_label=0, stop_time=600, n_simulations=1000):
        self.n_simulations = n_simulations
        self.full_dataset = pd.DataFrame()
        self.fault_label = fault_label
        self.model = ShipModel(self.fault_label)
        self.stop_time = stop_time

    def run_simulations(self):
        for sim_id in range(self.n_simulations):
            print(f"Simulation {sim_id}/{self.n_simulations}")

            # 2. Запуск симуляции
            sol, df = self.model.sim(self.stop_time)
            df['sim_id'] = sim_id
            self.full_dataset = pd.concat([self.full_dataset, df], ignore_index=True)

            # Периодическое сохранение на случай прерывания
            if sim_id % 100 == 0:
                self.save_to_file()

    def save_to_file(self):
        sim_number = len(self.full_dataset['sim_id'].unique())
        filename = f"data/simulation_fault{self.fault_label}_length{self.n_simulations}.parquet"
        self.full_dataset.to_parquet(filename)
        print(f"Data saved to {filename} | Total simulations: {sim_number} ")


# Пример использования
if __name__ == "__main__":
    print(f"start time {datetime.now().strftime('%H:%M:%S')}")
    for fault_type in range(6, -1, -1):
        runner = SimulationRunner(n_simulations=1000, fault_label=fault_type, stop_time=900)
        runner.run_simulations()
        runner.save_to_file()
        print(f"Finished fault {fault_type} at {datetime.now().strftime('%H:%M:%S')}")

