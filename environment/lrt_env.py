import numpy as np
import random
import time

class Lrt2Env:
    def __init__(self):
        self.stations = [
            "Recto", "Legarda", "Pureza", "V. Mapa", "J. Ruiz", "Gilmore", 
            "Betty Go-Belmonte", "Cubao", "Anonas", "Katipunan", "Santolan", 
            "Marikina-Pasig", "Antipolo"
        ]
        self.action_space = [0, 1, 2]
        self._station_count = len(self.stations)
        self.initial_carriages = 3
        self.max_carriages = 8
        self.initial_width_level = 1
        
        # --- CHANGE: Set a new max width level ---
        self.max_width_level = 3 # Can now be upgraded twice (Level 1 -> 2 -> 3)

        self.demand_recto_to_antipolo = { "Recto": 1.0, "Legarda": 0.6, "Pureza": 0.7, "V. Mapa": 0.8, "J. Ruiz": 0.5, "Gilmore": 0.6, "Betty Go-Belmonte": 0.5, "Cubao": 1.5, "Anonas": 0.9, "Katipunan": 1.2, "Santolan": 0.8, "Marikina-Pasig": 0.7 }
        self.demand_antipolo_to_recto = { "Marikina-Pasig": 1.0, "Santolan": 1.2, "Katipunan": 1.5, "Anonas": 0.8, "Cubao": 1.3, "Betty Go-Belmonte": 0.5, "Gilmore": 0.6, "J. Ruiz": 0.5, "V. Mapa": 0.7, "Pureza": 0.6, "Legarda": 0.8, "Recto": 0.5 }
        self.base_capacity_per_carriage = 100
        self.profit_per_passenger = 25.0
        self.unmet_demand_penalty = -30.0
        self.add_carriage_cost = -150.0

        # --- CHANGE: Adjust cost to be more competitive ---
        self.widen_carriage_cost = -400.0 # Lowered from -1000 to make it a viable choice
        
        self.failure_penalty = -5000.0
        self.last_direction = 1
        self.reset()

    def _get_state(self):
        return (self.current_station_idx, self.num_carriages, self.carriage_width_level, self.direction)

    def reset(self):
        self.num_carriages = self.initial_carriages
        self.carriage_width_level = self.initial_width_level
        # --- CHANGE: The _has_widened flag is no longer needed ---
        self.direction = 1 - self.last_direction
        self.last_direction = self.direction
        if self.direction == 0: self.current_station_idx = 0
        else: self.current_station_idx = self._station_count - 1
        return self._get_state()

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Choose from {self.action_space}.")

        # --- CHANGE: Updated failure logic for Lapad ---
        if (action == 0 and self.num_carriages >= self.max_carriages) or \
           (action == 1 and self.carriage_width_level >= self.max_width_level):
            return self._get_state(), self.failure_penalty, True, {"info": "Invalid move failure"}
        
        reward = self._calculate_reward(action)
        
        # --- CHANGE: Updated action logic for Lapad ---
        if action == 0: self.num_carriages += 1
        elif action == 1: self.carriage_width_level += 1
            
        done = False
        if self.direction == 0:
            self.current_station_idx += 1
            if self.current_station_idx == self._station_count - 1: done = True
        else:
            self.current_station_idx -= 1
            if self.current_station_idx == 0: done = True
            
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, action):
        action_cost = 0
        if action == 0: action_cost = self.add_carriage_cost
        elif action == 1: action_cost = self.widen_carriage_cost
        station_name = self.stations[self.current_station_idx]
        demand_model = self.demand_recto_to_antipolo if self.direction == 0 else self.demand_antipolo_to_recto
        capacity = self.num_carriages * (self.base_capacity_per_carriage * self.carriage_width_level)
        demand = demand_model.get(station_name, 0.0) * self.base_capacity_per_carriage
        passengers_served = min(capacity, demand)
        unmet_demand = demand - passengers_served
        revenue = passengers_served * self.profit_per_passenger
        penalty = unmet_demand * self.unmet_demand_penalty
        total_reward = revenue + action_cost + penalty
        return total_reward