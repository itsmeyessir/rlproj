import numpy as np
import random
import time

class Lrt2Env:
    """
    V3.2: Standardized reset to alternate direction instead of random.
    """
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
        self.max_width_level = 2
        self.demand_recto_to_antipolo = {
            "Recto": 1.0, "Legarda": 0.6, "Pureza": 0.7, "V. Mapa": 0.8,
            "J. Ruiz": 0.5, "Gilmore": 0.6, "Betty Go-Belmonte": 0.5,
            "Cubao": 1.5, "Anonas": 0.9, "Katipunan": 1.2, "Santolan": 0.8,
            "Marikina-Pasig": 0.7
        }
        self.demand_antipolo_to_recto = {
            "Marikina-Pasig": 1.0, "Santolan": 1.2, "Katipunan": 1.5, "Anonas": 0.8,
            "Cubao": 1.3, "Betty Go-Belmonte": 0.5, "Gilmore": 0.6, "J. Ruiz": 0.5,
            "V. Mapa": 0.7, "Pureza": 0.6, "Legarda": 0.8, "Recto": 0.5
        }
        self.base_capacity_per_carriage = 100
        self.profit_per_passenger = 25.0
        self.unmet_demand_penalty = -30.0
        self.add_carriage_cost = -150.0
        self.widen_carriage_cost = -1000.0

        # --- CHANGE: Initialize last_direction so the first trip is always R->A ---
        self.last_direction = 1 # Start with 1 (A->R) so the first reset flips it to 0 (R->A)
        self.reset()

    def _get_state(self):
        return (self.current_station_idx, self.num_carriages, self.carriage_width_level, self.direction)

    def reset(self):
        self.num_carriages = self.initial_carriages
        self.carriage_width_level = self.initial_width_level
        self._has_widened = False
        
        # --- CHANGE: Alternate direction instead of random ---
        self.direction = 1 - self.last_direction # Flip the direction (0 becomes 1, 1 becomes 0)
        self.last_direction = self.direction     # Remember this trip's direction for the next reset
        
        if self.direction == 0: # Recto -> Antipolo
            self.current_station_idx = 0
        else: # Antipolo -> Recto
            self.current_station_idx = self._station_count - 1
            
        return self._get_state()

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Choose from {self.action_space}.")
        reward = self._calculate_reward(action)
        if action == 0 and self.num_carriages < self.max_carriages:
            self.num_carriages += 1
        elif action == 1 and not self._has_widened:
            self.carriage_width_level = self.max_width_level
            self._has_widened = True
        done = False
        if self.direction == 0:
            self.current_station_idx += 1
            if self.current_station_idx == self._station_count - 1:
                done = True
        else:
            self.current_station_idx -= 1
            if self.current_station_idx == 0:
                done = True
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, action):
        action_cost = 0
        if action == 0:
            action_cost = self.add_carriage_cost
        elif action == 1 and not self._has_widened:
            action_cost = self.widen_carriage_cost
        station_name = self.stations[self.current_station_idx]
        if self.direction == 0:
            demand_model = self.demand_recto_to_antipolo
        else:
            demand_model = self.demand_antipolo_to_recto
        capacity = self.num_carriages * (self.base_capacity_per_carriage * self.carriage_width_level)
        demand = demand_model.get(station_name, 0.0) * self.base_capacity_per_carriage
        passengers_served = min(capacity, demand)
        unmet_demand = demand - passengers_served
        revenue = passengers_served * self.profit_per_passenger
        penalty = unmet_demand * self.unmet_demand_penalty
        total_reward = revenue + action_cost + penalty
        return total_reward