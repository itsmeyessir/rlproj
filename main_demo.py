import pygame
import os
import time
import random

# Import our custom classes
from environment.lrt_env import Lrt2Env
from agents.monte_carlo_agent import MonteCarloAgent
from agents.q_learning_agent import QLearningAgent
from agents.actor_critic_agent import ActorCriticAgent

# --- Pygame and Demo Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
WHITE = (255, 255, 255); BLACK = (0, 0, 0); BLUE = (0, 0, 255); RED = (255, 0, 0)
GREEN = (0, 200, 0); PURPLE = (147, 112, 219); STATION_COLOR = (200, 200, 200)
FONT_SIZE_LARGE = 48; FONT_SIZE_MEDIUM = 32; FONT_SIZE_SMALL = 24
STEP_DELAY_MS = 1500

# --- Best Performing Models ---
BEST_MODELS = {
    'MonteCarlo': 'models/MonteCarlo_low_discount_factor_policy.pkl',
    'QLearning': 'models/QLearning_default_params_policy.pkl',
    'ActorCritic': 'models/ActorCritic_low_discount_policy.pt'
}
AGENT_CLASSES = { 'MonteCarlo': MonteCarloAgent, 'QLearning': QLearningAgent, 'ActorCritic': ActorCriticAgent }

# --- Drawing Functions ---
def draw_environment(screen, font, stations):
    track_y = SCREEN_HEIGHT * 0.6
    pygame.draw.line(screen, BLACK, (50, track_y), (SCREEN_WIDTH - 50, track_y), 4)
    for i, name in enumerate(stations):
        x_pos = 50 + i * ((SCREEN_WIDTH - 100) / (len(stations) - 1))
        pygame.draw.circle(screen, STATION_COLOR, (x_pos, track_y), 10)
        pygame.draw.circle(screen, BLACK, (x_pos, track_y), 10, 2)
        text = font.render(name, True, BLACK)
        text_rect = text.get_rect(center=(x_pos, track_y - 30))
        screen.blit(text, text_rect)

def draw_train(screen, state, num_stations):
    station_idx, num_carriages, width_level, _ = state
    track_y = SCREEN_HEIGHT * 0.6
    train_width = 20 * num_carriages
    train_height = 30 * width_level
    x_pos = 50 + station_idx * ((SCREEN_WIDTH - 100) / (num_stations - 1))
    train_rect = pygame.Rect(0, 0, train_width, train_height)
    train_rect.center = (x_pos, track_y)
    pygame.draw.rect(screen, PURPLE, train_rect)
    pygame.draw.rect(screen, BLACK, train_rect, 2)

def draw_capacity_bar(screen, font, env, state):
    station_idx, num_carriages, width_level, direction = state
    capacity = num_carriages * (env.base_capacity_per_carriage * width_level)
    station_name = env.stations[station_idx]
    demand_model = env.demand_recto_to_antipolo if direction == 0 else env.demand_antipolo_to_recto
    demand = demand_model.get(station_name, 0.0) * env.base_capacity_per_carriage
    max_bar_width = SCREEN_WIDTH - 100
    capacity_width = (capacity / (env.max_carriages * env.max_width_level * env.base_capacity_per_carriage)) * max_bar_width
    demand_width = (demand / (env.max_carriages * env.max_width_level * env.base_capacity_per_carriage)) * max_bar_width
    display_text(screen, font, f"Capacity: {int(capacity)}", (50, 20))
    pygame.draw.rect(screen, GREEN, (50, 50, capacity_width, 30))
    display_text(screen, font, f"Demand: {int(demand)}", (50, 90))
    pygame.draw.rect(screen, RED, (50, 50, min(demand_width, max_bar_width), 30))

def display_text(screen, font, text, position, color=BLACK):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def reset_episode(env):
    state = env.reset()
    cumulative_reward = 0
    last_action_name = "Starting new episode..."
    return state, cumulative_reward, last_action_name

def run_demo():
    random.seed(time.time())
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("LRT-2 'Dagdag o Lapad' RL Demo")
    clock = pygame.time.Clock()
    font_lg = pygame.font.Font(None, FONT_SIZE_LARGE); font_md = pygame.font.Font(None, FONT_SIZE_MEDIUM); font_sm = pygame.font.Font(None, FONT_SIZE_SMALL)
    env = Lrt2Env()
    agent = None
    game_state = 'MENU'
    mc_button = pygame.Rect(300, 200, 600, 50); ql_button = pygame.Rect(300, 270, 600, 50); ac_button = pygame.Rect(300, 340, 600, 50); human_button = pygame.Rect(300, 410, 600, 50)
    state, cumulative_reward, last_action_name = None, 0, ""
    last_step_time = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if game_state == 'MENU':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    agent_to_load = None
                    if mc_button.collidepoint(event.pos): agent_to_load = 'MonteCarlo'
                    elif ql_button.collidepoint(event.pos): agent_to_load = 'QLearning'
                    elif ac_button.collidepoint(event.pos): agent_to_load = 'ActorCritic'
                    elif human_button.collidepoint(event.pos):
                        game_state = 'HUMAN_PLAYING'
                        state, cumulative_reward, last_action_name = reset_episode(env)
                        last_action_name = "Your turn! Press 1, 2, or 3."
                        continue
                    if agent_to_load:
                        agent = AGENT_CLASSES[agent_to_load](action_space=env.action_space)
                        agent.load_policy(BEST_MODELS[agent_to_load])
                        if hasattr(agent, 'epsilon'): agent.epsilon = 0.0
                        game_state = 'AGENT_PLAYING'
                        state, cumulative_reward, last_action_name = reset_episode(env)
            elif game_state == 'HUMAN_PLAYING':
                if event.type == pygame.KEYDOWN:
                    action = None
                    if event.key == pygame.K_1: action = 0
                    elif event.key == pygame.K_2: action = 1
                    elif event.key == pygame.K_3: action = 2
                    elif event.key == pygame.K_ESCAPE: game_state = 'MENU'
                    if action is not None:
                        last_action_name = {0: "YOU chose Dagdag", 1: "YOU chose Lapad", 2: "YOU chose No Action"}[action]
                        next_state, reward, done, _ = env.step(action)
                        cumulative_reward += reward
                        state = next_state
                        if done:
                            time.sleep(2)
                            state, cumulative_reward, last_action_name = reset_episode(env)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                game_state = 'MENU'

        screen.fill(WHITE)
        if game_state == 'MENU':
            display_text(screen, font_lg, "Choose a Player", (SCREEN_WIDTH/2 - 150, 100))
            pygame.draw.rect(screen, BLUE, mc_button); display_text(screen, font_md, "Watch Best Monte Carlo Agent", (mc_button.centerx - 200, mc_button.centery - 15), WHITE)
            pygame.draw.rect(screen, BLUE, ql_button); display_text(screen, font_md, "Watch Best Q-Learning Agent", (ql_button.centerx - 200, ql_button.centery - 15), WHITE)
            pygame.draw.rect(screen, BLUE, ac_button); display_text(screen, font_md, "Watch Best Actor-Critic Agent", (ac_button.centerx - 200, ac_button.centery - 15), WHITE)
            pygame.draw.rect(screen, GREEN, human_button); display_text(screen, font_md, "Play as a Human", (human_button.centerx - 100, human_button.centery - 15), WHITE)
        elif game_state in ['AGENT_PLAYING', 'HUMAN_PLAYING']:
            if game_state == 'AGENT_PLAYING':
                current_time = pygame.time.get_ticks()
                if current_time - last_step_time > STEP_DELAY_MS:
                    last_step_time = current_time
                    action = agent.choose_action(state)
                    last_action_name = {0: "Agent chose Dagdag", 1: "Agent chose Lapad", 2: "Agent chose No Action"}[action]
                    next_state, reward, done, _ = env.step(action)
                    cumulative_reward += reward
                    state = next_state
                    # --- FIX: DRAW THE FINAL FRAME, PAUSE, THEN RESET ---
                    if done:
                        # 1. Draw the final state
                        screen.fill(WHITE)
                        draw_capacity_bar(screen, font_sm, env, state)
                        draw_environment(screen, font_sm, env.stations)
                        draw_train(screen, state, len(env.stations))
                        display_text(screen, font_md, "Episode finished! Resetting...", (50, 210), BLUE)
                        pygame.display.flip() # 2. Update the screen
                        time.sleep(2) # 3. Pause for the user to see
                        state, cumulative_reward, last_action_name = reset_episode(env) # 4. Then reset
            
            # This check is needed so the 'done' frame doesn't get instantly overwritten
            if state:
                draw_capacity_bar(screen, font_sm, env, state)
                draw_environment(screen, font_sm, env.stations)
                draw_train(screen, state, len(env.stations))
                station_name = env.stations[state[0]]
                direction = "Recto -> Antipolo" if state[3] == 0 else "Antipolo -> Recto"
                config_text = f"Carriages: {state[1]}, Width: {'Wide' if state[2] > 1 else 'Standard'}"
                display_text(screen, font_md, f"Current Station: {station_name}", (50, 150))
                display_text(screen, font_md, f"Direction: {direction}", (50, 180))
                display_text(screen, font_md, last_action_name, (50, 210), BLUE)
                display_text(screen, font_md, config_text, (50, 240))
                display_text(screen, font_sm, "Press ESC to return to menu", (SCREEN_WIDTH - 250, 20))
                display_text(screen, font_md, f"Score: {cumulative_reward:.2f}", (SCREEN_WIDTH - 250, 50), GREEN)
                if game_state == 'HUMAN_PLAYING':
                    display_text(screen, font_sm, "Controls: [1] Dagdag, [2] Lapad, [3] No Action", (50, 270), RED)

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    run_demo()