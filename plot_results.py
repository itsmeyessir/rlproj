import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast # To safely evaluate the string representation of the list

# Load the data
df = pd.read_csv('evaluation_results.csv')

# --- Agent Performance Comparison ---

# a. Average Configuration Cost (Bar Chart)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='agent', y='cost', hue='config')
plt.title('Average Configuration Cost per Episode')
plt.ylabel('Average Cost')
plt.xlabel('Agent')
plt.tight_layout()
plt.savefig('avg_config_cost.png')
# plt.show() # <-- We move this to the end

# b. Score Distribution & Performance Consistency (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='agent', y='reward', hue='config')
plt.title('Score Distribution and Consistency')
plt.ylabel('Cumulative Reward')
plt.xlabel('Agent')
plt.tight_layout()
plt.savefig('score_distribution.png')
# plt.show() # <-- We move this to the end

# --- Other Visualizations ---

# Efficiency: Score per Unit Cost
# Add a small number to cost to avoid division by zero if cost is 0
df['efficiency'] = df['reward'] / (df['cost'] + 1e-6) 
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='agent', y='efficiency', hue='config')
plt.title('Efficiency (Reward per Unit Cost)')
plt.ylabel('Efficiency Score')
plt.xlabel('Agent')
plt.tight_layout()
plt.savefig('efficiency.png')
# plt.show() # <-- We move this to the end

# Action Distribution Comparison (Count Plot)
# Safely convert the 'actions' string back to a list
df['actions'] = df['actions'].apply(ast.literal_eval)
df_actions = df.explode('actions')
df_actions['action_name'] = df_actions['actions'].map({0: 'Dagdag', 1: 'Lapad', 2: 'No Action'})

plt.figure(figsize=(12, 7))
sns.countplot(data=df_actions, x='agent', hue='action_name', order=['MonteCarlo', 'QLearning', 'ActorCritic'])
plt.title('Action Distribution Comparison Across All Episodes')
plt.ylabel('Total Action Count')
plt.xlabel('Agent')
plt.tight_layout()
plt.savefig('action_distribution.png')
# plt.show() # <-- We move this to the end

# --- Now, show all the plots at once ---
print("All plot images have been saved to your directory.")
plt.show()