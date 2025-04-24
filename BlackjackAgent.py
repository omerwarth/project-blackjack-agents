from __future__ import annotations
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import gymnasium as gym

class BlackjackAgent:
    # Initializes the Blackjack agent with the given values
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(lambda: np.zeros(2))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    # Returns the optimal action for a given state according to the algorithm
    def get_action(self, env,  obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[obs]))

    # Updates the dictionary of states/values based on the type of algorithm 
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: int,
        next_obs: tuple[int, int, bool],
        next_act: int,
        type: bool,
    ):
        if next_act != 2.0:
            # This is the difference between SARSA and Q-learning
            if type:
                target = reward + self.discount_factor * self.q_values[next_obs][next_act]
            else:
                target = reward + self.discount_factor * np.max(self.q_values[next_obs]) 
        else:
            target = reward
        self.q_values[obs][action] = (1-self.lr) * self.q_values[obs][action] + self.lr * target

    # Decays the epsilon to have a greater chance to use the optimal policy
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # Creates the grids that show the optimal policy and value of each state
    def create_grids(agent, usable_ace=False):
        state_value = defaultdict(float)
        policy = defaultdict(int)
        for obs, action_values in agent.q_values.items():
            state_value[obs] = float(np.max(action_values))
            policy[obs] = int(np.argmax(action_values))

        player_count, dealer_count = np.meshgrid(
            np.arange(12, 22),
            np.arange(1, 11),
        )

        value = np.apply_along_axis(
            lambda obs: state_value[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        value_grid = player_count, dealer_count, value

        policy_grid = np.apply_along_axis(
            lambda obs: policy[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        return value_grid, policy_grid


    # This function actually plots the output of the previous function to visualize it
    def create_plots(value_grid, policy_grid, title: str):
        player_count, dealer_count, value = value_grid
        fig = plt.figure(figsize=plt.figaspect(0.4))
        fig.suptitle(title, fontsize=16)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(
            player_count,
            dealer_count,
            value,
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )
        plt.xticks(range(12, 22), range(12, 22))
        plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
        ax1.set_title(f"State values: {title}")
        ax1.set_xlabel("Player sum")
        ax1.set_ylabel("Dealer showing")
        ax1.zaxis.set_rotate_label(False)
        ax1.set_zlabel("Value", fontsize=14, rotation=90)
        ax1.view_init(20, 220)

        fig.add_subplot(1, 2, 2)
        ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
        ax2.set_title(f"Policy: {title}")
        ax2.set_xlabel("Player sum")
        ax2.set_ylabel("Dealer showing")
        ax2.set_xticklabels(range(12, 22))
        ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

        legend_elements = [
            Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
            Patch(facecolor="grey", edgecolor="black", label="Stick"),
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
        return fig
