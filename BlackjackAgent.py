from __future__ import annotations

from collections import defaultdict
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import gymnasium as gym


class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(2))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env,  obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[obs]))

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
            if type:
                target = reward + self.discount_factor * self.q_values[next_obs][next_act]
            else:
                target = reward + self.discount_factor * np.max(self.q_values[next_obs]) 
        else:
            target = reward
        self.q_values[obs][action] = (1-self.lr) * self.q_values[obs][action] + self.lr * target

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def print_q_values(self):
        for state, actions in self.q_values.items():
            if actions[0] >= actions[1]:
                print(f"State: {state}, Action: Stay")
            else:
                print(f"State: {state}, Action: Hit")

    def create_grids(agent, usable_ace=False):
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
        state_value = defaultdict(float)
        policy = defaultdict(int)
        for obs, action_values in agent.q_values.items():
            state_value[obs] = float(np.max(action_values))
            policy[obs] = int(np.argmax(action_values))

        player_count, dealer_count = np.meshgrid(
            # players count, dealers face-up card
            np.arange(12, 22),
            np.arange(1, 11),
        )

        # create the value grid for plotting
        value = np.apply_along_axis(
            lambda obs: state_value[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        value_grid = player_count, dealer_count, value

        # create the policy grid for plotting
        policy_grid = np.apply_along_axis(
            lambda obs: policy[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        return value_grid, policy_grid


    def create_plots(value_grid, policy_grid, title: str):
        """Creates a plot using a value and policy grid."""
        # create a new figure with 2 subplots (left: state values, right: policy)
        player_count, dealer_count, value = value_grid
        fig = plt.figure(figsize=plt.figaspect(0.4))
        fig.suptitle(title, fontsize=16)

        # plot the state values
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

        # plot the policy
        fig.add_subplot(1, 2, 2)
        ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
        ax2.set_title(f"Policy: {title}")
        ax2.set_xlabel("Player sum")
        ax2.set_ylabel("Dealer showing")
        ax2.set_xticklabels(range(12, 22))
        ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

        # add a legend
        legend_elements = [
            Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
            Patch(facecolor="grey", edgecolor="black", label="Stick"),
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
        return fig