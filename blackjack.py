import argparse
import os
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import time

# Comparing the sum of the cards
def cmp(a, b):
    if a > b:
        return 1
    else:
        return -1
    #return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


# Does this hand have a usable ace?
def usable_ace(hand):
    return int(1 in hand and sum(hand) + 10 <= 21)


# Return current hand total
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


# Is this hand a bust?
def is_bust(hand):
    return sum_hand(hand) > 21


# What is the score of this hand (0 if bust)
def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


# Is this hand a natural blackjack?
def is_natural(hand):
    return sorted(hand) == [1, 10]


# Modified blackjack environment to support two hands at a time
class BlackjackEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ## Description
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards. All cards are drawn from an infinite deck
    (i.e. with replacement).

    The card values are:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-10) have a value equal to their number.

    The player has the sum of cards held. The player can request
    additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
    immediate loss).

    After the player sticks, the dealer reveals their facedown card, and draws cards
    until their sum is 17 or greater. If the dealer goes bust, the player wins.

    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#blackjack_ref">1</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 1}` indicating
    whether to stick or hit.

    - 0: Stick
    - 1: Hit

    ## Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    The observation is returned as `(int(), int(), int())`.

    ## Starting State
    The starting state is initialised with the following values.

    | Observation               | Values         |
    |---------------------------|----------------|
    | Player current sum        |  4, 5, ..., 21 |
    | Dealer showing card value |  1, 2, ..., 10 |
    | Usable Ace                |  0, 1          |

    ## Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
    +1.5 (if <a href="#nat">natural</a> is True)
    +1 (if <a href="#nat">natural</a> is False)

    ## Episode End
    The episode ends if the following happens:

    - Termination:
    1. The player hits and the sum of hand exceeds 21.
    2. The player sticks.

    An ace will always be counted as usable (11) unless it busts the player.

    ## Information

    No additional information is returned.

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat"></a>`natural=False`: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab"></a>`sab=False`: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ## References
    <a id="blackjack_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    ## Version History
    * v1: Fix the natural handling in Blackjack
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2),
                spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2)
            )
        )

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.render_mode = render_mode
        self.terminated = [False, False]

    def step(self, actions):
        rewards = [0.0, 0.0]
        assert self.action_space.contains(actions[0])
        assert self.action_space.contains(actions[1])

        # Iterate through player actions
        for i, action in enumerate(actions):
            if not self.terminated[i]:
                if action:  # hit: add a card to players hand and return
                    self.players[i].append(draw_card(self.np_random))
                    if is_bust(self.players[i]):
                        self.terminated[i] = True
                        rewards[i] = -1.0
                # stick, yield to next player or dealer
                else:
                        self.terminated[i] = True

        #Dealer's turn
        if all(self.terminated):
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            self.reveal_dealer_card = True
            self.dealer_hidden_card_suit = self.np_random.choice(["C", "D", "H", "S"])
            hidden_value = self.dealer[1]
            self.dealer_hidden_card_value_str = (
                "A" if hidden_value == 1 else
                self.np_random.choice(["J", "Q", "K"]) if hidden_value == 10 else
                str(hidden_value)
                )
            self.dealer_total = score(self.dealer)


            for i in range(2):
                if not is_bust(self.players[i]):
                    rewards[i] = cmp(score(self.players[i]), score(self.dealer))
                    if self.sab and is_natural(self.players[i]) and not is_natural(self.dealer):
                        # Player automatically wins. Rules consistent with S&B
                        rewards[i] = 1.0
                    elif (
                        not self.sab
                        and self.natural
                        and is_natural(self.players[i])
                        and rewards[i] == 1.0
                    ):
                        # Natural gives extra points, but doesn't autowin. Legacy implementation
                        rewards[i] = 1.5

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), rewards, self.terminated, score(self.dealer), {}

    def _get_obs(self):
        return (
            sum_hand(self.players[0]), self.dealer[0], usable_ace(self.players[0]),
            sum_hand(self.players[1]), self.dealer[0], usable_ace(self.players[1])
        )

    # Reset function used to reset the environment after a hand
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # Reset class attributes
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.players = [draw_hand(self.np_random), draw_hand(self.np_random)]
        self.terminated = [False, False]

        player1_sum, dealer_card_value, player1_usable_ace, player2_sum, _, player2_usable_ace = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        import os
        import numpy as np

        # Displaying the modified pygame representation
        player_sum1, dealer_card_value, usable_ace1, player_sum2, dealer_hidden_value, usable_ace2 = self._get_obs()
        screen_width, screen_height = 800, 600
        card_img_height = screen_height // 4
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 30

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            return pygame.image.load(os.path.join(cwd, path))

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            return pygame.font.Font(os.path.join(cwd, path), size)

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        small_font = pygame.font.SysFont("Arial", screen_height // 20)
        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 8)

        dealer_text = small_font.render("Dealer:", True, white)
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        dealer_card_y = dealer_text_rect.bottom + spacing
        card_y = dealer_card_y
        card_x_center = screen_width // 2
        left_card_x = card_x_center - card_img_width - spacing // 2
        right_card_x = card_x_center + spacing // 2

        dealer_card_img = scale_card_img(get_image(
            os.path.join("images", f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png")
        ))
        self.screen.blit(dealer_card_img, (left_card_x, card_y))

        if getattr(self, "reveal_dealer_card", False):
            hidden_card_img = scale_card_img(get_image(
                os.path.join("images", f"{self.dealer_hidden_card_suit}{self.dealer_hidden_card_value_str}.png")
            ))
        else:
            hidden_card_img = scale_card_img(get_image(os.path.join("images", "Card.png")))

        self.screen.blit(hidden_card_img, (right_card_x, card_y))

        p1_text = small_font.render("Player 1", True, white)
        p1_text_rect = self.screen.blit(p1_text, (spacing, card_y + card_img_height + 2 * spacing))

        p1_sum_text = large_font.render(str(player_sum1), True, white)
        p1_sum_rect = self.screen.blit(
            p1_sum_text, (spacing, p1_text_rect.bottom + spacing)
        )

        if usable_ace1:
            ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                ace_text, (spacing, p1_sum_rect.bottom + spacing // 2)
            )

        p2_text = small_font.render("Player 2", True, white)
        p2_text_rect = self.screen.blit(
            p2_text, (screen_width - spacing - small_font.size("Player 2")[0], card_y + card_img_height + 2 * spacing)
        )

        p2_sum_text = large_font.render(str(player_sum2), True, white)
        p2_sum_rect = self.screen.blit(
            p2_sum_text,
            (
                screen_width - spacing - p2_sum_text.get_width(),
                p2_text_rect.bottom + spacing
            )
        )

        if usable_ace2:
            ace2_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                ace2_text, (
                    screen_width - spacing - ace2_text.get_width(),
                    p2_sum_rect.bottom + spacing // 2
                )
            )

        if getattr(self, "reveal_dealer_card", False):
            dealer_total = self.dealer_total
            dealer_sum_text = small_font.render(f"Dealer Total: {dealer_total}", True, white)
            self.screen.blit(dealer_sum_text, (card_x_center - dealer_sum_text.get_width() // 2, card_y + card_img_height + spacing))

            def get_result_text(player_sum):
                if player_sum > 21:
                    return "Lose"
                elif dealer_total > 21 or player_sum > dealer_total:
                    return "Win"
                elif player_sum == dealer_total:
                    return "Draw"
                else:
                    return "Lose"

            result1 = get_result_text(player_sum1)
            result2 = get_result_text(player_sum2)

            result1_text = small_font.render(result1, True, white)
            self.screen.blit(result1_text, (spacing, p1_sum_rect.bottom + 3 * spacing))

            result2_text = small_font.render(result2, True, white)
            self.screen.blit(result2_text, (
                screen_width - spacing - result2_text.get_width(),
                p2_sum_rect.bottom + 3 * spacing
            ))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
