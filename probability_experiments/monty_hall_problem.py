# Monty hall problem
import numpy as np


class MontyHallProblem:
    def __init__(self, switch_doors=True):
        self.switch_doors = switch_doors

    @staticmethod
    def _initialize_problem():
        three_doors = np.zeros(shape=(1, 3))
        return three_doors

    @staticmethod
    def _place_goat(doors):
        goat_door = np.random.randint(0, 3)
        doors[:, goat_door] += 1
        return doors

    @staticmethod
    def _make_choice():
        our_choice = np.random.choice([0, 1, 2])
        return our_choice

    @staticmethod
    def _flatten(xss):
        return [x for xs in xss for x in xs]

    def play_game(self):
        # We place a goat in one of the doors
        doors = self._initialize_problem()
        doors = self._place_goat(doors)
        goat_position = np.argmax(doors)
        doors = self._flatten(doors)
        # Make player choice
        player_choice = self._make_choice()

        left_doors_for_open = np.array([i for i, _ in enumerate(doors) if i != player_choice and i != goat_position])

        # Here we need to remove one of the doors where there is no goat
        door_to_remove = np.random.choice(left_doors_for_open)
        left_doors = np.array([i for i, _ in enumerate(doors) if (i != door_to_remove)])
        if self.switch_doors:
            # Switch player choice doors if player wants to switch
            player_choice = left_doors[~(left_doors == player_choice)]

        if player_choice == goat_position:
            win = 1
        else:
            win = 0

        return win

    def play_many_games(self, num_games_to_play=10):
        game_num = 0
        results = []

        while game_num < num_games_to_play:
            win = self.play_game()
            results.append(win)
            game_num += 1

        return results


mhp = MontyHallProblem(switch_doors=True)

results = mhp.play_many_games(num_games_to_play=100000)

# Confirmed with experiment 66% approximately for switching
sum(results) / len(results)

# Without switching
mhp = MontyHallProblem(switch_doors=False)

results = mhp.play_many_games(num_games_to_play=100000)

sum(results) / len(results)
