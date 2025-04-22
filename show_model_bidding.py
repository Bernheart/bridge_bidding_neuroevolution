import numpy as np

from batch import batch_from_file
import globals as g
from saving_population import load_population
from utils_functions import evaluation_fitness
from colorama import init, Fore

init(autoreset=True)  # Automatically resets styles after each print


def print_bridge_hand(hand_tuple):
    # Define suit symbols and card face mappings
    suits = [Fore.GREEN + '♣', Fore.YELLOW + '♦', Fore.RED + '♥', Fore.BLUE + '♠']
    card_faces = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suit_colors = {'♠': 'black', '♥': 'red', '♦': 'red', '♣': 'black'}

    # Split the tuple into North and South hands
    handN, handS = hand_tuple

    # Helper function to convert hand to suit format
    def hand_to_suit_representation(hand):
        suits_representation = {suits[3]: [], suits[2]: [], suits[1]: [], suits[0]: []}

        for idx, card in reversed(list(enumerate(hand))):
            if idx != len(hand) - 1 and card == 1:
                suit = suits[idx // 13]  # Divide into 13 cards per suit
                card_number = idx % 13
                suits_representation[suit].append(card_faces[card_number])

        result = []
        for suit, cards in suits_representation.items():
            if cards:
                result.append(f"{suit}: {Fore.RESET + ''.join(cards)}")
            else:
                # If there's no card for the suit, mention the void and color
                result.append(f"{suit}:")

        return '\n'.join(result)

    if handN[len(handN) - 1] == 1:
        print(Fore.LIGHTRED_EX + "Vulnerable")
    else:
        print(Fore.LIGHTGREEN_EX + "Not Vulnerable")
    # Print North and South hands in a nice format
    print(Fore.WHITE + "North Hand:")
    print(hand_to_suit_representation(handN))
    print(Fore.WHITE + "\nSouth Hand:")
    print(hand_to_suit_representation(handS))


def print_bidding(bidding_mask):
    suits = [Fore.GREEN + '♣     ', Fore.YELLOW + '♦     ',
             Fore.RED + '♥     ', Fore.BLUE + '♠     ', Fore.WHITE + 'NT    ', Fore.GREEN + 'PASS  ']
    end = ['', '\n']
    bidding_length = 0
    for idx, bid in enumerate(bidding_mask):
        if bid == 1:
            bid_str = ''
            if idx in [0, len(bidding_mask) - 1]:
                bid_str = suits[len(suits) - 1]
            else:
                level = (idx - 1) // g.SUITS + 1
                suit = suits[(idx - 1) % g.SUITS]
                bid_str = str(level) + suit
            print(bid_str, end=end[bidding_length % 2])
            bidding_length += 1
    print('\n')


def print_model_bidding(agent, batch, n=np.inf):
    bidding_masks = evaluation_fitness(agent, batch, for_show=True)
    # print(bidding_masks)
    for idx, bidding_mask in enumerate(bidding_masks):
        print_bridge_hand(batch.hands[idx])
        print_bidding(bidding_mask)
        if idx + 1 == n:
            break


if __name__ == '__main__':
    population = load_population()
    batch_to_test = batch_from_file(202)
    print_model_bidding(population[0], batch_to_test)
    # for aget in population:
    #     print_model_bidding(aget, batch_to_test)
