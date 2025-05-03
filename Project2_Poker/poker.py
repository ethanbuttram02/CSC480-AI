import random
import math
import time
from enum import Enum, auto
from collections import defaultdict

# card representation
class Suit(Enum):
    SPADES = auto()
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __repr__(self):
        rank_symbols = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: '10', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K', Rank.ACE: 'A'
        }
        suit_symbols = {
            Suit.SPADES: '♠', Suit.HEARTS: '♥', Suit.DIAMONDS: '♦', Suit.CLUBS: '♣'
        }
        return f"{rank_symbols[self.rank]}{suit_symbols[self.suit]}"
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))

class HandRank(Enum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        self.cards = []
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(rank, suit))
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self):
        if not self.cards:
            raise ValueError("No cards left in the deck")
        return self.cards.pop()
    
    def draw_multiple(self, count):
        return [self.draw() for _ in range(count)]
    
    def remove_cards(self, cards):
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)
    
    def __len__(self):
        return len(self.cards)

class HandEvaluator:
    @staticmethod
    def evaluate_hand(cards):
        # requires 7 cards (2 hole + 5 community)
        if len(cards) != 7:
            raise ValueError("Hand evaluation requires exactly 7 cards")
        
        # check for straight flush (including royal flush)
        straight_flush = HandEvaluator._check_straight_flush(cards)
        if straight_flush:
            if straight_flush[0].rank == Rank.ACE:
                return (HandRank.ROYAL_FLUSH, straight_flush)
            return (HandRank.STRAIGHT_FLUSH, straight_flush)
        
        # check for four of a kind
        four_of_a_kind = HandEvaluator._check_four_of_a_kind(cards)
        if four_of_a_kind:
            return (HandRank.FOUR_OF_A_KIND, four_of_a_kind)
        
        # check for full house
        full_house = HandEvaluator._check_full_house(cards)
        if full_house:
            return (HandRank.FULL_HOUSE, full_house)
        
        # check for flush
        flush = HandEvaluator._check_flush(cards)
        if flush:
            return (HandRank.FLUSH, flush)
        
        # check for straight
        straight = HandEvaluator._check_straight(cards)
        if straight:
            return (HandRank.STRAIGHT, straight)
        
        # check for three of a kind
        three_of_a_kind = HandEvaluator._check_three_of_a_kind(cards)
        if three_of_a_kind:
            return (HandRank.THREE_OF_A_KIND, three_of_a_kind)
        
        # check for two pair
        two_pair = HandEvaluator._check_two_pair(cards)
        if two_pair:
            return (HandRank.TWO_PAIR, two_pair)
        
        # check for pair
        pair = HandEvaluator._check_pair(cards)
        if pair:
            return (HandRank.PAIR, pair)
        
        # high card
        return (HandRank.HIGH_CARD, HandEvaluator._get_high_cards(cards, 5))
    
    @staticmethod
    def _check_straight_flush(cards):
        # group cards by suit
        by_suit = defaultdict(list)
        for card in cards:
            by_suit[card.suit].append(card)
        
        # check each suit for a straight
        for suit, suited_cards in by_suit.items():
            if len(suited_cards) >= 5:
                straight = HandEvaluator._check_straight(suited_cards)
                if straight:
                    return straight
        return None
    
    @staticmethod
    def _check_four_of_a_kind(cards):
        # group cards by rank
        by_rank = defaultdict(list)
        for card in cards:
            by_rank[card.rank].append(card)
        
        # find four of a kind
        for rank, cards_of_rank in by_rank.items():
            if len(cards_of_rank) == 4:
                # add a high card kicker
                four_cards = cards_of_rank
                remaining = [card for card in cards if card.rank != rank]
                remaining.sort(key=lambda card: card.rank.value, reverse=True)
                return four_cards + [remaining[0]]
        return None
    
    @staticmethod
    def _check_full_house(cards):
        # group cards by rank
        by_rank = defaultdict(list)
        for card in cards:
            by_rank[card.rank].append(card)
        
        # find three of a kind and pair
        three_of_a_kind = None
        pair = None
        
        # first look for the highest three of a kind
        for rank, cards_of_rank in sorted(by_rank.items(), key=lambda x: x[0].value, reverse=True):
            if len(cards_of_rank) >= 3:
                three_of_a_kind = cards_of_rank[:3]
                break
        
        if three_of_a_kind:
            # then look for the highest pair different from the three of a kind
            for rank, cards_of_rank in sorted(by_rank.items(), key=lambda x: x[0].value, reverse=True):
                if rank != three_of_a_kind[0].rank and len(cards_of_rank) >= 2:
                    pair = cards_of_rank[:2]
                    break
            
            if pair:
                return three_of_a_kind + pair
        
        return None
    
    @staticmethod
    def _check_flush(cards):
        # group cards by suit
        by_suit = defaultdict(list)
        for card in cards:
            by_suit[card.suit].append(card)
        
        # find flush
        for suit, suited_cards in by_suit.items():
            if len(suited_cards) >= 5:
                # take the 5 highest cards of the flush suit
                suited_cards.sort(key=lambda card: card.rank.value, reverse=True)
                return suited_cards[:5]
        
        return None
    
    @staticmethod
    def _check_straight(cards):
        # get unique ranks in descending order
        ranks = sorted(set(card.rank for card in cards), key=lambda r: r.value, reverse=True)
        
        # handle ace as low card (a-5-4-3-2)
        if Rank.ACE in ranks:
            ranks_with_low_ace = ranks + [Rank.ACE if r == Rank.TWO else r for r in ranks]
            ranks_with_low_ace_values = [1 if r == Rank.ACE and i > len(ranks) else r.value for i, r in enumerate(ranks_with_low_ace)]
            
            # check for 5 consecutive ranks
            for i in range(len(ranks_with_low_ace_values) - 4):
                if ranks_with_low_ace_values[i] - ranks_with_low_ace_values[i+4] == 4:
                    # get cards for these 5 ranks
                    straight_ranks = [ranks_with_low_ace[i+j] for j in range(5)]
                    straight_cards = []
                    
                    for rank in straight_ranks:
                        # handle special case for low ace
                        if rank == Rank.ACE and i > len(ranks) - 5:
                            # add the ace card (there should be only one)
                            for card in cards:
                                if card.rank == Rank.ACE and card not in straight_cards:
                                    straight_cards.append(card)
                                    break
                        else:
                            # add the highest card of this rank that's not already in straight_cards
                            cards_of_rank = [card for card in cards if card.rank == rank and card not in straight_cards]
                            if cards_of_rank:
                                straight_cards.append(cards_of_rank[0])
                    
                    if len(straight_cards) == 5:
                        # sort the straight by rank value (high to low)
                        return sorted(straight_cards, key=lambda card: card.rank.value, reverse=True)
        
        # normal straight check (no ace-low special case)
        for i in range(len(ranks) - 4):
            if ranks[i].value - ranks[i+4].value == 4:
                # get cards for these 5 ranks
                straight_ranks = [ranks[i+j] for j in range(5)]
                straight_cards = []
                
                for rank in straight_ranks:
                    # add the highest card of this rank that's not already in straight_cards
                    cards_of_rank = [card for card in cards if card.rank == rank and card not in straight_cards]
                    if cards_of_rank:
                        straight_cards.append(cards_of_rank[0])
                
                if len(straight_cards) == 5:
                    # sort the straight by rank value (high to low)
                    return sorted(straight_cards, key=lambda card: card.rank.value, reverse=True)
        
        return None
    
    @staticmethod
    def _check_three_of_a_kind(cards):
        # group cards by rank
        by_rank = defaultdict(list)
        for card in cards:
            by_rank[card.rank].append(card)
        
        # find three of a kind
        three_of_a_kind = None
        for rank, cards_of_rank in sorted(by_rank.items(), key=lambda x: x[0].value, reverse=True):
            if len(cards_of_rank) >= 3:
                three_of_a_kind = cards_of_rank[:3]
                break
        
        if three_of_a_kind:
            # add two high card kickers
            remaining = [card for card in cards if card.rank != three_of_a_kind[0].rank]
            remaining.sort(key=lambda card: card.rank.value, reverse=True)
            return three_of_a_kind + remaining[:2]
        
        return None
    
    @staticmethod
    def _check_two_pair(cards):
        # group cards by rank
        by_rank = defaultdict(list)
        for card in cards:
            by_rank[card.rank].append(card)
        
        # find pairs
        pairs = []
        for rank, cards_of_rank in sorted(by_rank.items(), key=lambda x: x[0].value, reverse=True):
            if len(cards_of_rank) >= 2:
                pairs.append(cards_of_rank[:2])
                if len(pairs) == 2:
                    break
        
        if len(pairs) == 2:
            # add a high card kicker
            paired_ranks = [pairs[0][0].rank, pairs[1][0].rank]
            remaining = [card for card in cards if card.rank not in paired_ranks]
            remaining.sort(key=lambda card: card.rank.value, reverse=True)
            return pairs[0] + pairs[1] + [remaining[0]]
        
        return None
    
    @staticmethod
    def _check_pair(cards):
        # group cards by rank
        by_rank = defaultdict(list)
        for card in cards:
            by_rank[card.rank].append(card)
        
        # find pair
        pair = None
        for rank, cards_of_rank in sorted(by_rank.items(), key=lambda x: x[0].value, reverse=True):
            if len(cards_of_rank) >= 2:
                pair = cards_of_rank[:2]
                break
        
        if pair:
            # add three high card kickers
            remaining = [card for card in cards if card.rank != pair[0].rank]
            remaining.sort(key=lambda card: card.rank.value, reverse=True)
            return pair + remaining[:3]
        
        return None
    
    @staticmethod
    def _get_high_cards(cards, count=5):
        sorted_cards = sorted(cards, key=lambda card: card.rank.value, reverse=True)
        return sorted_cards[:count]
    
    @staticmethod
    def compare_hands(hand1, hand2):
        # compare two hands, return 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        rank1, cards1 = hand1
        rank2, cards2 = hand2
        
        # compare hand ranks first
        if rank1.value > rank2.value:
            return 1
        elif rank1.value < rank2.value:
            return -1
        
        # if hand ranks are the same, compare the card values
        for card1, card2 in zip(cards1, cards2):
            if card1.rank.value > card2.rank.value:
                return 1
            elif card1.rank.value < card2.rank.value:
                return -1
        
        # if all card values are the same, it's a tie
        return 0

class MCTSNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        
    def add_child(self, child):
        self.children.append(child)
        return child
    
    def update(self, result):
        self.visits += 1
        self.wins += result
    
    def ucb1(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    @property
    def win_probability(self):
        if self.visits == 0:
            return 0
        return self.wins / self.visits

class PokerBot:
    def __init__(self, time_limit=10, exploration_weight=1.41):
        self.time_limit = time_limit
        self.exploration_weight = exploration_weight
        
    def make_decision(self, hole_cards, community_cards, phase):
        # decide whether to fold or stay based on monte carlo simulations
        print(f"Decision at {phase}:")
        print(f"My hole cards: {hole_cards}")
        print(f"Community cards: {community_cards}")
        
        # instead of deep mcts tree, use flat monte carlo simulation for poker
        # this avoids recursion issues and is more appropriate for this problem
        
        # initialize the deck and remove known cards
        deck = Deck()
        deck.remove_cards(hole_cards + community_cards)
        
        # track metrics
        simulations = 0
        wins = 0
        ties = 0
        start_time = time.time()
        
        # run simulations until time limit is reached
        while time.time() - start_time < self.time_limit:
            # run a single simulation
            result = self._run_simulation(hole_cards, community_cards, deck.cards.copy())
            simulations += 1
            
            if result == 1:
                wins += 1
            elif result == 0.5:
                ties += 0.5  # count ties as half-wins
        
        # calculate the win probability
        win_probability = (wins + ties) / simulations if simulations > 0 else 0
        
        # make decision based on win probability
        decision = "stay" if win_probability >= 0.5 else "fold"
        
        # print results
        elapsed_time = time.time() - start_time
        print(f"Simulations: {simulations} in {elapsed_time:.2f} seconds")
        print(f"Win probability: {win_probability:.4f}")
        print(f"Decision: {decision}")
        
        return decision, win_probability
    
    def _simulate(self, node, hole_cards, community_cards, available_cards):
        # run a single mcts simulation - iterative version to avoid recursion depth issues
        current = node
        
        # selection phase - find the most promising leaf node
        while current.children:
            current = max(current.children, key=lambda c: c.ucb1(self.exploration_weight))
        
        # expansion and simulation phase
        if current.visits > 0:
            # create a new child node
            child = current.add_child(MCTSNode(parent=current))
            current = child
        
        # run simulation
        result = self._run_simulation(hole_cards, community_cards, available_cards)
        
        # backpropagate result iteratively instead of recursively
        while current:
            current.update(result)
            current = current.parent
        
        return result
    
    def _run_simulation(self, hole_cards, community_cards, available_cards):
        # simulate a random game and return 1 if my hand wins, 0.5 if tie, 0 otherwise
        
        # shuffle available cards
        random.shuffle(available_cards)
        
        # create a copy of available cards to draw from
        draw_pile = available_cards.copy()
        
        # draw opponent's hole cards
        opponent_hole_cards = []
        for _ in range(2):
            card = draw_pile.pop()
            opponent_hole_cards.append(card)
        
        # draw remaining community cards
        additional_community_cards = []
        for _ in range(5 - len(community_cards)):
            card = draw_pile.pop()
            additional_community_cards.append(card)
        
        # combine all community cards
        all_community_cards = community_cards + additional_community_cards
        
        # evaluate both hands
        my_hand = HandEvaluator.evaluate_hand(hole_cards + all_community_cards)
        opponent_hand = HandEvaluator.evaluate_hand(opponent_hole_cards + all_community_cards)
        
        # compare hands to determine the winner
        comparison = HandEvaluator.compare_hands(my_hand, opponent_hand)
        
        # return 1 if my hand wins, 0.5 if tie, 0 if opponent wins
        if comparison > 0:
            return 1
        elif comparison == 0:
            return 0.5
        else:
            return 0
    
    def _backpropagate(self, node, result):
        # backpropagate the result up the tree - iterative version to avoid recursion depth issues
        current = node
        while current:
            current.update(result)
            current = current.parent

def main():
    bot = PokerBot()
    
    # example usage
    print("\n----- example 1: pocket aces -----")
    # pre-flop (no community cards yet)
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]
    community_cards = []
    decision, probability = bot.make_decision(hole_cards, community_cards, "Pre-Flop")
    
    print("\n----- example 2: pocket kings -----")
    # try another hand
    hole_cards = [Card(Rank.KING, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    community_cards = []
    decision, probability = bot.make_decision(hole_cards, community_cards, "Pre-Flop")
    
    print("\n----- example 3: potential straight -----")
    # try a hand with potential
    hole_cards = [Card(Rank.QUEEN, Suit.SPADES), Card(Rank.JACK, Suit.HEARTS)]
    community_cards = [Card(Rank.KING, Suit.DIAMONDS), Card(Rank.TEN, Suit.CLUBS), Card(Rank.TWO, Suit.SPADES)]
    decision, probability = bot.make_decision(hole_cards, community_cards, "Flop")
    
    print("\n----- example 4: weak hand -----")
    # try a weak hand
    hole_cards = [Card(Rank.TWO, Suit.SPADES), Card(Rank.SEVEN, Suit.HEARTS)]
    community_cards = [Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.KING, Suit.CLUBS), Card(Rank.QUEEN, Suit.SPADES)]
    decision, probability = bot.make_decision(hole_cards, community_cards, "Flop")

if __name__ == "__main__":
    main()