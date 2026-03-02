'''
Strength-aware poker bot with conservative auctions, board texture adjustments,
opponent tendency tracking, and bounded Monte Carlo for close late-street spots.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import eval7


RANK_VALUE = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}


class Player(BaseBot):
    '''A pokerbot.'''

    def __init__(self) -> None:
        # Per-match lightweight opponent modeling (1000 rounds max).
        self.last_round_payoff = 0
        self.rounds_seen = 0
        self.opp_pressure_spots = 0
        self.opp_pressure_amount = 0
        self.opp_auction_cards_seen = 0
        self.opp_high_reveal_count = 0

        self._card_cache = {f'{r}{s}': eval7.Card(f'{r}{s}') for r in '23456789TJQKA' for s in 'cdhs'}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.last_round_payoff = 0
        self.rounds_seen = game_info.round_num

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.last_round_payoff = current_state.payoff
        for c in current_state.opp_revealed_cards:
            self.opp_auction_cards_seen += 1
            if RANK_VALUE[c[0]] >= 11:
                self.opp_high_reveal_count += 1

    def _preflop_strength(self, cards: list[str]) -> float:
        ranks = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
        high, low = ranks[0], ranks[1]
        suited = cards[0][1] == cards[1][1]
        gap = abs(high - low)

        score = (high + low) / 28.0
        if high == low:
            score += 0.55 + high / 40.0
        if suited:
            score += 0.08
        if gap <= 1:
            score += 0.08
        elif gap == 2:
            score += 0.04
        if high >= 13 and low >= 10:
            score += 0.10
        return min(score, 1.0)

    def _board_texture_adjustment(self, board: list[str], my_cards: list[str]) -> tuple[float, float]:
        '''Returns (equity_adjustment, aggression_adjustment).'''
        if len(board) < 3:
            return (0.0, 0.0)

        board_ranks = [RANK_VALUE[c[0]] for c in board]
        board_suits = [c[1] for c in board]

        rank_counts = {}
        for r in board_ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        max_pairing = max(rank_counts.values())

        suit_counts = {}
        for s in board_suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit = max(suit_counts.values())

        unique_sorted = sorted(set(board_ranks))
        if 14 in unique_sorted:
            unique_sorted = [1] + unique_sorted

        longest = 1
        run = 1
        for i in range(1, len(unique_sorted)):
            if unique_sorted[i] == unique_sorted[i - 1] + 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 1

        equity_adj = 0.0
        aggro_adj = 0.0

        # Wet boards: reduce one-pair confidence and reduce bluffing frequency.
        if max_suit >= 3:
            equity_adj -= 0.03
            aggro_adj -= 0.05
        if longest >= 3:
            equity_adj -= 0.03
            aggro_adj -= 0.04

        # Paired/trips board: top pair loses value, but nutted hands can still value-bet.
        if max_pairing >= 2:
            equity_adj -= 0.02
            aggro_adj -= 0.02

        # If our hole cards share dominant suit on a wet board, recover confidence slightly.
        if max_suit >= 3:
            dominant_suit = max(suit_counts, key=suit_counts.get)
            my_suit_hits = sum(1 for c in my_cards if c[1] == dominant_suit)
            if my_suit_hits == 2:
                equity_adj += 0.03

        return (equity_adj, aggro_adj)

    def _postflop_strength(self, my_cards: list[str], board: list[str]) -> float:
        cards = my_cards + board
        ranks = [RANK_VALUE[c[0]] for c in cards]

        counts = {}
        for rank in ranks:
            counts[rank] = counts.get(rank, 0) + 1
        multiples = sorted(counts.values(), reverse=True)

        suits = {}
        for card in cards:
            suits[card[1]] = suits.get(card[1], 0) + 1
        max_suit = max(suits.values())
        is_flush_draw = max_suit >= 4

        unique_ranks = sorted(set(ranks))
        if 14 in unique_ranks:
            unique_ranks = [1] + unique_ranks

        longest_run = 1
        run = 1
        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i - 1] + 1:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 1

        if longest_run >= 5:
            raw = 0.95
        elif multiples[0] == 4:
            raw = 0.93
        elif multiples[0] == 3 and len(multiples) > 1 and multiples[1] >= 2:
            raw = 0.90
        elif max_suit >= 5:
            raw = 0.88
        elif multiples[0] == 3:
            raw = 0.76
        elif len(multiples) > 1 and multiples[0] == 2 and multiples[1] == 2:
            raw = 0.70
        elif multiples[0] == 2:
            raw = 0.58
        elif longest_run == 4 and is_flush_draw:
            raw = 0.56
        elif longest_run == 4 or is_flush_draw:
            raw = 0.48
        else:
            hole_high = max(RANK_VALUE[c[0]] for c in my_cards)
            raw = 0.20 + hole_high / 70.0

        tex_eq, _ = self._board_texture_adjustment(board, my_cards)
        return max(0.05, min(0.98, raw + tex_eq))

    def _street_strength(self, current_state: PokerState) -> float:
        if current_state.street == 'pre-flop':
            return self._preflop_strength(current_state.my_hand)
        return self._postflop_strength(current_state.my_hand, current_state.board)

    def _opponent_pressure_factor(self) -> float:
        if self.opp_pressure_spots == 0:
            return 0.0
        avg_pressure = self.opp_pressure_amount / self.opp_pressure_spots
        # Convert chips to a small normalized factor.
        return min(0.08, avg_pressure / 1200.0)

    def _revealed_high_card_factor(self) -> float:
        if self.opp_auction_cards_seen == 0:
            return 0.0
        high_rate = self.opp_high_reveal_count / self.opp_auction_cards_seen
        # If opponent often reveals high cards in auctions, respect pressure slightly more.
        return min(0.05, max(0.0, (high_rate - 0.45) * 0.12))

    def _auction_bid(self, current_state: PokerState) -> int:
        strength = self._street_strength(current_state)
        pot = max(current_state.pot, 1)

        # Keep bids small to avoid auction bleed.
        base_cap = int(min(current_state.my_chips * 0.06, pot * 0.32 + 6))

        if strength < 0.45:
            target = int(base_cap * 0.75)
        elif strength < 0.70:
            target = int(base_cap * 1.00)
        else:
            target = int(base_cap * 0.65)

        # If opponent often shows high cards from auction reveals, pay a touch less.
        target = int(target * (1.0 - self._revealed_high_card_factor()))
        return max(0, min(target, current_state.my_chips))

    def _risk_mode(self, game_info: GameInfo) -> float:
        '''
        Returns a risk scalar in [0, 1].
        0 = default play, 1 = strong loss-avoid mode.
        '''
        rounds_left = max(1, 1000 - game_info.round_num)
        # If losing big late, protect stack and avoid high-variance spots.
        if game_info.bankroll < -900 and rounds_left < 450:
            return 1.0
        if game_info.bankroll < -600 and rounds_left < 650:
            return 0.7
        if game_info.bankroll < -350:
            return 0.4
        return 0.0

    def _max_safe_call(self, current_state: PokerState, risk_mode: float) -> int:
        '''Cap single-call exposure to reduce heavy-loss outliers.'''
        pot = max(current_state.pot, 1)
        stack = max(current_state.my_chips, 1)
        # strict when losing; looser when neutral/winning
        pot_cap = 0.55 - 0.22 * risk_mode
        stack_cap = 0.10 - 0.04 * risk_mode
        return int(min(pot * pot_cap, stack * stack_cap + 50))

    def _capped_raise(self, min_raise: int, max_raise: int, pot: int, chips: int, risk_mode: float, frac: float) -> int:
        '''Choose raise size with hard cap to avoid giant losses.'''
        target = min_raise + int((max_raise - min_raise) * frac)
        hard_cap = int(min(max_raise, pot * (0.75 - 0.30 * risk_mode), chips * (0.22 - 0.08 * risk_mode) + min_raise))
        amount = max(min_raise, min(target, hard_cap))
        return max(min_raise, min(amount, max_raise))

    def _monte_carlo_equity(self, current_state: PokerState, samples: int) -> float:
        '''Fast bounded equity estimate for turn/river only.''' 
        my_cards = current_state.my_hand
        board = current_state.board
        known = set(my_cards + board + current_state.opp_revealed_cards)
        if len(board) < 4:
            return -1.0

        my_eval_cards = [self._card_cache[c] for c in my_cards]
        board_eval_cards = [self._card_cache[c] for c in board]

        full_deck = [f'{r}{s}' for r in '23456789TJQKA' for s in 'cdhs']
        remaining = [c for c in full_deck if c not in known]

        if len(remaining) < 3:
            return -1.0

        wins = 0.0
        for _ in range(samples):
            draw = random.sample(remaining, 3)

            if current_state.opp_revealed_cards:
                opp = [self._card_cache[current_state.opp_revealed_cards[0]], self._card_cache[draw[0]]]
                idx = 1
            else:
                opp = [self._card_cache[draw[0]], self._card_cache[draw[1]]]
                idx = 2

            if len(board_eval_cards) == 4:
                community = board_eval_cards + [self._card_cache[draw[idx]]]
            else:
                community = board_eval_cards

            my_val = eval7.evaluate(community + my_eval_cards)
            opp_val = eval7.evaluate(community + opp)

            if my_val > opp_val:
                wins += 1.0
            elif my_val == opp_val:
                wins += 0.5

        return wins / samples

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        if current_state.street == 'auction':
            base_bid = self._auction_bid(current_state)
            # If we are in loss-avoid mode, trim auction exposure further.
            trim = 1.0 - 0.25 * self._risk_mode(game_info)
            return ActionBid(int(base_bid * trim))

        if current_state.cost_to_call > 0:
            self.opp_pressure_spots += 1
            self.opp_pressure_amount += current_state.cost_to_call

        strength = self._street_strength(current_state)
        pot = max(current_state.pot, 1)
        cost_to_call = max(current_state.cost_to_call, 0)
        pot_odds = cost_to_call / (pot + cost_to_call) if cost_to_call > 0 else 0.0
        risk_mode = self._risk_mode(game_info)

        _, tex_aggro = self._board_texture_adjustment(current_state.board, current_state.my_hand)
        pressure_penalty = self._opponent_pressure_factor() + self._revealed_high_card_factor()
        effective_strength = max(0.01, min(0.99, strength + tex_aggro - pressure_penalty * 0.6 - 0.05 * risk_mode))

        # Bounded Monte Carlo: only in close, expensive, late-street decisions with enough time bank.
        if (
            current_state.street in ('turn', 'river')
            and cost_to_call > 0
            and game_info.time_bank > 6.0
            and abs(effective_strength - pot_odds) < 0.10
            and cost_to_call > pot * 0.12
        ):
            mc_samples = 30 if game_info.time_bank < 12.0 else 55
            mc_equity = self._monte_carlo_equity(current_state, mc_samples)
            if mc_equity >= 0:
                effective_strength = 0.55 * effective_strength + 0.45 * mc_equity

        if current_state.can_act(ActionRaise):
            min_raise, max_raise = current_state.raise_bounds
            strong_threshold = 0.80 + max(0.0, -tex_aggro) * 0.3 + 0.08 * risk_mode
            medium_threshold = 0.68 + max(0.0, -tex_aggro) * 0.25 + 0.06 * risk_mode

            if effective_strength > strong_threshold:
                frac = 0.22 if current_state.street == 'river' else 0.30
                return ActionRaise(self._capped_raise(min_raise, max_raise, pot, current_state.my_chips, risk_mode, frac))

            if effective_strength > medium_threshold and current_state.street in ('flop', 'turn', 'river'):
                return ActionRaise(min_raise)

        if current_state.can_act(ActionCheck):
            return ActionCheck()

        if current_state.can_act(ActionCall):
            margin = (0.08 if current_state.street == 'pre-flop' else 0.05) - 0.02 * risk_mode
            max_safe = self._max_safe_call(current_state, risk_mode)
            # Emergency brake against very large single decision losses.
            if cost_to_call > max_safe and effective_strength < 0.86:
                return ActionFold()
            if effective_strength + margin >= pot_odds:
                return ActionCall()
            return ActionFold()

        return ActionFold()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
