"""
Sneak Peek Hold'em Bot
IIT Pokerbots 2026
Single-file submission (Evaluator integrated)
"""

import random
import itertools
from collections import Counter

from pkbot.base import BaseBot
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot


# ============================================================
# Constants
# ============================================================

BIG_BLIND = 20
SMALL_BLIND = 10

MC_SAMPLES_BET = 12
MC_SAMPLES_AUCTION = 40

THRESH_VALUE_BET = 0.60
THRESH_CALL = 0.40

BET_FRAC_STRONG = 0.75
BET_FRAC_MEDIUM = 0.50
BET_FRAC_SMALL = 0.33

AUCTION_BASE_FRAC = 0.30


# ============================================================
# Lightweight Hand Evaluator (Integrated)
# ============================================================

ALL_RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS     = ['s','h','d','c']
RANK_VAL  = {r: i for i, r in enumerate(ALL_RANKS)}

_CHEN = {'A':10,'K':8,'Q':7,'J':6,'T':5,'9':4.5,'8':4,
         '7':3.5,'6':3,'5':2.5,'4':2,'3':1.5,'2':1}

def make_deck():
    return [r+s for r in ALL_RANKS for s in SUITS]

class HandRank:
    __slots__ = ('_tup',)
    def __init__(self, cat, tb=()):
        self._tup = (cat,) + tuple(tb)
    def __gt__(self, o): return self._tup > o._tup
    def __eq__(self, o): return self._tup == o._tup

def _rv(c): return RANK_VAL[c[0]]
def _su(c): return c[1]

def _eval5(cards5):
    rv = sorted([_rv(c) for c in cards5], reverse=True)
    su = [_su(c) for c in cards5]
    cnt = Counter(rv)
    freq = sorted(cnt.values(), reverse=True)
    is_flush = len(set(su)) == 1

    def straight(rs):
        if len(set(rs)) != 5: return False, 0
        if rs[0]-rs[4] == 4: return True, rs[0]
        if rs == [12,3,2,1,0]: return True, 3
        return False, 0

    is_st, st_top = straight(rv)
    tb = tuple(r for r,_ in sorted(cnt.items(), key=lambda x:(x[1],x[0]), reverse=True))

    if is_flush and is_st: return HandRank(8, (st_top,))
    if freq[0]==4: return HandRank(7, tb)
    if freq[0]==3 and freq[1]==2: return HandRank(6, tb)
    if is_flush: return HandRank(5, tuple(rv))
    if is_st: return HandRank(4, (st_top,))
    if freq[0]==3: return HandRank(3, tb)
    if freq[0]==2 and freq[1]==2: return HandRank(2, tb)
    if freq[0]==2: return HandRank(1, tb)
    return HandRank(0, tuple(rv))

def evaluate_best(hole, board):
    all_c = hole + board
    if len(all_c) < 5:
        return HandRank(0, (0,))
    best = None
    for combo in itertools.combinations(all_c, 5):
        r = _eval5(combo)
        if not best or r > best:
            best = r
    return best

def equity_mc_fast(hole, board, opp_known=None, n_samples=12):
    opp_known = list(opp_known) if opp_known else []
    known = set(hole) | set(board) | set(opp_known)
    deck = [c for c in make_deck() if c not in known]
    need = 5 - len(board)

    wins = 0.0
    total = 0

    for _ in range(n_samples):
        if opp_known:
            avail = [c for c in deck if c not in opp_known]
            if not avail: continue
            opp = opp_known + [random.choice(avail)]
        else:
            if len(deck) < 2: continue
            opp = random.sample(deck, 2)

        rem = [c for c in deck if c not in opp]
        if len(rem) < need: continue

        run = board + random.sample(rem, need)
        mr = evaluate_best(hole, run)
        or_ = evaluate_best(opp, run)

        if mr > or_: wins += 1
        elif mr == or_: wins += 0.5
        total += 1

    return wins / max(total, 1)

def chen_score(hole):
    if len(hole) != 2: return 0.0
    c1,c2 = hole
    r1,r2 = _rv(c1),_rv(c2)
    s1,s2 = _su(c1),_su(c2)
    if r1 < r2: r1,r2,s1,s2 = r2,r1,s2,s1
    sc = _CHEN[ALL_RANKS[r1]]
    if r1 == r2: return max(sc*2, 5)
    if s1 == s2: sc += 2
    gap = r1 - r2 - 1
    sc -= [0,1,2,4,5][min(gap,4)]
    if gap <= 2 and min(r1,r2) >= 2: sc += 1
    return sc

def chen_to_equity(chen):
    return max(0.30, min(0.85, 0.30 + (chen + 2) / 22.0 * 0.55))


# ============================================================
# Player Bot
# ============================================================

class Player(BaseBot):

    def __init__(self):
        self._equity_cache = {}
        self._opp_revealed = []

    def on_hand_start(self, game_info, state):
        self._equity_cache = {}
        self._opp_revealed = []

    def on_hand_end(self, game_info, state):
        pass

    def get_move(self, game_info, state):

        if game_info.time_bank < 0.5:
            return self._instant_action(state)

        if state.opp_revealed_cards:
            self._opp_revealed = list(state.opp_revealed_cards)

        if state.street == 'auction':
            return self._auction_bid(state)

        equity = self._get_equity(state)
        return self._betting_action(state, equity)

    # --------------------------------------------------------

    def _get_equity(self, state):
        key = (tuple(state.board), tuple(self._opp_revealed))
        if key in self._equity_cache:
            return self._equity_cache[key]

        hole = list(state.my_hand)
        board = list(state.board)

        if not board:
            eq = chen_to_equity(chen_score(hole))
        else:
            eq = equity_mc_fast(
                hole,
                board,
                opp_known=self._opp_revealed if self._opp_revealed else None,
                n_samples=MC_SAMPLES_BET
            )

        self._equity_cache[key] = eq
        return eq

    # --------------------------------------------------------

    def _auction_bid(self, state):

        hole = list(state.my_hand)
        board = list(state.board)
        pot = state.pot
        chips = state.my_chips

        equity = equity_mc_fast(
            hole,
            board,
            n_samples=MC_SAMPLES_AUCTION
        )

        uncertainty = 4.0 * equity * (1.0 - equity)
        protection = max(0.0, equity - 0.55) * 2.0
        combined = min(1.0, uncertainty + protection * 0.4)

        bid = int(combined * pot * AUCTION_BASE_FRAC)
        bid = max(0, min(bid, chips))
        return ActionBid(bid)

    # --------------------------------------------------------

    def _betting_action(self, state, equity):

        pot = max(state.pot, 1)
        cost = state.cost_to_call
        pot_odds = cost / (pot + cost) if cost > 0 else 0.0

        def safe_call():
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()

        def raise_frac(frac):
            if not state.can_act(ActionRaise):
                return None
            min_r, max_r = state.raise_bounds
            target = int(pot * frac)
            amount = max(min_r, min(target, max_r))
            if amount < min_r:
                return None
            return ActionRaise(amount)

        if equity >= 0.82:
            r = raise_frac(BET_FRAC_STRONG)
            return r if r else safe_call()

        if equity >= THRESH_VALUE_BET:
            if cost == 0:
                r = raise_frac(BET_FRAC_MEDIUM)
                return r if r else safe_call()
            return safe_call()

        if THRESH_CALL <= equity < THRESH_VALUE_BET and cost == 0 and state.street != 'river':
            r = raise_frac(BET_FRAC_SMALL)
            return r if r else safe_call()

        if equity > pot_odds + 0.04 and equity >= THRESH_CALL:
            return safe_call()

        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    # --------------------------------------------------------

    def _instant_action(self, state):
        if state.street == 'auction':
            return ActionBid(0)
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall) and state.cost_to_call <= 2 * BIG_BLIND:
            return ActionCall()
        return ActionFold()


# ============================================================

if __name__ == '__main__':
    run_bot(Player(), parse_args())