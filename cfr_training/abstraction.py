"""
abstraction.py
Shared hand/state abstraction for both offline CFR training and online bot.

Defines the information-set key format, hand bucketing, board texture,
pot/SPR bucketing, and bid/bet discretisation. Both cfr_train.py and
bot.py import from here so the abstraction is guaranteed identical.
"""

import itertools
import random
from collections import Counter

# ── Card primitives ───────────────────────────────────────────────────────────
ALL_RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS     = ['s','h','d','c']
RANK_VAL  = {r: i for i, r in enumerate(ALL_RANKS)}

def make_deck():
    return [r+s for r in ALL_RANKS for s in SUITS]

def rv(c):  return RANK_VAL[c[0]]
def suit(c): return c[1]

# ── Hand evaluator (lightweight, no external deps) ────────────────────────────
class HandRank:
    __slots__ = ('_t',)
    def __init__(self, cat, tb=()):  self._t = (cat,)+tuple(tb)
    def __gt__(self, o): return self._t > o._t
    def __ge__(self, o): return self._t >= o._t
    def __eq__(self, o): return self._t == o._t
    def __lt__(self, o): return self._t < o._t

_WORST = HandRank(0,(0,)*5)

def _eval5(cards5):
    vs = sorted([rv(c) for c in cards5], reverse=True)
    ss = [suit(c) for c in cards5]
    cnt  = Counter(vs)
    freq = sorted(cnt.values(), reverse=True)
    fl   = len(set(ss)) == 1
    def _st(rs):
        if len(set(rs))!=5: return False,0
        if rs[0]-rs[4]==4:  return True,rs[0]
        if rs==[12,3,2,1,0]: return True,3
        return False,0
    is_st,stop = _st(vs)
    tb = tuple(r for r,_ in sorted(cnt.items(),key=lambda x:(x[1],x[0]),reverse=True))
    if fl and is_st: return HandRank(9 if stop==12 else 8,(stop,))
    if freq[0]==4:   return HandRank(7,tb)
    if freq[0]==3 and freq[1]==2: return HandRank(6,tb)
    if fl:           return HandRank(5,tuple(vs))
    if is_st:        return HandRank(4,(stop,))
    if freq[0]==3:   return HandRank(3,tb)
    if freq[0]==2 and freq[1]==2: return HandRank(2,tb)
    if freq[0]==2:   return HandRank(1,tb)
    return HandRank(0,tuple(vs))

def evaluate_best(hole, board):
    cards = hole + board
    if len(cards) < 5: return _WORST
    best = _WORST
    for c5 in itertools.combinations(cards,5):
        r = _eval5(c5)
        if r > best: best = r
    return best

def equity_mc(hole, board, opp_known=None, n=200):
    opp_known = list(opp_known) if opp_known else []
    known = set(hole)|set(board)|set(opp_known)
    deck  = [c for c in make_deck() if c not in known]
    need  = 5-len(board)
    wins,total = 0.0,0
    for _ in range(n):
        if opp_known:
            av = [c for c in deck if c not in opp_known]
            if not av: continue
            opp = opp_known+[random.choice(av)]
        else:
            if len(deck)<2+need: continue
            opp = random.sample(deck,2)
        rem = [c for c in deck if c not in opp]
        if len(rem)<need: continue
        run = board+random.sample(rem,need)
        mr  = evaluate_best(hole,run)
        or_ = evaluate_best(opp,run)
        wins += 1.0 if mr>or_ else (0.5 if mr==or_ else 0.0)
        total += 1
    return wins/max(total,1)

# ── Bucketing functions ───────────────────────────────────────────────────────
N_HAND_BUCKETS   = 16  # Increased to 16 for elite precision
N_BOARD_BUCKETS  = 8   # Increased to 8 using bitmask logic
N_POT_BUCKETS    = 4   
N_SPR_BUCKETS    = 4   
N_CALL_BUCKETS   = 4   
N_RAISES_CLIP    = 3   
N_REVEALED_BUCKETS = 4 

def hand_bucket(equity: float) -> int:
    """Map equity [0,1] → bucket [0, N_HAND_BUCKETS-1]."""
    return min(int(equity * N_HAND_BUCKETS), N_HAND_BUCKETS-1)

def board_bucket(board: list) -> int:
    """
    Bitmask Scoring (0 to 7):
    Bit 0 (Value 1): Paired Board
    Bit 1 (Value 2): Flush Draw / Made Flush
    Bit 2 (Value 4): Straight Draw / Made Straight
    """
    if not board: return 0
    suits_b = [suit(c) for c in board]
    ranks_b = sorted([rv(c) for c in board], reverse=True)
    cnt_s   = Counter(suits_b)
    cnt_r   = Counter(ranks_b)

    # 1. Paired board
    is_paired = cnt_r.most_common(1)[0][1] >= 2

    # 2. Flush draw (or made flush)
    max_suit = max(cnt_s.values())
    is_fd = max_suit >= (2 if len(board)==3 else 3)

    # 3. Straight draw (window of 5 ranks contains 3+)
    uniq = sorted(set(ranks_b))
    is_sd = any(sum(1 for r in uniq if lo<=r<=lo+4) >= 3 for lo in range(0,9))

    score = (1 if is_paired else 0) + (2 if is_fd else 0) + (4 if is_sd else 0)
    return score

def pot_bucket(pot: int) -> int:
    if pot < 60:  return 0
    if pot < 200: return 1
    if pot < 500: return 2
    return 3

def spr_bucket(stack: int, pot: int) -> int:
    spr = stack / max(pot, 1)
    if spr < 1:  return 0
    if spr < 3:  return 1
    if spr < 8:  return 2
    return 3

def call_bucket(cost: int, pot: int) -> int:
    if cost <= 0: return 0
    frac = cost / max(pot, 1)
    if frac < 0.25: return 1
    if frac < 0.50: return 2
    return 3

def revealed_bucket(card_or_none) -> int:
    if card_or_none is None: return 0
    r = rv(card_or_none)
    if r <= 5:  return 1   # 2-7
    if r <= 8:  return 2   # 8-T
    return 3               # J-A

# ── Bid discretisation ────────────────────────────────────────────────────────
BID_FRACS   = [0.0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.20, 999.0]
BID_LABELS  = ['bid_0','bid_10p','bid_20p','bid_35p','bid_55p','bid_80p','bid_120p','bid_allin']
N_BID_BUCKETS = len(BID_LABELS)

def bid_label_to_amount(label: str, pot: int, chips: int) -> int:
    idx  = BID_LABELS.index(label)
    frac = BID_FRACS[idx]
    if frac >= 999: return chips
    return min(int(pot * frac), chips)

def amount_to_bid_label(amount: int, pot: int, chips: int) -> str:
    frac = amount / max(pot, 1)
    if amount >= chips: return 'bid_allin'
    for i in range(len(BID_FRACS)-2, -1, -1):
        if frac >= (BID_FRACS[i]+BID_FRACS[i+1])/2:
            return BID_LABELS[i]
    return BID_LABELS[0]

# ── Bet/action discretisation ─────────────────────────────────────────────────
BET_LABELS  = ['fold','check','call','bet_33','bet_55','bet_100','bet_allin']
BET_FRACS   = {'bet_33':0.33,'bet_55':0.55,'bet_100':1.00,'bet_allin':999.0}

def bet_label_to_amount(label: str, pot: int, min_r: int, max_r: int) -> int:
    frac = BET_FRACS.get(label, 0)
    if frac >= 999: return max_r
    return max(min_r, min(int(pot*frac), max_r))

def make_infoset_key(
    street:       str,
    hb:           int,   
    bb:           int,   
    pb:           int,   
    spb:          int,   
    pos:          int,   
    auction:      int,   
    rev_b:        int,   
    cb:           int,   
    raises:       int,   
) -> str:
    return f"{street}|{hb}|{bb}|{pb}|{spb}|{pos}|{auction}|{rev_b}|{cb}|{min(raises,N_RAISES_CLIP)}"