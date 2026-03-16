"""
Sneak Peek Hold'em Bot  —  IIT Pokerbots 2026
Polarized EV-Maximizer Architecture (Patched)
"""

import base64
import pickle
import random
import itertools
import zlib
from collections import Counter, defaultdict

from pkbot.base import BaseBot
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot

STRATEGY_DATA = None  # <-- build_bot.py replaces this line
STRATEGY_FILE = "strategy.pkl"   

def _load_strategy() -> dict:
    if STRATEGY_DATA is not None:
        try:
            raw = zlib.decompress(base64.b64decode(STRATEGY_DATA))
            strat = pickle.loads(raw)
            print(f"[bot] Blueprint loaded from embedded data: {len(strat):,} infosets", flush=True)
            return strat
        except Exception as e:
            print(f"[bot] Embedded data decode failed: {e}", flush=True)
    import os
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "rb") as f:
                strat = pickle.load(f)
            print(f"[bot] Blueprint loaded from file: {len(strat):,} infosets", flush=True)
            return strat
        except Exception as e: pass
    print("[bot] No blueprint — running heuristic only", flush=True)
    return {}

BIG_BLIND      = 20
SMALL_BLIND    = 10
STARTING_STACK = 5000

MC_SAMPLES_BET       = 12
MC_SAMPLES_AUCTION   = 40
MC_SAMPLES_POSTAUCT  = 40   

THRESH_VALUE_BET = 0.60
THRESH_STRONG    = 0.70
THRESH_CALL      = 0.40

BET_FRAC_MONSTER = 1.50   # Overbet the pot!
BET_FRAC_STRONG  = 0.75
BET_FRAC_MEDIUM  = 0.50
BET_FRAC_SMALL   = 0.33

BID_STACK_CAP = 0.10    

ALL_RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS     = ['s','h','d','c']
RANK_VAL  = {r: i for i, r in enumerate(ALL_RANKS)}

_CHEN = {'A':10,'K':8,'Q':7,'J':6,'T':5,'9':4.5,'8':4,'7':3.5,'6':3,'5':2.5,'4':2,'3':1.5,'2':1}

def _make_deck(): return [r+s for r in ALL_RANKS for s in SUITS]

class HandRank:
    __slots__ = ('_t',)
    def __init__(self, cat, tb=()):  self._t = (cat,)+tuple(tb)
    def __gt__(self, o): return self._t > o._t
    def __ge__(self, o): return self._t >= o._t
    def __eq__(self, o): return self._t == o._t
    def __lt__(self, o): return self._t < o._t

_WORST = HandRank(0, (0,)*5)

def _rv(c):  return RANK_VAL[c[0]]
def _su(c):  return c[1]

def _eval5(cards5):
    vs   = sorted([_rv(c) for c in cards5], reverse=True)
    ss   = [_su(c) for c in cards5]
    cnt  = Counter(vs)
    freq = sorted(cnt.values(), reverse=True)
    fl   = len(set(ss)) == 1
    def _st(rs):
        if len(set(rs)) != 5: return False, 0
        if rs[0]-rs[4] == 4:  return True, rs[0]
        if rs == [12,3,2,1,0]: return True, 3
        return False, 0
    is_st, stop = _st(vs)
    tb = tuple(r for r,_ in sorted(cnt.items(), key=lambda x:(x[1],x[0]), reverse=True))
    if fl and is_st: return HandRank(9 if stop==12 else 8, (stop,))
    if freq[0]==4:                        return HandRank(7, tb)
    if freq[0]==3 and freq[1]==2:         return HandRank(6, tb)
    if fl:                                return HandRank(5, tuple(vs))
    if is_st:                             return HandRank(4, (stop,))
    if freq[0]==3:                        return HandRank(3, tb)
    if freq[0]==2 and freq[1]==2:         return HandRank(2, tb)
    if freq[0]==2:                        return HandRank(1, tb)
    return HandRank(0, tuple(vs))

def _evaluate_best(hole, board):
    cards = hole + board
    if len(cards) < 5: return _WORST
    best = _WORST
    for c5 in itertools.combinations(cards, 5):
        r = _eval5(c5)
        if r > best: best = r
    return best

def _equity_mc(hole, board, opp_known=None, n=12):
    opp_known = list(opp_known) if opp_known else []
    known = set(hole) | set(board) | set(opp_known)
    deck  = [c for c in _make_deck() if c not in known]
    need  = 5 - len(board)
    wins, total = 0.0, 0
    for _ in range(n):
        if opp_known:
            av = [c for c in deck if c not in opp_known]
            if not av: continue
            opp = opp_known + [random.choice(av)]
        else:
            if len(deck) < 2: continue
            opp = random.sample(deck, 2)
        rem = [c for c in deck if c not in opp]
        if len(rem) < need: continue
        run  = board + random.sample(rem, need)
        mr   = _evaluate_best(hole, run)
        or_  = _evaluate_best(opp,  run)
        wins += 1.0 if mr > or_ else (0.5 if mr == or_ else 0.0)
        total += 1
    return wins / max(total, 1)

def _chen(hole):
    if len(hole) != 2: return 0.0
    c1, c2 = hole
    r1, r2 = _rv(c1), _rv(c2)
    s1, s2 = _su(c1), _su(c2)
    if r1 < r2: r1, r2, s1, s2 = r2, r1, s2, s1
    sc = _CHEN.get(ALL_RANKS[r1], 0)
    if r1 == r2: return max(sc*2, 5)
    if s1 == s2: sc += 2
    gap = r1 - r2 - 1
    sc -= [0,1,2,4,5][min(gap,4)]
    if gap <= 2 and min(r1,r2) >= 2: sc += 1
    return sc

def _chen_equity(chen): return max(0.30, min(0.85, 0.30 + (chen+2)/22.0*0.55))

def _hand_bucket(eq): return min(int(eq*8), 7)

def _board_bucket(board):
    if not board: return 0
    ss    = [_su(c) for c in board]
    rs    = sorted([_rv(c) for c in board], reverse=True)
    cnt_s = Counter(ss)
    if Counter(rs).most_common(1)[0][1] >= 2: return 3
    fl = max(cnt_s.values()) >= (2 if len(board)==3 else 3)
    uniq = sorted(set(rs))
    st   = any(sum(1 for r in uniq if lo<=r<=lo+4)>=3 for lo in range(9))
    return min(int(fl)+int(st), 2)

def _pot_bucket(pot): return 0 if pot<60 else 1 if pot<200 else 2 if pot<500 else 3

def _spr_bucket(stack, pot):
    spr = stack / max(pot, 1)
    return 0 if spr<1 else 1 if spr<3 else 2 if spr<8 else 3

def _call_bucket(cost, pot):
    if cost <= 0: return 0
    f = cost / max(pot, 1)
    return 1 if f<0.25 else 2 if f<0.50 else 3

def _revealed_bucket(card):
    if card is None: return 0
    r = _rv(card)
    return 1 if r<=5 else 2 if r<=8 else 3

_STREET_MAP = {'preflop':'preflop', 'flop':'flop_bet', 'turn':'turn', 'river':'river'}

def _make_key(street, eq, board, pot, stack, pos, auction, rev_card, cost, raises):
    return f"{_STREET_MAP.get(street, street)}|{_hand_bucket(eq)}|{_board_bucket(board)}|{_pot_bucket(pot)}|{_spr_bucket(stack, pot)}|{pos}|{auction}|{_revealed_bucket(rev_card)}|{_call_bucket(cost, pot)}|{min(raises, 3)}"

def _sample_blueprint(strategy, key):
    dist = strategy.get(key)
    if not dist: return None
    actions = list(dist.keys())
    weights = [dist[a] for a in actions]
    return random.choices(actions, weights=weights)[0]

def _label_to_action(label, state, pot, opp_chips, ActionFold, ActionCall, ActionCheck, ActionRaise):
    if label == 'fold': return ActionFold()
    if label in ('check', 'call'):
        if label == 'check' and state.can_act(ActionCheck): return ActionCheck()
        if state.can_act(ActionCall): return ActionCall()
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()
    fracs = {'bet_33': 0.33, 'bet_55': 0.55, 'bet_100': 1.00, 'bet_allin': 999.0}
    frac  = fracs.get(label, 0.55)
    if not state.can_act(ActionRaise): return ActionCall() if state.can_act(ActionCall) else ActionCheck()
    min_r, max_r = state.raise_bounds
    max_r = min(max_r, opp_chips)
    if max_r < min_r: return ActionCall() if state.can_act(ActionCall) else ActionCheck()
    target = int(pot * frac) if frac < 999 else max_r
    return ActionRaise(max(min_r, min(target, max_r)))

class _OppModel:
    _N = 3
    _BID_L = [[0.50, 0.25, 0.05], [0.30, 0.30, 0.10], [0.15, 0.30, 0.30], [0.05, 0.15, 0.55]]
    _BET_L = [[0.45, 0.25, 0.05], [0.30, 0.30, 0.15], [0.15, 0.30, 0.35], [0.10, 0.15, 0.45]]

    def __init__(self):
        self._p  = [1.0/self._N]*self._N
        self.n   = 0

    def _update(self, table, bucket):
        L    = [table[bucket][t] for t in range(self._N)]
        unnorm = [self._p[t]*L[t] for t in range(self._N)]
        tot  = sum(unnorm)
        if tot > 0: self._p = [u/tot for u in unnorm]
        self.n += 1

    def observe_bid(self, amount, pot):
        f = amount / max(pot, 1)
        b = 0 if amount==0 else (1 if f<0.20 else (2 if f<0.60 else 3))
        self._update(self._BID_L, b)

    def observe_bet(self, amount, pot):
        f = amount / max(pot, 1)
        b = 0 if amount==0 else (1 if f<0.33 else (2 if f<0.75 else 3))
        self._update(self._BET_L, b)

    def _confidence(self):
        import math
        h = -sum(q*math.log(q+1e-12) for q in self._p)
        return max(0.0, 1.0 - h/math.log(self._N))

    def thresh_adjustments(self):
        if self.n < 10: return 0.0, 0.0
        p = self._p
        dv = -0.06*p[0] + 0.00*p[1] + 0.07*p[2]
        dc = +0.05*p[0] + 0.00*p[1] - 0.06*p[2]
        lam = min(self._confidence(), 0.80)
        return dv*lam, dc*lam

    def bid_multiplier(self):
        if self.n < 10: return 1.0
        p = self._p
        return 0.85*p[0] + 1.00*p[1] + 1.30*p[2]

class Player(BaseBot):
    def __init__(self):
        self._strategy = _load_strategy()
        self._equity_cache      = {}
        self._opp_revealed      = []    
        self._we_won_auction    = False
        self._our_card_exposed  = None  
        self._first_postauct    = True
        self._raises            = 0     
        self._prev_street       = None
        self._opp_model         = _OppModel()
        self._hands             = 0
        self._opp_auct_wins     = 0
        self._our_auct_wins     = 0

    def on_hand_start(self, game_info, state):
        self._equity_cache     = {}
        self._opp_revealed     = []
        self._we_won_auction   = False
        self._our_card_exposed = None
        self._first_postauct   = True
        self._raises           = 0
        self._prev_street      = state.street

    def on_hand_end(self, game_info, state):
        self._hands += 1
        self._opp_model.observe_bet(state.opp_wager, max(state.pot, 1))

    def get_move(self, game_info, state):
        if game_info.time_bank < 0.4: return self._instant(state)
        if state.street != self._prev_street:
            self._raises      = 0
            self._prev_street = state.street

        self._process_revealed(state)
        if state.street == 'auction': return self._auction_bid(state)

        equity = self._get_equity(state)
        action = self._blueprint(state, equity)
        if action is not None: return action
        return self._heuristic(state, equity)

    def _process_revealed(self, state):
        if not state.opp_revealed_cards: return
        card    = state.opp_revealed_cards[0]
        my_hand = set(state.my_hand)
        if card in my_hand:
            if self._our_card_exposed != card:   
                self._we_won_auction   = False
                self._our_card_exposed = card
                self._opp_revealed     = []
                self._opp_auct_wins   += 1
                self._opp_model.observe_bid(int(state.pot * 0.65), state.pot)
        else:
            if not self._we_won_auction:
                self._we_won_auction   = True
                self._opp_revealed     = [card]
                self._our_card_exposed = None
                self._equity_cache     = {}     
                self._our_auct_wins   += 1

    def _get_equity(self, state):
        key = (tuple(state.board), tuple(self._opp_revealed))
        if key in self._equity_cache: return self._equity_cache[key]
        hole  = list(state.my_hand)
        board = list(state.board)
        if not board: eq = _chen_equity(_chen(hole))
        else:
            n  = MC_SAMPLES_POSTAUCT if (self._we_won_auction and self._first_postauct) else MC_SAMPLES_BET
            self._first_postauct = False
            eq = _equity_mc(hole, board, opp_known=self._opp_revealed or None, n=n)
        self._equity_cache[key] = eq
        return eq

    def _auction_bid(self, state):
        hole  = list(state.my_hand)
        board = list(state.board)
        pot   = max(state.pot, 1)
        chips = state.my_chips
        eq = _equity_mc(hole, board, n=MC_SAMPLES_AUCTION)
        
        # POLARIZED BIDDING WITH NOISE
        if eq > 0.80:
            bid = int(pot * (eq - 0.5) * 2.0) 
        elif eq > 0.65:
            bid = int(pot * 0.25)
        else:
            # NOISE: 30% of the time, bid a random small amount to mask our weakness.
            # 70% of the time, bid 0 to avoid the auction tax.
            if random.random() < 0.30:
                bid = int(pot * random.uniform(0.05, 0.15))
            else:
                bid = 0 

        if bid > 0: bid = int(bid * self._opp_model.bid_multiplier())
        bid = min(bid, int(chips * BID_STACK_CAP))
        bid = min(bid, chips)
        return ActionBid(max(0, bid))

    def _blueprint(self, state, equity):
        if not self._strategy: return None
        pot  = max(state.pot, 1)
        pos  = 0 if not state.is_bb else 1
        rev  = self._opp_revealed[0] if self._opp_revealed else None
        auct = 1 if self._we_won_auction else (2 if self._our_card_exposed else 0)
        key   = _make_key(state.street, equity, list(state.board), pot, min(state.my_chips, state.opp_chips), pos, auct, rev, state.cost_to_call, self._raises)
        label = _sample_blueprint(self._strategy, key)
        if label is None: return None
        action = _label_to_action(label, state, pot, state.opp_chips, ActionFold, ActionCall, ActionCheck, ActionRaise)
        if isinstance(action, ActionRaise): self._raises += 1
        return action

    def _heuristic(self, state, equity):
        pot       = max(state.pot, 1)
        cost      = state.cost_to_call
        opp_chips = state.opp_chips
        street    = state.street
        pot_odds  = cost / (pot+cost) if cost > 0 else 0.0

        val_thresh  = THRESH_VALUE_BET
        call_thresh = THRESH_CALL

        # --- BLOATED POT OVERRIDE ---
        # If the pot is small/medium, use the Bayesian opponent model to exploit.
        # If the pot is massive (>1000), they aren't bluffing. Ignore the model and play tight.
        if pot < 1000:
            dv, dc = self._opp_model.thresh_adjustments()
            val_thresh  += dv
            call_thresh += dc
        else:
            val_thresh  += 0.15
            call_thresh += 0.10

        if self._we_won_auction and self._opp_revealed:
            rv = _rv(self._opp_revealed[0])
            if rv <= 4:     val_thresh -= 0.10;  call_thresh -= 0.08
            elif rv <= 7:   val_thresh -= 0.04;  call_thresh -= 0.03
            elif rv >= 11:  val_thresh += 0.10;  call_thresh += 0.08
        elif self._our_card_exposed is not None:
            tighten = 0.06
            if self._hands >= 20 and (self._opp_auct_wins / max(self._hands, 1)) > 0.40: tighten = 0.12   
            val_thresh  += tighten
            call_thresh += tighten * 0.6

        # EXTREME LOSS MINIMIZATION
        if pot > 400:
            danger_penalty = min(0.20, (pot - 400) / 3000.0)
            call_thresh += danger_penalty
            val_thresh  += danger_penalty
        if cost > pot * 0.8:
            call_thresh += 0.10

        def safe_call():
            if state.can_act(ActionCall):  return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        def safe_check():
            if state.can_act(ActionCheck): return ActionCheck()
            return safe_call()

        # --- CAP-AND-CALL SAFETY NET ---
        # Never raise more than twice per street unless we have the absolute nuts.
        can_raise_more = (self._raises < 2) or (equity > 0.95)

        def _raise(frac):
            if not can_raise_more: return safe_call() # Cap the raise war and just call
            if not state.can_act(ActionRaise): return None
            mn, mx = state.raise_bounds
            mx = min(mx, opp_chips)
            if mx < mn: return None
            
            # Track the raise internally to trigger the cap later
            action = ActionRaise(max(mn, min(int(pot*frac), mx)))
            self._raises += 1 
            return action

        if self._our_card_exposed is not None and cost > 0:
            if equity > pot_odds + 0.06 and equity >= call_thresh: return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        # Tier 1: Monster
        if equity >= 0.88:
            if street in ['turn', 'river']: r = _raise(BET_FRAC_MONSTER)
            else: r = _raise(BET_FRAC_STRONG)
            return r if r else safe_call()

        # Tier 2: Value
        if equity >= val_thresh:
            if cost == 0:
                frac = BET_FRAC_MEDIUM if equity >= THRESH_STRONG else BET_FRAC_SMALL
                r = _raise(frac)
                return r if r else safe_check()
            else:
                if equity >= THRESH_STRONG:
                    r = _raise(BET_FRAC_MEDIUM)
                    if r: return r
                return safe_call()

        # Tier 3: Marginal
        if call_thresh <= equity < val_thresh:
            if cost == 0:
                if street != 'river' and self._we_won_auction:
                    r = _raise(BET_FRAC_SMALL)
                    if r: return r
                return safe_check()
            if equity > pot_odds + 0.04: return safe_call()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        # Tier 4: Weak
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()

    def _instant(self, state):
        if state.street == 'auction': return ActionBid(0)
        if state.can_act(ActionCheck): return ActionCheck()
        if state.can_act(ActionCall) and state.cost_to_call <= 2*BIG_BLIND: return ActionCall()
        return ActionFold()

if __name__ == '__main__':
    run_bot(Player(), parse_args())