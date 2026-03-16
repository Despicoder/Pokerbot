"""
Sneak Peek Hold'em Bot  —  IIT Pokerbots 2026
==============================================
Single-file submission.  DO NOT EDIT — edit bot_template.py then run build pipeline.

    python cfr_train.py   --iters 300000 --out strategy.pkl
    python build_bot.py   --strategy strategy.pkl --template bot_template.py --out bot.py

Design philosophy (v6 — CHIP MAXIMIZATION for tournament ranking):
  Tournament ranking = total chips won, not win rate.
  A 50% WR bot that sizes bets correctly beats a 52% WR bot that sizes poorly.

  Core insight (from EV math):
    vs calling station (folds <10%): overbet 150-200% pot, EV = +186 vs +94 at 50%
    vs passive folder  (folds >55%): exploit with 100-150% pot
    vs balanced Nash   (folds ~35%): pot-sized bets
    vs aggressive LAG  (folds ~20%): trap with check-raise, then overbet

  Per-decision flow:
    T0  Emergency     — instant action when time_bank < 0.4 s
    T1  Blueprint     — DCFR Nash mixed strategy (Nash action selection)
    T2  Exploit size  — override raise AMOUNT with opponent-model sizing
    T3  Heuristic     — threshold rules as fallback

  Key fixes vs earlier versions:
    • opp_revealed_cards disambiguation  (engine puts OUR card there when we lose)
    • No re-raise after losing auction   (caused -89K chip wars in log)
    • SPR-aware auction bidding
    • Bayesian model over 4 opponent types → dynamic bet sizing
"""

import base64
import itertools
import pickle
import random
import zlib
from collections import Counter

from pkbot.base import BaseBot
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot


# ============================================================
# Embedded strategy (injected by build_bot.py at build time)
# ============================================================

STRATEGY_DATA = None  # <-- build_bot.py replaces this line

STRATEGY_FILE = "strategy.pkl"


def _load_strategy() -> dict:
    if STRATEGY_DATA is not None:
        try:
            raw = zlib.decompress(base64.b64decode(STRATEGY_DATA))
            s   = pickle.loads(raw)
            print(f"[bot] Blueprint: {len(s):,} infosets (embedded)", flush=True)
            return s
        except Exception as e:
            print(f"[bot] Embedded decode failed: {e}", flush=True)
    import os
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "rb") as f:
                s = pickle.load(f)
            print(f"[bot] Blueprint: {len(s):,} infosets (file)", flush=True)
            return s
        except Exception as e:
            print(f"[bot] File load failed: {e}", flush=True)
    print("[bot] No blueprint — heuristic only", flush=True)
    return {}


# ============================================================
# Constants
# ============================================================

BIG_BLIND      = 20
SMALL_BLIND    = 10
STARTING_STACK = 5000

MC_SAMPLES_BET      = 12
MC_SAMPLES_AUCTION  = 40
MC_SAMPLES_POSTAUCT = 40

THRESH_BET    = 0.55   # lowered from 0.60: value-bet more hands for chip EV
THRESH_STRONG = 0.68
THRESH_CALL   = 0.38

VOI_SCALE     = 0.35
SPR_CAP       = 3.0
BID_MIN_FRAC  = 0.05
BID_STACK_CAP = 0.10

# Stack-commit: shove when strong hand + low SPR
STACK_COMMIT_EQ  = 0.72
STACK_COMMIT_SPR = 2.5


# ============================================================
# Hand evaluator (zero external deps)
# ============================================================

ALL_RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS     = ['s','h','d','c']
RANK_VAL  = {r: i for i, r in enumerate(ALL_RANKS)}

_CHEN = {'A':10,'K':8,'Q':7,'J':6,'T':5,'9':4.5,'8':4,
         '7':3.5,'6':3,'5':2.5,'4':2,'3':1.5,'2':1}

def _make_deck():
    return [r+s for r in ALL_RANKS for s in SUITS]

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

def _chen_equity(chen):
    return max(0.30, min(0.85, 0.30 + (chen+2)/22.0*0.55))


# ============================================================
# Blueprint infoset key  (must match cfr_train.py exactly)
# ============================================================

def _hand_bucket(eq):  return min(int(eq*8), 7)

def _board_bucket(board):
    if not board: return 0
    ss    = [_su(c) for c in board]
    rs    = sorted([_rv(c) for c in board], reverse=True)
    cnt_s = Counter(ss)
    if Counter(rs).most_common(1)[0][1] >= 2: return 3
    fl   = max(cnt_s.values()) >= (2 if len(board)==3 else 3)
    uniq = sorted(set(rs))
    st   = any(sum(1 for r in uniq if lo<=r<=lo+4)>=3 for lo in range(9))
    return min(int(fl)+int(st), 2)

def _pot_bucket(pot):
    return 0 if pot<60 else 1 if pot<200 else 2 if pot<500 else 3

def _spr_bucket(stack, pot):
    spr = stack / max(pot, 1)
    return 0 if spr<1 else 1 if spr<3 else 2 if spr<8 else 3

def _call_bucket(cost, pot):
    if cost <= 0: return 0
    f = cost / max(pot, 1)
    return 1 if f<0.25 else 2 if f<0.50 else 3

def _rev_bucket(card):
    if card is None: return 0
    r = _rv(card)
    return 1 if r<=5 else 2 if r<=8 else 3

_STREET_MAP = {'preflop':'preflop','flop':'flop_bet','turn':'turn','river':'river'}

def _make_key(street, eq, board, pot, stack, pos, auction, rev_card, cost, raises):
    return (
        f"{_STREET_MAP.get(street,street)}"
        f"|{_hand_bucket(eq)}|{_board_bucket(board)}"
        f"|{_pot_bucket(pot)}|{_spr_bucket(stack,pot)}"
        f"|{pos}|{auction}|{_rev_bucket(rev_card)}"
        f"|{_call_bucket(cost,pot)}|{min(raises,3)}"
    )

def _sample_blueprint(strategy, key):
    dist = strategy.get(key)
    if not dist: return None
    acts = list(dist.keys())
    return random.choices(acts, weights=[dist[a] for a in acts])[0]

def _label_to_action(label, state, pot, opp_chips):
    """Blueprint label → engine Action (uses Nash sizing, not exploit sizing)."""
    if label == 'fold': return ActionFold()
    if label in ('check','call'):
        if label=='check' and state.can_act(ActionCheck): return ActionCheck()
        if state.can_act(ActionCall): return ActionCall()
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()
    fracs = {'bet_33':0.33,'bet_55':0.55,'bet_100':1.00,'bet_allin':999.0}
    frac  = fracs.get(label, 0.55)
    if not state.can_act(ActionRaise):
        return ActionCall() if state.can_act(ActionCall) else ActionCheck()
    mn, mx = state.raise_bounds
    mx = min(mx, opp_chips)
    if mx < mn: return ActionCall() if state.can_act(ActionCall) else ActionCheck()
    target = int(pot*frac) if frac<999 else mx
    return ActionRaise(max(mn, min(target, mx)))


# ============================================================
# Opponent Model  — chip-EV aware, 4 archetypes
# ============================================================

class _OppModel:
    """
    Bayesian model over 4 opponent archetypes.

    Archetypes:
      0 = calling_station  — calls almost everything, barely folds
      1 = passive_weak     — checks/calls small bets, folds to pressure
      2 = balanced         — roughly GTO-like
      3 = aggressive_LAG   — bets and raises wide, occasional bluff

    Key output: optimal_bet_frac(equity, pot, max_bet)
      Searches over discrete bet sizes and returns the fraction that
      maximises:  EV = P(fold)*pot + P(call)*(equity*(pot+2*bet) - (1-eq)*bet)
    """

    _N = 4

    # P(action_bucket | type).  action_bucket: 0=fold, 1=call, 2=raise
    _ACT_L = [
        [0.08, 0.77, 0.15],   # calling_station
        [0.52, 0.36, 0.12],   # passive_weak
        [0.33, 0.40, 0.27],   # balanced
        [0.18, 0.37, 0.45],   # aggressive_LAG
    ]

    # P(bid_bucket | type).  bid_bucket: 0=zero, 1=small(<20%), 2=med, 3=large
    _BID_L = [
        [0.55, 0.28, 0.12, 0.05],
        [0.30, 0.35, 0.25, 0.10],
        [0.12, 0.28, 0.38, 0.22],
        [0.05, 0.12, 0.28, 0.55],
    ]

    # P(bet_size_bucket | type).  0=check, 1=small(<33%), 2=med(33-75%), 3=large
    _BET_L = [
        [0.35, 0.40, 0.20, 0.05],
        [0.45, 0.30, 0.18, 0.07],
        [0.28, 0.22, 0.32, 0.18],
        [0.15, 0.15, 0.28, 0.42],
    ]

    # Base fold rates vs pot-sized bet for each type
    _BASE_FOLD = [0.08, 0.52, 0.33, 0.18]

    def __init__(self):
        self._p    = [1.0/self._N]*self._N
        self.n_obs = 0

    def _update(self, lhoods):
        unnorm = [self._p[t]*lhoods[t] for t in range(self._N)]
        tot    = sum(unnorm)
        if tot > 0:
            self._p = [u/tot for u in unnorm]
        self.n_obs += 1

    def observe_action(self, action: str):
        """Call after observing opponent's response to our bet: fold/call/raise."""
        b = {'fold':0,'call':1,'raise':2}.get(action, 1)
        self._update([self._ACT_L[t][b] for t in range(self._N)])

    def observe_bid(self, amount, pot):
        f = amount / max(pot, 1)
        b = 0 if amount==0 else (1 if f<0.20 else (2 if f<0.60 else 3))
        self._update([self._BID_L[t][b] for t in range(self._N)])

    def observe_bet_size(self, amount, pot):
        f = amount / max(pot, 1)
        b = 0 if amount==0 else (1 if f<0.33 else (2 if f<0.75 else 3))
        self._update([self._BET_L[t][b] for t in range(self._N)])

    def _confidence(self):
        import math
        h = -sum(q*math.log(q+1e-12) for q in self._p)
        return max(0.0, 1.0 - h/math.log(self._N))

    def fold_rate(self, bet_frac: float) -> float:
        """
        Expected fold rate vs a bet of `bet_frac` * pot.
        Each archetype has a base fold rate that shifts with bet size:
          - calling_station barely responds to bet sizing (nearly flat)
          - passive_weak folds significantly more to bigger bets
          - balanced/LAG respond moderately
        """
        p = self._p
        per_type = [
            max(0.02, self._BASE_FOLD[0] + (bet_frac-1.0)*0.02),   # calling station: flat
            max(0.10, self._BASE_FOLD[1] + (bet_frac-1.0)*0.22),   # passive: steep
            max(0.10, self._BASE_FOLD[2] + (bet_frac-1.0)*0.14),   # balanced: moderate
            max(0.05, self._BASE_FOLD[3] + (bet_frac-1.0)*0.08),   # LAG: stubborn
        ]
        return sum(p[t]*per_type[t] for t in range(self._N))

    def optimal_bet_frac(self, equity: float, pot: int, max_bet: int) -> float:
        """
        Grid search over candidate bet fracs to maximise:
          EV(f) = P(fold|f)*pot + P(call|f)*(equity*(pot+2*f*pot) - (1-equity)*f*pot)
        Returns the winning fraction.
        """
        candidates = [0.33, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50]
        max_frac   = max_bet / max(pot, 1)
        candidates = [f for f in candidates if f <= max_frac + 0.05]
        if not candidates:
            candidates = [min(max_frac, 0.33)]

        best_ev, best_frac = -1e9, candidates[0]
        for frac in candidates:
            actual_bet = min(frac * pot, max_bet)
            fr         = self.fold_rate(frac)
            cr         = 1.0 - fr
            ev         = fr*pot + cr*(equity*(pot+2*actual_bet) - (1-equity)*actual_bet)
            if ev > best_ev:
                best_ev   = ev
                best_frac = frac
        return best_frac

    def should_trap(self, equity: float) -> bool:
        """
        Check-raise trap: use when confident opponent is LAG and hand is strong.
        LAG bets wide; we check, they bet, we re-raise for maximum extraction.
        """
        if self.n_obs < 15 or equity < THRESH_STRONG:
            return False
        return self._p[3] > 0.50 and self._confidence() > 0.40

    def thresh_adjustments(self):
        """Δval_thresh, Δcall_thresh. Scales with confidence."""
        if self.n_obs < 10:
            return 0.0, 0.0
        p   = self._p
        # calling_station: bet lighter (they'll call anything)
        # passive_weak: bet lighter (they fold to big bets, small bets get more calls)
        # LAG: tighten up (they 3-bet a lot, don't spew)
        dv  = -0.08*p[0] - 0.05*p[1] + 0.00*p[2] + 0.06*p[3]
        dc  = +0.05*p[0] + 0.02*p[1] + 0.00*p[2] - 0.04*p[3]
        lam = min(self._confidence(), 0.80)
        return dv*lam, dc*lam

    def bid_multiplier(self):
        """Auction bid scaling: bid more vs LAG (they overbid), less vs passive."""
        if self.n_obs < 10: return 1.0
        p = self._p
        return 0.78*p[0] + 0.85*p[1] + 1.00*p[2] + 1.35*p[3]
    
    def exploit_mode(self):
        """
        Returns aggression multiplier if opponent is highly exploitable.
        """
        if self.n_obs < 15:
            return 1.0

        conf = self._confidence()
        p = self._p

        # If highly likely passive or calling station
        if conf > 0.45 and (p[0] + p[1]) > 0.65:
            return 1.25   # increase aggression

        # If highly likely LAG
        if conf > 0.45 and p[3] > 0.55:
            return 1.15

        return 1.0

# ============================================================
# Player Bot
# ============================================================

class Player(BaseBot):

    def __init__(self):
        self._strategy = _load_strategy()

        # Per-hand state
        self._equity_cache     = {}
        self._opp_revealed     = []
        self._we_won_auction   = False
        self._our_card_exposed = None
        self._first_postauct   = True
        self._raises           = 0
        self._prev_street      = None
        self._prev_opp_wager   = 0     # track bet-size deltas for model updates

        # Cross-hand state (opponent model)
        self._opp_model     = _OppModel()
        self._hands         = 0
        self._opp_auct_wins = 0
        self._our_auct_wins = 0

    # ── Engine callbacks ──────────────────────────────────────────────────────

    def on_hand_start(self, game_info, state):
        self._equity_cache     = {}
        self._opp_revealed     = []
        self._we_won_auction   = False
        self._our_card_exposed = None
        self._first_postauct   = True
        self._raises           = 0
        self._prev_street      = state.street
        self._prev_opp_wager   = state.opp_wager

    def on_hand_end(self, game_info, state):
        self._hands += 1
        # Final wager signal: large final wager → aggressive type
        self._opp_model.observe_bet_size(state.opp_wager, max(state.pot, 1))

    # ── Main dispatch ─────────────────────────────────────────────────────────

    def get_move(self, game_info, state):
        # T0: Emergency
        if game_info.time_bank < 0.4:
            return self._instant(state)

        # Update opponent model with any new betting delta
        opp_now = state.opp_wager
        if state.street not in ('auction', 'preflop') and opp_now > self._prev_opp_wager:
            self._opp_model.observe_bet_size(
                opp_now - self._prev_opp_wager, max(state.pot, 1)
            )
        self._prev_opp_wager = opp_now

        # Reset raise count on new street
        if state.street != self._prev_street:
            self._raises      = 0
            self._prev_street = state.street

        # Auction disambiguation (critical bug fix)
        self._process_revealed(state)

        if state.street == 'auction':
            return self._auction_bid(state)

        equity = self._get_equity(state)

        # T1: Blueprint (action selection)
        action = self._blueprint(state, equity)
        if action is not None:
            return action

        # T3: Heuristic with exploit sizing
        return self._heuristic(state, equity)

    # ── Auction disambiguation ────────────────────────────────────────────────

    def _process_revealed(self, state):
        """
        Engine fills opp_revealed_cards with OUR card when opponent wins.
        Distinguish by checking set membership in my_hand.
        """
        if not state.opp_revealed_cards:
            return
        card    = state.opp_revealed_cards[0]
        my_hand = set(state.my_hand)

        if card in my_hand:
            # We LOST — card is ours (opponent saw it)
            if self._our_card_exposed != card:
                self._we_won_auction   = False
                self._our_card_exposed = card
                self._opp_revealed     = []
                self._opp_auct_wins   += 1
                self._opp_model.observe_bid(int(state.pot * 0.65), state.pot)
        else:
            # We WON — card is theirs
            if not self._we_won_auction:
                self._we_won_auction   = True
                self._opp_revealed     = [card]
                self._our_card_exposed = None
                self._equity_cache     = {}
                self._our_auct_wins   += 1

    # ── Equity ────────────────────────────────────────────────────────────────

    def _get_equity(self, state):
        key = (tuple(state.board), tuple(self._opp_revealed))
        if key in self._equity_cache:
            return self._equity_cache[key]
        hole  = list(state.my_hand)
        board = list(state.board)
        if not board:
            eq = _chen_equity(_chen(hole))
        else:
            n = MC_SAMPLES_POSTAUCT if (self._we_won_auction and self._first_postauct) \
                else MC_SAMPLES_BET
            self._first_postauct = False
            eq = _equity_mc(hole, board, opp_known=self._opp_revealed or None, n=n)
        self._equity_cache[key] = eq
        return eq

    # ── T2: SPR-aware auction bid ─────────────────────────────────────────────

    def _auction_bid(self, state):
        """
        bid = 4·E·(1-E) · pot · VOI_SCALE · spr_factor
        spr_factor = min(SPR, SPR_CAP) / SPR_CAP ∈ (0,1]
        """
        hole  = list(state.my_hand)
        board = list(state.board)
        pot   = max(state.pot, 1)
        chips = state.my_chips

        eq = _equity_mc(hole, board, n=MC_SAMPLES_AUCTION)

        spr        = min(chips, state.opp_chips) / pot
        spr_factor = max(min(spr, SPR_CAP) / SPR_CAP, 0.15)

        uncertainty = 4.0 * eq * (1.0 - eq)
        protection  = max(0.0, eq - 0.60) * 0.60
        voi_frac    = (uncertainty + protection) * VOI_SCALE * spr_factor

        bid = int(voi_frac * pot)
        bid = max(int(BID_MIN_FRAC * pot), bid)

        if uncertainty > 0.70:
            bid = int(bid * self._opp_model.bid_multiplier())

        bid = min(bid, int(chips * BID_STACK_CAP), chips)
        return ActionBid(max(0, bid))

    # ── T1: Blueprint ─────────────────────────────────────────────────────────

    def _blueprint(self, state, equity):
        if not self._strategy:
            return None
        pot  = max(state.pot, 1)
        pos  = 0 if not state.is_bb else 1
        rev  = self._opp_revealed[0] if self._opp_revealed else None
        auct = 1 if self._we_won_auction else (2 if self._our_card_exposed else 0)
        key  = _make_key(state.street, equity, list(state.board), pot,
                         min(state.my_chips, state.opp_chips),
                         pos, auct, rev, state.cost_to_call, self._raises)
        label = _sample_blueprint(self._strategy, key)
        if label is None:
            return None
        action = _label_to_action(label, state, pot, state.opp_chips)

        # --- NEW: Override raise sizing with exploit sizing ---
        if isinstance(action, ActionRaise):
            confidence = self._opp_model._confidence()
            if confidence > 0.35:
                mn, mx = state.raise_bounds
                mx = min(mx, state.opp_chips)
                if mx >= mn:
                    frac = self._opp_model.optimal_bet_frac(equity, pot, mx)
                    frac *= self._opp_model.exploit_mode()
                    target = int(pot * frac)
                    action = ActionRaise(max(mn, min(target, mx)))
            self._raises += 1

        return action

    # ── T3: Heuristic + exploit sizing ────────────────────────────────────────

    def _heuristic(self, state, equity):
        pot       = max(state.pot, 1)
        cost      = state.cost_to_call
        opp_chips = state.opp_chips
        my_chips  = state.my_chips
        street    = state.street
        pot_odds  = cost / (pot+cost) if cost > 0 else 0.0
        spr       = min(my_chips, opp_chips) / pot

        # ── Thresholds ────────────────────────────────────────────────────────
        val_thresh  = THRESH_BET
        call_thresh = THRESH_CALL

        dv, dc = self._opp_model.thresh_adjustments()
        val_thresh  += dv
        call_thresh += dc

        # Global aggression scaling
        aggr = self._opp_model.exploit_mode()
        val_thresh -= 0.05 * (aggr - 1.0)
        
        if self._we_won_auction and self._opp_revealed:
            rv = _rv(self._opp_revealed[0])
            if rv <= 4:      val_thresh -= 0.10; call_thresh -= 0.08
            elif rv <= 7:    val_thresh -= 0.04; call_thresh -= 0.03
            elif rv >= 11:   val_thresh += 0.10; call_thresh += 0.08
        elif self._our_card_exposed is not None:
            tighten = 0.06
            if self._hands >= 20:
                rate = self._opp_auct_wins / max(self._hands, 1)
                if rate > 0.40: tighten = 0.12
            val_thresh  += tighten
            call_thresh += tighten * 0.6

        # ── Raise helpers ─────────────────────────────────────────────────────
        can_raise = state.can_act(ActionRaise)
        mn, mx    = state.raise_bounds if can_raise else (BIG_BLIND, 0)
        mx_eff    = min(mx, opp_chips)

        def _do_raise(frac):
            if not can_raise or mx_eff < mn: return None
            target = int(pot*frac) if frac < 999 else mx_eff
            return ActionRaise(max(mn, min(target, mx_eff)))

        def _exploit_raise(equity):
            """Chip-EV-maximising raise via opponent model grid search."""
            if not can_raise or mx_eff < mn: return None
            frac   = self._opp_model.optimal_bet_frac(equity, pot, mx_eff)
            target = int(pot * frac) if frac < 999 else mx_eff
            return ActionRaise(max(mn, min(target, mx_eff)))

        def safe_call():
            if state.can_act(ActionCall):  return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        def safe_check():
            if state.can_act(ActionCheck): return ActionCheck()
            return safe_call()

        # ── CRITICAL: After losing auction, never re-raise ────────────────────
        # Opponent has full info on our hand; raise wars = guaranteed leaks.
        if self._our_card_exposed is not None and cost > 0:
            if equity > pot_odds + 0.06 and equity >= call_thresh:
                return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        # ── Stack-commit: shove when strong + low SPR ─────────────────────────
        # Avoids death-by-small-bets; forces maximum extraction in committed pots.
        if equity >= STACK_COMMIT_EQ and spr < STACK_COMMIT_SPR:
            r = _do_raise(999.0)
            if r: return r
            return safe_call()

        # ── Tier 1: Monster (≥ 0.82) → exploit-sized overbet ─────────────────
        if equity >= 0.82:
            r = _exploit_raise(equity)
            return r if r else safe_call()

        # ── Tier 2: Strong value (≥ val_thresh) ───────────────────────────────
        if equity >= val_thresh:
            if cost == 0:
                if self._opp_model.should_trap(equity):
                    # Check-trap vs aggressive opponent: they'll bet, then we re-raise
                    return safe_check()
                r = _exploit_raise(equity)
                return r if r else safe_check()
            else:
                if equity >= THRESH_STRONG:
                    r = _exploit_raise(equity)
                    if r: return r
                return safe_call()

        # ── Tier 3: Marginal ──────────────────────────────────────────────────
        if call_thresh <= equity < val_thresh:
            if cost == 0:
                if street != 'river' and self._we_won_auction:
                    r = _do_raise(0.33)   # semi-bluff: fixed small size, not exploit-sized
                    if r: return r
                return safe_check()
            if equity > pot_odds + 0.04:
                return safe_call()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        # ── Tier 4: Weak ──────────────────────────────────────────────────────
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()

    # ── T0: Emergency ─────────────────────────────────────────────────────────

    def _instant(self, state):
        if state.street == 'auction':  return ActionBid(0)
        if state.can_act(ActionCheck): return ActionCheck()
        if state.can_act(ActionCall) and state.cost_to_call <= 2*BIG_BLIND:
            return ActionCall()
        return ActionFold()


# ============================================================

if __name__ == '__main__':
    run_bot(Player(), parse_args())