"""
Sneak Peek Hold'em Bot
IIT Pokerbots 2026
Single-file submission (Evaluator integrated)

v2 — Key fixes over v1:
  1. AUCTION: bid aggressively to consistently win the auction.
     In the test match, losing EVERY auction caused 97% of total chip loss.
     Second-price auction: overbidding costs nothing (we pay opp's bid),
     so there is almost no downside to bidding high.
  2. POST-AUCTION WIN: tighten fold threshold with revealed card info.
     Use more MC samples for the first (most critical) post-auction decision.
  3. POST-AUCTION LOSS: probe-bet when opponent checked after winning auction
     (they likely saw our strong card → we're ahead, extract value).
  4. OPPONENT MODEL: track aggression to adjust call thresholds.
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

BIG_BLIND      = 20
SMALL_BLIND    = 10
STARTING_STACK = 5000

# MC sample counts
MC_SAMPLES_BET          = 12   # betting streets (cached → ~1ms)
MC_SAMPLES_POST_AUCTION = 30   # first decision after winning auction
MC_SAMPLES_AUCTION      = 50   # auction bid (once per hand)

# Equity thresholds
THRESH_VALUE_BET = 0.58
THRESH_STRONG    = 0.70
THRESH_CALL      = 0.38

# Bet sizing (fraction of pot)
BET_FRAC_STRONG = 0.80
BET_FRAC_MEDIUM = 0.55
BET_FRAC_SMALL  = 0.33

# Auction bidding parameters
# BASE_FRAC raised from 0.30 to 0.90: the information is worth ~pot * 0.9 in EV
AUCTION_BASE_FRAC = 0.90
AUCTION_MIN_BID   = BIG_BLIND * 2      # always bid at least 40 chips
AUCTION_MAX_FRAC  = 0.15               # cap at 15% of stack per auction


# ============================================================
# Hand Evaluator (Integrated, Zero Dependencies)
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
    def __ge__(self, o): return self._tup >= o._tup
    def __eq__(self, o): return self._tup == o._tup
    def __lt__(self, o): return self._tup < o._tup

def _rv(c): return RANK_VAL[c[0]]
def _su(c): return c[1]

def _eval5(cards5):
    rv  = sorted([_rv(c) for c in cards5], reverse=True)
    su  = [_su(c) for c in cards5]
    cnt = Counter(rv)
    freq = sorted(cnt.values(), reverse=True)
    is_flush = len(set(su)) == 1

    def straight(rs):
        if len(set(rs)) != 5: return False, 0
        if rs[0]-rs[4] == 4:  return True, rs[0]
        if rs == [12,3,2,1,0]: return True, 3
        return False, 0

    is_st, st_top = straight(rv)
    # tiebreak: (count desc, rank desc) — fixed tiebreaker bug from v1
    tb = tuple(r for r,_ in sorted(cnt.items(), key=lambda x:(x[1],x[0]), reverse=True))

    if is_flush and is_st: return HandRank(9 if st_top==12 else 8, (st_top,))
    if freq[0]==4:                       return HandRank(7, tb)
    if freq[0]==3 and freq[1]==2:        return HandRank(6, tb)
    if is_flush:                         return HandRank(5, tuple(rv))
    if is_st:                            return HandRank(4, (st_top,))
    if freq[0]==3:                       return HandRank(3, tb)
    if freq[0]==2 and freq[1]==2:        return HandRank(2, tb)
    if freq[0]==2:                       return HandRank(1, tb)
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
    """
    Monte Carlo equity. Cached at call site.
    opp_known: list of revealed opponent cards (0 or 1 element).
    """
    opp_known = list(opp_known) if opp_known else []
    known = set(hole) | set(board) | set(opp_known)
    deck  = [c for c in make_deck() if c not in known]
    need  = 5 - len(board)

    wins, total = 0.0, 0
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
        run  = board + random.sample(rem, need)
        mr   = evaluate_best(hole, run)
        or_  = evaluate_best(opp, run)
        if mr > or_:   wins += 1.0
        elif mr == or_: wins += 0.5
        total += 1

    return wins / max(total, 1)

def chen_score(hole):
    if len(hole) != 2: return 0.0
    c1,c2 = hole
    r1,r2 = _rv(c1),_rv(c2)
    s1,s2 = _su(c1),_su(c2)
    if r1 < r2: r1,r2,s1,s2 = r2,r1,s2,s1
    sc = _CHEN.get(ALL_RANKS[r1], 0)
    if r1 == r2: return max(sc*2, 5)
    if s1 == s2: sc += 2
    gap = r1-r2-1
    sc -= [0,1,2,4,5][min(gap,4)]
    if gap <= 2 and min(r1,r2) >= 2: sc += 1
    return sc

def chen_to_equity(chen):
    return max(0.30, min(0.85, 0.30 + (chen+2)/22.0*0.55))


# ============================================================
# Player Bot
# ============================================================

class Player(BaseBot):

    def __init__(self):
        # Per-hand state (reset in on_hand_start)
        self._equity_cache        = {}
        self._opp_revealed        = []
        self._we_won_auction      = False
        self._first_post_auction  = True   # True until first equity computed post-auction
        self._post_auction_equity = None

        # Opponent model (persists across hands)
        self.hands_played   = 0
        self.opp_wagers     = []    # sliding window of 100 final opp wagers
        self.opp_auctions_won = 0   # count of hands opp won the auction (we lost)

    # --------------------------------------------------------
    # ENGINE CALLBACKS
    # --------------------------------------------------------

    def on_hand_start(self, game_info, state):
        self._equity_cache        = {}
        self._opp_revealed        = []
        self._we_won_auction      = False
        self._first_post_auction  = True
        self._post_auction_equity = None

    def on_hand_end(self, game_info, state):
        self.hands_played += 1
        self.opp_wagers.append(state.opp_wager)
        if len(self.opp_wagers) > 100:
            self.opp_wagers.pop(0)

        # If opp_revealed_cards is empty at hand end, they won the auction
        # (we'd have seen ours if we won, engine shows opp's to us)
        # Actually we track this via _we_won_auction flag
        if not self._we_won_auction:
            self.opp_auctions_won += 1

    def get_move(self, game_info, state):

        # Emergency guard — near time limit, act instantly
        if game_info.time_bank < 0.4:
            return self._instant_action(state)

        # Update revealed card (set by engine after we win auction)
        if state.opp_revealed_cards:
            revealed = list(state.opp_revealed_cards)
            if revealed != self._opp_revealed:
                self._opp_revealed   = revealed
                self._we_won_auction = True
                self._equity_cache   = {}   # invalidate so we recompute with info

        # Auction phase
        if state.street == 'auction':
            return self._compute_auction_bid(state)

        # Normal betting
        equity = self._get_equity(state)
        return self._betting_action(state, equity)

    # --------------------------------------------------------
    # AUCTION BID
    # --------------------------------------------------------

    def _compute_auction_bid(self, state):
        """
        Bid our true value of information in this Vickrey auction.

        Key insight: we pay the OPPONENT'S bid (not ours).
        Bidding high has near-zero cost (we only overpay in exact ties).
        Losing the auction every hand costs ~43 chips/hand in EV.

        Formula:
          uncertainty = 4 * E * (1-E)   [=1.0 at E=0.5, max uncertainty]
          combined    = uncertainty + protection_bonus
          bid         = max(FLOOR, int(combined * pot * 0.90))

        At E=0.50, pot=40: bid = max(40, int(1.0*40*0.90)) = max(40,36) = 40
        At E=0.50, pot=80: bid = max(40, int(1.0*80*0.90)) = 72
        Always beats an opponent bidding formulaically at ~14-24 chips.
        """
        hole  = list(state.my_hand)
        board = list(state.board)
        pot   = max(state.pot, 1)
        chips = state.my_chips

        equity = equity_mc_fast(hole, board, n_samples=MC_SAMPLES_AUCTION)

        uncertainty = 4.0 * equity * (1.0 - equity)      # peaks at 1.0
        protection  = max(0.0, equity - 0.55) * 2.0      # extra bid when strong
        combined    = min(1.0, uncertainty + protection * 0.5)

        # Boost bid if opponent has been winning auctions (they bid high)
        opp_factor = 1.0
        if self.opp_auctions_won > 20:
            opp_factor = 1.25

        bid = int(combined * pot * AUCTION_BASE_FRAC * opp_factor)
        bid = max(AUCTION_MIN_BID, bid)              # floor: always at least 40
        bid = min(bid, int(chips * AUCTION_MAX_FRAC))  # cap at 15% of stack
        bid = min(bid, chips)
        bid = max(0, bid)
        return ActionBid(bid)

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    def _get_equity(self, state):
        board_key    = tuple(state.board)
        revealed_key = tuple(self._opp_revealed)
        cache_key    = (board_key, revealed_key)

        if cache_key in self._equity_cache:
            return self._equity_cache[cache_key]

        hole  = list(state.my_hand)
        board = list(state.board)

        if not board:
            eq = chen_to_equity(chen_score(hole))
        else:
            # First post-auction decision is most critical — use more samples
            if self._we_won_auction and self._first_post_auction:
                n = MC_SAMPLES_POST_AUCTION
                self._first_post_auction  = False
                self._post_auction_equity = None   # will be set below
            else:
                n = MC_SAMPLES_BET

            eq = equity_mc_fast(
                hole, board,
                opp_known=self._opp_revealed if self._opp_revealed else None,
                n_samples=n,
            )
            if self._post_auction_equity is None and self._we_won_auction:
                self._post_auction_equity = eq

        self._equity_cache[cache_key] = eq
        return eq

    # --------------------------------------------------------
    # BETTING DECISION
    # --------------------------------------------------------

    def _betting_action(self, state, equity):
        pot       = max(state.pot, 1)
        cost      = state.cost_to_call
        opp_chips = state.opp_chips
        street    = state.street
        pot_odds  = cost / (pot + cost) if cost > 0 else 0.0

        # ── Helpers ─────────────────────────────────────────────────────────
        def safe_raise(frac):
            if not state.can_act(ActionRaise): return None
            min_r, max_r = state.raise_bounds
            max_r = min(max_r, opp_chips)   # can't raise more than opp has
            if max_r < min_r: return None
            target = int(pot * frac)
            amount = max(min_r, min(target, max_r))
            return ActionRaise(amount)

        def safe_call():
            if state.can_act(ActionCall):   return ActionCall()
            if state.can_act(ActionCheck):  return ActionCheck()
            return ActionFold()

        def safe_check():
            if state.can_act(ActionCheck): return ActionCheck()
            return safe_call()

        # ── Threshold adjustments based on auction outcome ────────────────────
        val_thresh  = THRESH_VALUE_BET
        call_thresh = THRESH_CALL

        if self._we_won_auction:
            # We have info → be more aggressive (lower thresholds)
            val_thresh  -= 0.05
            call_thresh -= 0.04
        else:
            # Blind vs opponent who has info → be tighter
            val_thresh  += 0.04
            call_thresh += 0.03

        # ── Revealed card adjustment ──────────────────────────────────────────
        if self._opp_revealed:
            opp_rv = RANK_VAL.get(self._opp_revealed[0][0], 0)
            if opp_rv >= 9:    # T or higher — their revealed card is strong
                val_thresh  += 0.04
                call_thresh += 0.04
            elif opp_rv <= 5:  # 7 or lower — their revealed card is weak
                val_thresh  -= 0.04
                call_thresh -= 0.04

        # ── Opponent model ───────────────────────────────────────────────────
        if self.hands_played >= 30 and self.opp_wagers:
            avg_w = sum(self.opp_wagers[-30:]) / min(len(self.opp_wagers), 30)
            if avg_w > 150:    # aggressive opponent → call lighter
                call_thresh -= 0.03
            elif avg_w < 60:   # passive opponent → their bets are value, fold more
                call_thresh += 0.03

        # ═══════════════════════════════════════════════════════════════════
        # TIER 1: Monster (>= 0.82) → raise large
        # ═══════════════════════════════════════════════════════════════════
        if equity >= 0.82:
            r = safe_raise(BET_FRAC_STRONG)
            return r if r else safe_call()

        # ═══════════════════════════════════════════════════════════════════
        # TIER 2: Strong (>= val_thresh) → bet/raise for value
        # ═══════════════════════════════════════════════════════════════════
        if equity >= val_thresh:
            if cost == 0:
                frac = BET_FRAC_MEDIUM if equity >= THRESH_STRONG else BET_FRAC_SMALL
                r = safe_raise(frac)
                return r if r else safe_check()
            else:
                if equity >= THRESH_STRONG:
                    r = safe_raise(BET_FRAC_MEDIUM)
                    if r: return r
                return safe_call()

        # ═══════════════════════════════════════════════════════════════════
        # TIER 3: Marginal / semi-bluff
        # ═══════════════════════════════════════════════════════════════════
        if call_thresh <= equity < val_thresh:
            if cost == 0:
                # Semi-bluff only when we have information (not blind)
                if street not in ('river',) and self._we_won_auction:
                    r = safe_raise(BET_FRAC_SMALL)
                    if r: return r

                # PROBE BET: opponent won auction, then CHECKED to us.
                # They likely saw our strong card → they're weak here.
                # Extract value with a small bet even on non-draws.
                if not self._we_won_auction and equity >= 0.50:
                    r = safe_raise(BET_FRAC_SMALL)
                    if r: return r

                return safe_check()
            else:
                # Facing a bet from opponent who has our info
                if equity > pot_odds + 0.04:
                    return safe_call()
                # Call small probes from opponent (might be bluffing with weak card)
                if not self._we_won_auction:
                    bet_to_pot = cost / pot
                    if bet_to_pot < 0.33 and equity > 0.40:
                        return safe_call()
                if state.can_act(ActionCheck): return ActionCheck()
                return ActionFold()

        # ═══════════════════════════════════════════════════════════════════
        # TIER 4: Weak → fold or check
        # ═══════════════════════════════════════════════════════════════════
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()

    # --------------------------------------------------------
    # INSTANT / EMERGENCY
    # --------------------------------------------------------

    def _instant_action(self, state):
        if state.street == 'auction':
            return ActionBid(min(AUCTION_MIN_BID, state.my_chips))
        if state.can_act(ActionCheck): return ActionCheck()
        if state.can_act(ActionCall) and state.cost_to_call <= BIG_BLIND * 2:
            return ActionCall()
        return ActionFold()


# ============================================================

if __name__ == '__main__':
    run_bot(Player(), parse_args())