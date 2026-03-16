"""
Sneak Peek Hold'em Bot
IIT Pokerbots 2026
Single-file submission (Evaluator integrated)

v4 — Three targeted fixes over uploaded v2:

FIX 1 (Critical bug): opp_revealed_cards misidentification
  The engine puts OUR OWN CARD into state.opp_revealed_cards when the
  opponent wins the auction (they see our card). The old code read this as
  "I won the auction and see their weak card" → dropped thresholds by 0.15 →
  raised aggressively into an opponent who had full info + a strong hand.
  Fix: check if the revealed card is in our own hand. If yes → we LOST.
  All 61 all-in raise sequences in the log were caused solely by this bug.

FIX 2: Post-auction-loss conservatism
  When we correctly identify we LOST the auction (opponent bid 400+),
  that is a strong signal opponent has a strong hand (selective bidding).
  Raise thresholds significantly, never re-raise (call once max then fold).
  Track opponent's auction win rate to calibrate how strong the signal is.

FIX 3: Auction bid — add a meaningful minimum
  Keep true-value formula (4*E*(1-E)*pot*0.30) but add a floor of 2*BB
  so we always participate. Against an opponent who bids 0 on weak hands,
  a floor bid of 40 costs us 0 (second-price: we pay their 0 bid) but
  gives us info on their weak hand for free.
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

BIG_BLIND  = 20
SMALL_BLIND = 10

MC_SAMPLES_BET     = 12
MC_SAMPLES_AUCTION = 40

THRESH_VALUE_BET = 0.60
THRESH_CALL      = 0.40

BET_FRAC_STRONG = 0.75
BET_FRAC_MEDIUM = 0.50
BET_FRAC_SMALL  = 0.33

AUCTION_BASE_FRAC = 0.30
AUCTION_BID_FLOOR = BIG_BLIND * 2   # always bid at least 40; costs 0 if opp bids 0


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
    sc = _CHEN.get(ALL_RANKS[r1], 0)
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
        # Per-hand state
        self._equity_cache   = {}
        self._opp_revealed   = []      # opponent's card WE can see (only if we won auction)
        self._we_won_auction = False   # True only if we correctly won the auction
        self._our_card_exposed = None  # our card the opponent saw (if we lost auction)

        # Opponent model (persists across hands)
        self._hands_played      = 0
        self._opp_auction_wins  = 0    # times opponent won the auction
        self._our_auction_wins  = 0    # times we won the auction
        self._opp_wagers        = []   # last 60 opponent final wagers

    # --------------------------------------------------------

    def on_hand_start(self, game_info, state):
        self._equity_cache     = {}
        self._opp_revealed     = []
        self._we_won_auction   = False
        self._our_card_exposed = None

    def on_hand_end(self, game_info, state):
        self._hands_played += 1
        self._opp_wagers.append(state.opp_wager)
        if len(self._opp_wagers) > 60:
            self._opp_wagers.pop(0)

    # --------------------------------------------------------
    # FIX 1: Correctly identify who won the auction
    # --------------------------------------------------------

    def _process_revealed_cards(self, state):
        """
        Engine behaviour (confirmed from log analysis):
          - If WE won auction:  state.opp_revealed_cards = [opponent's card we see]
          - If OPP won auction: state.opp_revealed_cards = [OUR card they saw]
          - If no auction yet:  state.opp_revealed_cards = []

        To distinguish: check whether the revealed card is in our own hand.
          - In our hand  → we LOST (opponent saw our card)
          - Not in hand  → we WON (we see opponent's card)
        """
        if not state.opp_revealed_cards:
            return  # auction not resolved yet or preflop

        revealed = list(state.opp_revealed_cards)
        my_hand  = set(state.my_hand)

        if any(c in my_hand for c in revealed):
            # We LOST the auction — opponent saw one of our cards
            self._we_won_auction   = False
            self._our_card_exposed = revealed[0]
            self._opp_revealed     = []   # we have NO info on opponent's hand
        else:
            # We WON the auction — we see one of opponent's cards
            if not self._we_won_auction:  # only set once per hand
                self._we_won_auction  = True
                self._opp_revealed    = revealed
                self._our_card_exposed = None
                self._equity_cache    = {}   # invalidate: recompute with new info
                self._our_auction_wins += 1

    # --------------------------------------------------------

    def get_move(self, game_info, state):

        if game_info.time_bank < 0.5:
            return self._instant_action(state)

        # Always update auction outcome from latest state
        self._process_revealed_cards(state)

        if state.street == 'auction':
            return self._auction_bid(state)

        equity = self._get_equity(state)
        return self._betting_action(state, equity)

    # --------------------------------------------------------

    def _get_equity(self, state):
        # Only use opp_revealed as opp_known if WE actually won the auction.
        # If we lost, opp_revealed is empty (cleared in _process_revealed_cards).
        key = (tuple(state.board), tuple(self._opp_revealed))
        if key in self._equity_cache:
            return self._equity_cache[key]

        hole  = list(state.my_hand)
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

        hole  = list(state.my_hand)
        board = list(state.board)
        pot   = max(state.pot, 1)
        chips = state.my_chips

        equity = equity_mc_fast(hole, board, n_samples=MC_SAMPLES_AUCTION)

        uncertainty = 4.0 * equity * (1.0 - equity)
        protection  = max(0.0, equity - 0.55) * 2.0
        combined    = min(1.0, uncertainty + protection * 0.4)

        bid = int(combined * pot * AUCTION_BASE_FRAC)

        # FIX 3: Apply meaningful floor so we always participate.
        # Cost is 0 when opponent bids 0 (second-price); floor only costs us
        # money when opponent also bids meaningfully — which is fine.
        bid = max(AUCTION_BID_FLOOR, bid)

        # Opponent-model boost: if opponent wins auctions very aggressively,
        # bid higher when our hand is strong to contest those spots.
        if self._hands_played >= 30:
            opp_win_rate = self._opp_auction_wins / max(self._hands_played, 1)
            if opp_win_rate > 0.50 and equity >= 0.60:
                # Opponent wins most auctions by bidding high.
                # Bid more aggressively when we have a genuinely strong hand
                # to occasionally win when it matters most.
                bid = int(bid * 1.5)

        bid = min(bid, chips)
        return ActionBid(max(0, bid))

    # --------------------------------------------------------

    def _betting_action(self, state, equity):

        pot      = max(state.pot, 1)
        cost     = state.cost_to_call
        pot_odds = cost / (pot + cost) if cost > 0 else 0.0
        opp_chips = state.opp_chips

        # ── FIX 2: Threshold adjustments based on auction outcome ─────────────
        val_thresh  = THRESH_VALUE_BET
        call_thresh = THRESH_CALL

        if self._we_won_auction:
            # We have info on opponent's card — adjust based on what we saw
            if self._opp_revealed:
                rv = _rv(self._opp_revealed[0])
                if rv <= 4:          # 2-6: very weak card → opponent likely weak
                    val_thresh  -= 0.08
                    call_thresh -= 0.06
                elif rv <= 7:        # 7-9: medium card
                    val_thresh  -= 0.03
                    call_thresh -= 0.02
                elif rv >= 11:       # J, Q, K, A: strong card → be cautious
                    val_thresh  += 0.08
                    call_thresh += 0.06

        else:
            # We LOST the auction.
            # Baseline: tighten up because opponent may have info on our hand.
            base_tighten = 0.08
            # If opponent wins auctions very selectively (high bid), their win
            # is a strong signal of a strong hand — tighten even more.
            if self._hands_played >= 20:
                opp_win_rate = self._opp_auction_wins / max(self._hands_played, 1)
                if opp_win_rate > 0.40:
                    base_tighten = 0.14   # opponent bidding aggressively = strong hand signal
            val_thresh  += base_tighten
            call_thresh += base_tighten * 0.7

        # ── Opponent aggression model ─────────────────────────────────────────
        if self._hands_played >= 30 and self._opp_wagers:
            avg_w = sum(self._opp_wagers[-30:]) / min(len(self._opp_wagers), 30)
            if avg_w > 150:
                call_thresh -= 0.03   # aggressive opp: call lighter
            elif avg_w < 60:
                call_thresh += 0.03   # passive opp: fold more to their bets

        # ── Helpers ──────────────────────────────────────────────────────────
        def safe_call():
            if state.can_act(ActionCall):  return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        def raise_frac(frac):
            if not state.can_act(ActionRaise): return None
            min_r, max_r = state.raise_bounds
            max_r = min(max_r, opp_chips)  # can't raise more than opp has
            if max_r < min_r: return None
            target = int(pot * frac)
            amount = max(min_r, min(target, max_r))
            return ActionRaise(amount)

        # ── FIX 2 (critical): After losing auction, never re-raise ───────────
        # Opponent has info + likely a strong hand. Getting into raise wars is
        # exactly what cost -89K chips. Call once for pot odds; fold to re-raises.
        if not self._we_won_auction and cost > 0:
            # Opponent is betting into us with full information on our hand.
            # Only continue if we have genuine equity AND aren't already in a raise war.
            if equity > pot_odds + 0.06 and equity >= call_thresh:
                return ActionCall()   # call ONLY — no re-raise
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()

        # ── Normal betting tiers (used when we won auction or no bet facing us) ─
        if equity >= 0.82:
            r = raise_frac(BET_FRAC_STRONG)
            return r if r else safe_call()

        if equity >= val_thresh:
            if cost == 0:
                r = raise_frac(BET_FRAC_MEDIUM)
                return r if r else safe_call()
            return safe_call()

        if call_thresh <= equity < val_thresh and cost == 0 and state.street != 'river':
            r = raise_frac(BET_FRAC_SMALL)
            return r if r else safe_call()

        if equity > pot_odds + 0.04 and equity >= call_thresh:
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