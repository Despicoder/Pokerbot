"""
cfr_train.py  —  High-Performance DCFR Blueprint Trainer
=======================================================================
Combines:
  1. Ultra-fast data structures (Mutable Lists, Int Tuples, bisect)
  2. True Showdown Evaluator & -1.15x Risk-Averse Payoff
  3. Expanded 16 Hand Buckets & 8 Board Buckets mapping
  4. Flop MC n=6 for high fidelity noise reduction
"""

import argparse
import bisect
import pickle
import random
import sys
import time
from collections import defaultdict

from abstraction import (
    ALL_RANKS, SUITS, RANK_VAL, make_deck, rv, suit,
    equity_mc, evaluate_best, evaluate_showdown,
    hand_bucket, board_bucket, pot_bucket, spr_bucket,
    call_bucket, revealed_bucket,
    BID_FRACS, BET_FRACS,
    bid_label_to_amount,
)

# ── Game parameters ───────────────────────────────────────────────────────────
BIG_BLIND   = 20
START_STACK = 5000

ALPHA = 1.5
BETA  = 0.0   # neg_discount = 0.5 forever
GAMMA = 2.0
_NEG_DISCOUNT = 0.5  

# ── Street indices ────────────────────────────────────────────────────────────
ST_PREFLOP  = 0
ST_FLOP     = 1
ST_AUCTION  = 2
ST_FLOP_BET = 3
ST_TURN     = 4
ST_RIVER    = 5

_ST_NEXT = (ST_FLOP, ST_AUCTION, -1, ST_TURN, ST_RIVER, -1)
_ST_NAME = ('preflop', 'flop', 'auction', 'flop_bet', 'turn', 'river')

# ── Betting action indices ────────────────────────────────────────────────────
A_FOLD  = 0
A_CHECK = 1
A_CALL  = 2
A_B33   = 3
A_B55   = 4
A_B100  = 5
A_ALLIN = 6
N_BET   = 7

_BET_FRAC = (None, None, None, 0.33, 0.55, 1.00, 999.0)
_BET_LBL  = ('fold','check','call','bet_33','bet_55','bet_100','bet_allin')

# ── Bid action indices ────────────────────────────────────────────────────────
N_BID = 8
_BID_FRAC = (0.0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.20, 999.0)
_BID_LBL  = ('bid_0','bid_10p','bid_20p','bid_35p','bid_55p','bid_80p','bid_120p','bid_allin')

# ── Mutable state as a list (22 elements) ────────────────────────────────────
I_ST   = 0   
I_HB0  = 1   
I_HB1  = 2   
I_BB0  = 3   
I_POT  = 4   
I_ST0  = 5   
I_ST1  = 6   
I_POS  = 7   
I_AUC  = 8   
I_REV0 = 9   
I_REV1 = 10  
I_RZ   = 11  
I_TC   = 12  
I_LA   = 13  
I_TERM = 14  
I_PY0  = 15  
I_HOLE0 = 16
I_HOLE1 = 17
I_BOARD_RIVER = 18   
I_HB0_ARR = 19       
I_HB1_ARR = 20       
I_BB_ARR  = 21       

_N_STATE = 22

# ── Information Maps ─────────────────────────────────────────────────────────

# Extrapolated for 16 buckets. Calculates [low_prob, low_prob + mid_prob]
_REV_WEIGHTS = tuple(
    (0.55 - i * 0.03, 0.85 - i * 0.036) for i in range(16)
)

def _rand_rev_bucket(hb):
    """Sample a revealed_bucket given opponent hand bucket (now scaled 0-15)."""
    thresholds = _REV_WEIGHTS[hb if hb < 16 else 15]
    r = random.random()
    if r < thresholds[0]: return 1
    if r < thresholds[1]: return 2
    return 3

def _key(state, player):
    hb  = state[I_HB0] if player == 0 else state[I_HB1]
    auc = state[I_AUC]
    if player == 1:
        auc = (0, 2, 1)[auc]
    rev  = state[I_REV0] if player == 0 else state[I_REV1]
    pot  = state[I_POT]
    tc   = state[I_TC]
    stk  = state[I_ST0] if state[I_ST0] < state[I_ST1] else state[I_ST1]

    pb  = 0 if pot<60 else (1 if pot<200 else (2 if pot<500 else 3))
    spr = stk/pot if pot > 0 else 999.0
    spb = 0 if spr<1 else (1 if spr<3 else (2 if spr<8 else 3))
    cb  = 0 if tc<=0 else (1 if tc/pot<0.25 else (2 if tc/pot<0.50 else 3)) if pot>0 else 0
    rz  = state[I_RZ] if state[I_RZ] < 3 else 3

    return (state[I_ST], hb, state[I_BB0], pb, spb, player, auc, rev, cb, rz)

def _legal_acts(state):
    tc  = state[I_TC]
    pos = state[I_POS]
    si  = state[I_ST0 + pos]       
    sj  = state[I_ST0 + (1 - pos)] 
    pot = state[I_POT]
    mb  = si if si < sj else sj     

    if tc > 0: acts = [A_FOLD, A_CALL]
    else: acts = [A_CHECK]

    min_bet = BIG_BLIND if BIG_BLIND > tc else tc
    if mb >= min_bet:
        p33, p55, p100 = int(pot * 0.33), int(pot * 0.55), pot
        if min_bet <= p33  <= mb: acts.append(A_B33)
        if min_bet <= p55  <= mb: acts.append(A_B55)
        if min_bet <= p100 <= mb: acts.append(A_B100)
        if mb > 0:                acts.append(A_ALLIN)
    return acts

def _sample(probs, acts):
    total = 0.0
    cum   = []
    for p in probs:
        total += p
        cum.append(total)
    r   = random.random() * total
    idx = bisect.bisect_left(cum, r)
    if idx >= len(acts): idx = len(acts) - 1
    return acts[idx]

def _regret_match(reg, acts):
    total = 0.0
    pos   = [0.0] * len(acts)
    for i, a in enumerate(acts):
        r = reg[a]
        if r > 0.0:
            pos[i] = r
            total += r
    if total > 0.0:
        inv = 1.0 / total
        return [p * inv for p in pos]
    n = len(acts)
    return [1.0 / n] * n

def _advance_street(state):
    nxt = _ST_NEXT[state[I_ST]]

    if nxt == -1:
        state[I_TERM] = 1
        winner = evaluate_showdown(state[I_HOLE0], state[I_HOLE1], state[I_BOARD_RIVER])
        if winner == 1: state[I_PY0] = state[I_POT] * 1.0
        elif winner == -1: state[I_PY0] = state[I_POT] * -1.15
        else: state[I_PY0] = 0.0
        return

    if nxt == ST_AUCTION:
        state[I_ST]  = ST_AUCTION
        state[I_POS] = 0; state[I_TC] = 0; state[I_RZ] = 0; state[I_LA] = -1
        return

    state[I_ST]  = nxt
    state[I_POS] = 1; state[I_TC] = 0; state[I_RZ] = 0; state[I_LA] = -1
    
    if nxt in (ST_FLOP, ST_FLOP_BET, ST_TURN, ST_RIVER):
        state[I_HB0] = state[I_HB0_ARR][nxt]
        state[I_HB1] = state[I_HB1_ARR][nxt]
        state[I_BB0] = state[I_BB_ARR][nxt]

def _apply_bet_inplace(state, action):
    pos = state[I_POS]; opp = 1 - pos; pot = state[I_POT]

    if action == A_FOLD:
        state[I_TERM] = 1
        state[I_PY0]  = pot if pos == 1 else -pot
        return True

    if action == A_CHECK:
        if state[I_LA] == A_CHECK:
            _advance_street(state)
            return state[I_TERM] == 1
        state[I_LA], state[I_POS], state[I_RZ]  = A_CHECK, opp, 0
        return False

    if action == A_CALL:
        call_amt = state[I_TC]
        si       = state[I_ST0 + pos]
        if call_amt > si: call_amt = si
        state[I_ST0 + pos] -= call_amt
        state[I_POT]       += call_amt
        state[I_TC]         = 0
        _advance_street(state)
        return state[I_TERM] == 1

    frac = _BET_FRAC[action]
    si, sj = state[I_ST0 + pos], state[I_ST0 + opp]
    if frac >= 999.0: amount = si if si < sj else sj
    else:
        amount = int(pot * frac)
        cap    = si if si < sj else sj
        if amount > cap:  amount = cap
        if amount < BIG_BLIND: amount = BIG_BLIND
        
    state[I_ST0 + pos] -= amount
    state[I_POT]       += amount
    state[I_TC]         = amount
    state[I_RZ]        += 1
    state[I_LA], state[I_POS] = action, opp
    return False

def _apply_bid_inplace(state, bid0_idx, bid1_idx):
    pot    = state[I_POT]
    chips0, chips1 = state[I_ST0], state[I_ST1]

    frac0, frac1  = _BID_FRAC[bid0_idx], _BID_FRAC[bid1_idx]
    amt0   = chips0 if frac0 >= 999.0 else min(int(pot * frac0), chips0)
    amt1   = chips1 if frac1 >= 999.0 else min(int(pot * frac1), chips1)

    if amt0 > amt1:
        state[I_ST0] -= amt1; state[I_POT] += amt1; state[I_AUC] = 1
        state[I_REV0] = _rand_rev_bucket(state[I_HB1])
        state[I_REV1] = 0
    elif amt1 > amt0:
        state[I_ST1] -= amt0; state[I_POT] += amt0; state[I_AUC]  = 2
        state[I_REV0] = 0
        state[I_REV1] = _rand_rev_bucket(state[I_HB0])
    else:
        state[I_ST0] -= amt0; state[I_ST1] -= amt1; state[I_POT] += amt0 + amt1; state[I_AUC] = 1
        state[I_REV0] = _rand_rev_bucket(state[I_HB1])
        state[I_REV1] = _rand_rev_bucket(state[I_HB0])

    state[I_ST]  = ST_FLOP_BET
    state[I_POS] = 1; state[I_TC] = 0; state[I_RZ] = 0; state[I_LA] = -1

# ── DCFR traversal ────────────────────────────────────────────────────────────
def dcfr_traverse(state, player, r0, r1, pos_d, neg_d, w_base, REG, AVG):
    if state[I_TERM]: return state[I_PY0] if player == 0 else -state[I_PY0]
    if state[I_ST] == ST_AUCTION: return _dcfr_auction(state, player, r0, r1, pos_d, neg_d, w_base, REG, AVG)

    acts  = _legal_acts(state)
    key   = _key(state, state[I_POS])
    reg   = REG.get(key)
    if reg is None:
        reg = [0.0] * N_BET
        REG[key] = reg
    strat = _regret_match(reg, acts)

    pos      = state[I_POS]
    reach_i  = r0 if pos == 0 else r1
    reach_j  = r1 if pos == 0 else r0

    if pos == player:
        utils = [0.0] * len(acts)
        for i, a in enumerate(acts):
            saved   = state[:]                         
            _apply_bet_inplace(state, a)
            new_r0  = r0 * strat[i] if pos == 0 else r0
            new_r1  = r1 * strat[i] if pos == 1 else r1
            utils[i] = dcfr_traverse(state, player, new_r0, new_r1, pos_d, neg_d, w_base, REG, AVG)
            state[:] = saved                           

        node_util = sum(strat[i] * utils[i] for i in range(len(acts)))

        for i, a in enumerate(acts):
            inst = utils[i] - node_util
            old  = reg[a]
            reg[a] = old * pos_d + inst * reach_j if old >= 0 else old * neg_d + inst * reach_j

        avgs = AVG.get(key)
        if avgs is None:
            avgs = [0.0] * N_BET
            AVG[key] = avgs
        w = w_base * reach_i
        for i, a in enumerate(acts):
            avgs[a] += w * strat[i]

        return node_util
    else:
        a       = _sample(strat, acts)
        saved   = state[:]
        _apply_bet_inplace(state, a)
        new_r0  = r0 * strat[acts.index(a)] if pos == 0 else r0
        new_r1  = r1 * strat[acts.index(a)] if pos == 1 else r1
        u       = dcfr_traverse(state, player, new_r0, new_r1, pos_d, neg_d, w_base, REG, AVG)
        state[:] = saved
        return u

def _dcfr_auction(state, player, r0, r1, pos_d, neg_d, w_base, REG, AVG):
    key0, key1 = _key(state, 0), _key(state, 1)

    reg0 = REG.get(key0)
    if reg0 is None: reg0 = [0.0]*N_BID; REG[key0] = reg0
    reg1 = REG.get(key1)
    if reg1 is None: reg1 = [0.0]*N_BID; REG[key1] = reg1

    bids = list(range(N_BID))
    s0, s1   = _regret_match(reg0, bids), _regret_match(reg1, bids)

    if player == 0:
        b1    = _sample(s1, bids)
        utils = [0.0] * N_BID
        for b0 in bids:
            saved = state[:]
            _apply_bid_inplace(state, b0, b1)
            utils[b0] = dcfr_traverse(state, player, r0 * s0[b0], r1, pos_d, neg_d, w_base, REG, AVG)
            state[:] = saved

        node_util = sum(s0[b] * utils[b] for b in bids)
        for b in bids:
            inst, old  = utils[b] - node_util, reg0[b]
            reg0[b] = old * pos_d + inst * r1 if old >= 0 else old * neg_d + inst * r1

        avgs0 = AVG.get(key0)
        if avgs0 is None: avgs0 = [0.0]*N_BID; AVG[key0] = avgs0
        w = w_base * r0
        for b in bids: avgs0[b] += w * s0[b]
        return node_util

    else:
        b0    = _sample(s0, bids)
        utils = [0.0] * N_BID
        for b1 in bids:
            saved = state[:]
            _apply_bid_inplace(state, b0, b1)
            utils[b1] = dcfr_traverse(state, player, r0, r1 * s1[b1], pos_d, neg_d, w_base, REG, AVG)
            state[:] = saved

        node_util = sum(s1[b] * utils[b] for b in bids)
        for b in bids:
            inst, old  = utils[b] - node_util, reg1[b]
            reg1[b] = old * pos_d + inst * r1 if old >= 0 else old * neg_d + inst * r1

        avgs1 = AVG.get(key1)
        if avgs1 is None: avgs1 = [0.0]*N_BID; AVG[key1] = avgs1
        w = w_base * r1
        for b in bids: avgs1[b] += w * s1[b]
        return -node_util

# ── Preflop equity cache ──────────────────────────────────────────────────────
_PREFLOP_EQ_CACHE = {}

def _prewarm_equity_cache():
    deck = make_deck()
    count = 0
    for i in range(len(deck)):
        for j in range(i+1, len(deck)):
            c1, c2 = deck[i], deck[j]
            r1, r2 = rv(c1), rv(c2)
            if r1 < r2: r1, r2 = r2, r1
            suited = 's' if suit(c1) == suit(c2) else 'o'
            key = (r1, r2, suited)
            if key not in _PREFLOP_EQ_CACHE:
                _PREFLOP_EQ_CACHE[key] = equity_mc([c1, c2], [], n=50)
                count += 1
    return count

def _preflop_equity(hand):
    c1, c2 = hand
    r1, r2 = rv(c1), rv(c2)
    if r1 < r2: r1, r2 = r2, r1
    suited = 's' if suit(c1) == suit(c2) else 'o'
    key = (r1, r2, suited)
    eq = _PREFLOP_EQ_CACHE.get(key)
    if eq is None:
        eq = equity_mc(hand, [], n=50)
        _PREFLOP_EQ_CACHE[key] = eq
    return eq

def _sample_initial_state():
    deck = make_deck()
    random.shuffle(deck)
    h0, h1   = deck[:2], deck[2:4]
    
    board_flop  = deck[4:7]
    board_turn  = deck[4:8]
    board_river = deck[4:9]

    eq0_pre, eq1_pre = _preflop_equity(h0), _preflop_equity(h1)
    
    hb0_arr, hb1_arr, bb_arr = [0]*6, [0]*6, [0]*6
    
    hb0_arr[ST_PREFLOP] = hand_bucket(eq0_pre)
    hb1_arr[ST_PREFLOP] = hand_bucket(eq1_pre)
    bb_arr[ST_PREFLOP]  = 0

    # Increased Flop precision to n=6 to smooth variance
    hb0_arr[ST_FLOP] = hand_bucket(equity_mc(h0, board_flop, n=6))
    hb1_arr[ST_FLOP] = hand_bucket(equity_mc(h1, board_flop, n=6))
    bb_arr[ST_FLOP]  = board_bucket(board_flop)
    
    hb0_arr[ST_FLOP_BET] = hb0_arr[ST_FLOP]
    hb1_arr[ST_FLOP_BET] = hb1_arr[ST_FLOP]
    bb_arr[ST_FLOP_BET]  = bb_arr[ST_FLOP]

    hb0_arr[ST_TURN] = hand_bucket(equity_mc(h0, board_turn, n=2))
    hb1_arr[ST_TURN] = hand_bucket(equity_mc(h1, board_turn, n=2))
    bb_arr[ST_TURN]  = board_bucket(board_turn)

    hb0_arr[ST_RIVER] = hand_bucket(equity_mc(h0, board_river, n=2))
    hb1_arr[ST_RIVER] = hand_bucket(equity_mc(h1, board_river, n=2))
    bb_arr[ST_RIVER]  = board_bucket(board_river)

    state = [0] * _N_STATE
    state[I_ST]   = ST_PREFLOP
    state[I_HB0]  = hb0_arr[ST_PREFLOP]
    state[I_HB1]  = hb1_arr[ST_PREFLOP]
    state[I_BB0]  = 0      
    state[I_POT]  = BIG_BLIND + BIG_BLIND // 2
    state[I_ST0]  = START_STACK - BIG_BLIND // 2
    state[I_ST1]  = START_STACK - BIG_BLIND
    state[I_POS]  = 0; state[I_AUC]  = 0; state[I_REV0] = 0; state[I_REV1] = 0
    state[I_RZ]   = 0; state[I_TC]   = BIG_BLIND // 2; state[I_LA]   = -1
    state[I_TERM] = 0; state[I_PY0]  = 0.0
    
    state[I_HOLE0] = h0
    state[I_HOLE1] = h1
    state[I_BOARD_RIVER] = board_river
    state[I_HB0_ARR] = tuple(hb0_arr)
    state[I_HB1_ARR] = tuple(hb1_arr)
    state[I_BB_ARR]  = tuple(bb_arr)
    
    return state

def _extract_strategy(AVG):
    strategy = {}
    for key, counts in AVG.items():
        total = sum(counts)
        if total <= 0: continue
        st = key[0]
        labels, n = (_BID_LBL, N_BID) if st == ST_AUCTION else (_BET_LBL, N_BET)

        str_key = (f"{_ST_NAME[key[0]]}|{key[1]}|{key[2]}|{key[3]}|{key[4]}"
                   f"|{key[5]}|{key[6]}|{key[7]}|{key[8]}|{key[9]}")
        inv     = 1.0 / total
        dist    = {labels[i]: counts[i] * inv for i in range(n) if counts[i] > 0}
        if dist: strategy[str_key] = dist
    return strategy

def train(n_iters, checkpoint_every, output_path):
    print(f"Starting Highly-Optimized Single-Thread DCFR: {n_iters:,} iterations → {output_path}")
    print("Pre-warming preflop equity cache...", end=' ', flush=True)
    t0     = time.time()
    n_warm = _prewarm_equity_cache()
    print(f"{n_warm} hand types computed in {time.time()-t0:.1f}s\n")

    REG = {}
    AVG = {}
    start_time = time.time()

    for i in range(n_iters):
        t     = i + 1
        pos_d = (t ** ALPHA) / (t ** ALPHA + 1.0)
        w_b   = t ** GAMMA

        state = _sample_initial_state()
        dcfr_traverse(state, 0, 1.0, 1.0, pos_d, _NEG_DISCOUNT, w_b, REG, AVG)
        dcfr_traverse(state, 1, 1.0, 1.0, pos_d, _NEG_DISCOUNT, w_b, REG, AVG)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate    = (i + 1) / elapsed
            print(f"  [{i+1:,}] {rate:.0f} iters/sec | infosets: {len(AVG):,} | "
                  f"elapsed: {elapsed:.0f}s", end='\r')

        if (i + 1) % checkpoint_every == 0:
            print()
            strategy = _extract_strategy(AVG)
            with open(output_path, 'wb') as f: pickle.dump(strategy, f, protocol=4)
            print(f"  [Checkpoint] iter {i+1:,} | infosets: {len(strategy):,} → {output_path}")
            sys.stdout.flush()

    print()
    strategy = _extract_strategy(AVG)
    with open(output_path, 'wb') as f:
        pickle.dump(strategy, f, protocol=4)

    elapsed = time.time() - start_time
    print(f"Done. {n_iters:,} iters in {elapsed:.0f}s ({n_iters/elapsed:.1f} iters/sec)")
    print(f"Final infosets: {len(strategy):,} → {output_path}")
    return strategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters',      type=int, default=300_000)
    parser.add_argument('--checkpoint', type=int, default=50_000)
    parser.add_argument('--out',        type=str, default='strategy.pkl')
    args = parser.parse_args()
    train(args.iters, args.checkpoint, args.out)