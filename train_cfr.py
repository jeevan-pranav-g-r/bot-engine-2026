#!/usr/bin/env python3

import multiprocessing as mp
import random
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
import eval7

# =========================
# CONFIG
# =========================

STACK = 5000
SB = 10
BB = 20

MAX_RAISES_PER_STREET = 3
MAX_DEPTH = 6

MC_SAMPLES_FLOP = 40
MC_SAMPLES_TURN = 25
MC_SAMPLES_RIVER = 15

DISCOUNT = 0.999

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

RAISE_SIZE = {"raise_small": 0.5, "raise_big": 1.0}

# =========================
# UTILS
# =========================

def dd():
    return defaultdict(float)

# =========================
# EQUITY CACHE
# =========================

equity_cache = {}

def equity_estimate(hole, board):
    key = (hole, tuple(board))
    if key in equity_cache:
        return equity_cache[key]

    if len(board) == 3:
        samples = MC_SAMPLES_FLOP
    elif len(board) == 4:
        samples = MC_SAMPLES_TURN
    else:
        samples = MC_SAMPLES_RIVER

    known = set(hole + tuple(board))
    remain = [c for c in FULL_DECK if c not in known]

    hero = [eval7.Card(hole[0]), eval7.Card(hole[1])]
    board_cards = [eval7.Card(c) for c in board]
    need = 5 - len(board)

    wins = 0
    for _ in range(samples):
        draw = random.sample(remain, 2 + need)
        opp = [eval7.Card(draw[0]), eval7.Card(draw[1])]
        runout = board_cards + [eval7.Card(x) for x in draw[2:]]
        hv = eval7.evaluate(runout + hero)
        ov = eval7.evaluate(runout + opp)
        if hv > ov:
            wins += 1
        elif hv == ov:
            wins += 0.5

    value = wins / samples
    equity_cache[key] = value
    return value

# =========================
# ABSTRACTION
# =========================

def preflop_bucket(hole):
    vals = sorted([RANKS.index(c[0]) + 2 for c in hole], reverse=True)
    hi, lo = vals
    suited = hole[0][1] == hole[1][1]
    pair = hi == lo
    gap = hi - lo

    score = (hi + lo) / 28.0
    if pair: score += 0.5
    if suited: score += 0.07
    if gap <= 1: score += 0.05

    return min(5, int(score * 6))

def postflop_bucket(hole, board):
    cards = list(hole) + list(board)
    ranks = [c[0] for c in cards]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)

    if counts[0] >= 4:
        return 5
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return 5
    if counts[0] == 3:
        return 4
    if len(counts) > 1 and counts[0] == 2 and counts[1] == 2:
        return 3
    if counts[0] == 2:
        return 2

    # Flush draw bucket
    suits = {}
    for c in cards:
        suits[c[1]] = suits.get(c[1], 0) + 1
    if max(suits.values()) >= 4:
        return 3

    return 1

def board_texture(board):
    if len(board) < 3:
        return 0
    suits = {}
    for c in board:
        suits[c[1]] = suits.get(c[1], 0) + 1
    return 1 if max(suits.values()) >= 3 else 0

def encode_key(street, h, e, t, p, c, r):
    return (
        (street << 20) |
        (h << 17) |
        (e << 14) |
        (t << 12) |
        (p << 9) |
        (c << 6) |
        r
    )

# =========================
# GAME STATE
# =========================

@dataclass
class Node:
    street: int
    board: List[str]
    hero: Tuple[str,str]
    opp: Tuple[str,str]
    to_act: int
    hero_contrib: int
    opp_contrib: int
    hero_stack: int
    opp_stack: int
    raises: int
    terminal: bool=False
    folded: int=-1

# =========================
# INFO KEY
# =========================

def info_key(st, player):
    hole = st.hero if player == 0 else st.opp

    if st.street == 0:
        h = preflop_bucket(hole)
        e = h
        t = 0
    else:
        visible = st.board[:3 if st.street==1 else 4 if st.street==2 else 5]
        h = preflop_bucket(hole)
        e = postflop_bucket(hole, visible)
        t = board_texture(visible)

    pot = st.hero_contrib + st.opp_contrib
    cost = abs(st.hero_contrib - st.opp_contrib)

    return encode_key(
        st.street,
        h,
        e,
        t,
        min(5, int(pot / 300)),
        min(4, int(cost / 200)),
        min(3, st.raises)
    )

# =========================
# STREET ADVANCE
# =========================

def advance_street(st):
    if st.street == 3:
        st.terminal = True
        return st
    return Node(
        street=st.street+1,
        board=st.board,
        hero=st.hero,
        opp=st.opp,
        to_act=0,
        hero_contrib=st.hero_contrib,
        opp_contrib=st.opp_contrib,
        hero_stack=st.hero_stack,
        opp_stack=st.opp_stack,
        raises=0
    )

# =========================
# LEGAL ACTIONS
# =========================

def legal(st):
    if st.terminal:
        return []

    cost = st.opp_contrib - st.hero_contrib if st.to_act==0 else st.hero_contrib - st.opp_contrib
    acts = []

    if cost > 0:
        acts += ["fold", "call"]
    else:
        acts.append("call")

    if st.raises < MAX_RAISES_PER_STREET:
        acts += ["raise_small", "raise_big"]

    return list(dict.fromkeys(acts))

# =========================
# APPLY
# =========================

def apply(st, action):
    s = Node(**st.__dict__)
    player = s.to_act

    if action == "fold":
        s.terminal = True
        s.folded = player
        return s

    cost = s.opp_contrib - s.hero_contrib if player==0 else s.hero_contrib - s.opp_contrib

    if action == "call":
        pay = min(cost, s.hero_stack if player==0 else s.opp_stack)
        if player==0:
            s.hero_contrib += pay
            s.hero_stack -= pay
        else:
            s.opp_contrib += pay
            s.opp_stack -= pay

        if s.hero_contrib == s.opp_contrib:
            return advance_street(s)

        s.to_act = 1-player
        return s

    # Proper pot-after-call sizing
    frac = RAISE_SIZE[action]
    pot = s.hero_contrib + s.opp_contrib
    total_after_call = pot + cost
    raise_amt = int(total_after_call * frac)

    if player == 0:
        raise_amt = min(raise_amt, s.hero_stack)
        s.hero_contrib += raise_amt
        s.hero_stack -= raise_amt
    else:
        raise_amt = min(raise_amt, s.opp_stack)
        s.opp_contrib += raise_amt
        s.opp_stack -= raise_amt

    s.raises += 1
    s.to_act = 1-player
    return s

# =========================
# TERMINAL / HEURISTIC
# =========================

def terminal_value(st):
    if st.folded == 0:
        return -st.hero_contrib
    if st.folded == 1:
        return st.opp_contrib

    board = st.board[:5]
    hero_cards = [eval7.Card(st.hero[0]), eval7.Card(st.hero[1])]
    opp_cards = [eval7.Card(st.opp[0]), eval7.Card(st.opp[1])]
    board_cards = [eval7.Card(c) for c in board]

    hv = eval7.evaluate(board_cards + hero_cards)
    ov = eval7.evaluate(board_cards + opp_cards)

    pot = st.hero_contrib + st.opp_contrib

    if hv > ov:
        return pot - st.hero_contrib
    if hv < ov:
        return -st.hero_contrib
    return 0

def heuristic_leaf(st):
    visible = st.board[:5]
    eq = equity_estimate(st.hero, visible)
    pot = st.hero_contrib + st.opp_contrib
    return eq * pot - st.hero_contrib

# =========================
# CFR+
# =========================

regret_sum = defaultdict(dd)
strategy_sum = defaultdict(dd)

def mccfr(st, p0, p1, updating, depth):
    if st.terminal:
        return terminal_value(st)

    if depth >= MAX_DEPTH:
        return heuristic_leaf(st)

    player = st.to_act
    key = info_key(st, player)
    acts = legal(st)

    pos = [max(0, regret_sum[key][a]) for a in acts]
    s = sum(pos)
    strat = {a:(pos[i]/s if s>0 else 1/len(acts)) for i,a in enumerate(acts)}

    for a in acts:
        strategy_sum[key][a] += (p0 if player==0 else p1)*strat[a]

    if player != updating:
        a = random.choices(acts, weights=[strat[a] for a in acts])[0]
        nxt = apply(st,a)
        return mccfr(nxt,
                     p0*strat[a] if player==0 else p0,
                     p1*strat[a] if player==1 else p1,
                     updating,
                     depth+1)

    node_util = 0
    action_util = {}

    for a in acts:
        nxt = apply(st,a)
        util = mccfr(nxt,
                     p0*strat[a] if player==0 else p0,
                     p1*strat[a] if player==1 else p1,
                     updating,
                     depth+1)
        action_util[a] = util
        node_util += strat[a]*util

    for a in acts:
        regret = action_util[a] - node_util
        if player==0:
            regret_sum[key][a] = max(0, regret_sum[key][a]*DISCOUNT + p1*regret)
        else:
            regret_sum[key][a] = max(0, regret_sum[key][a]*DISCOUNT + p0*(-regret))

    return node_util

# =========================
# DEAL / WORKER
# =========================

def deal():
    deck = FULL_DECK[:]
    random.shuffle(deck)
    hero = (deck.pop(), deck.pop())
    opp = (deck.pop(), deck.pop())
    board = [deck.pop() for _ in range(5)]
    return hero, opp, board

def worker(iters, seed):
    random.seed(seed)

    local_regret = defaultdict(dd)
    local_strategy = defaultdict(dd)

    global regret_sum, strategy_sum
    regret_sum = local_regret
    strategy_sum = local_strategy

    for i in range(1, iters+1):
        hero, opp, board = deal()
        root = Node(0, board, hero, opp, 0,
                    SB, BB,
                    STACK-SB, STACK-BB,
                    0)

        updating = i % 2
        mccfr(root,1,1,updating,0)

        if i % 10000 == 0:
            print(f"[PID {os.getpid()}] {i} iterations", flush=True)

    return local_regret, local_strategy

# =========================
# MERGE / EXPORT
# =========================

def merge(results):
    for local_regret, local_strategy in results:
        for k, amap in local_regret.items():
            for a,v in amap.items():
                regret_sum[k][a] += v
        for k, amap in local_strategy.items():
            for a,v in amap.items():
                strategy_sum[k][a] += v

def export_policy():
    avg = {}
    for k, amap in strategy_sum.items():
        s = sum(amap.values())
        if s > 0:
            avg[str(k)] = {a: round(v/s, 3) for a, v in amap.items()}

    with open("policy_embed.py", "w") as f:
        f.write("POLICY = {\n")
        first = True
        for k, amap in avg.items():
            if not first:
                f.write(",\n")
            first = False
            f.write(f'    "{k}": {json.dumps(amap)}')
        f.write("\n}\n")

    print(f"Exported {len(avg)} states.")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    total_iters = 1000000
    cores = mp.cpu_count()
    iters_per = total_iters // cores

    print(f"Using {cores} cores")

    with mp.Pool(cores) as pool:
        results = pool.starmap(worker,
                               [(iters_per, 42+i) for i in range(cores)])

    merge(results)
    export_policy()

    print("Training complete.")