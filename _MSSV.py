import numpy as np
import copy

def select_move(cur_state, remain_time):
    return minimaxCutOff(cur_state)
    # return monteCarlo(cur_state)

def minimaxCutOff(state, d = 6, cutoff = None, eval = None):
    player = state.player_to_move

    def max_value(state, alpha, beta, depth):
        if cutoff(state, depth):
            return eval(state)
        score = -np.inf
        for move in state.get_valid_moves:
            current = copy.deepcopy(state)
            current.act_move(move)
            score = max(score, min_value(current, alpha, beta, depth + 1))
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return score

    def min_value(state, alpha, beta, depth):
        if cutoff(state, depth):
            return eval(state)
        score = np.inf
        for move in state.get_valid_moves:
            current = copy.deepcopy(state)
            current.act_move(move)
            score = min(score, max_value(current, alpha, beta, depth + 1))
            if score <= alpha:
                return score
            beta = min(beta, score)
        return score

    cutoff = (cutoff or (lambda state, depth: depth > d or state.game_over))
    eval = eval or (lambda state: state.game_result(state.global_cells.reshape(3, 3)) * player if state.game_result(state.global_cells.reshape(3, 3)) else 0)
    best_score = -np.inf
    beta = np.inf
    best_move = None
    for move in state.get_valid_moves:
        current = copy.deepcopy(state)
        current.act_move(move)
        score = min_value(current, best_score, beta, 1)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move




class MCT_Node:
    def __init__(self, parent = None, state = None, U = 0, N = 0):
        self.__dict__.update(parent = parent, state = state, U = U, N = N)
        self.children = {}
        self.actions = None

def ucb(n, C = np.sqrt(2.0)):
    return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)

def monteCarlo(state, N = 10000):
    def select(n):
        if n.children:
            return select(max(n.children.keys(), key = ucb))
        else:
            return n

    def expand(n):
        if not n.children and not n.state.game_over:
            for action in n.state.get_valid_moves:
                newState = copy.deepcopy(n.state)
                newState.act_move(action)
                n.children = {MCT_Node(state = newState, parent = n) : action}
        return select(n)

    def simulate(state):
        player = state.player_to_move
        while not state.game_over and len(state.get_valid_moves) > 0:
            action = np.random.choice(state.get_valid_moves)
            state.act_move(action)
        v = state.game_result(state.global_cells.reshape(3, 3)) * player if state.game_result(state.global_cells.reshape(3, 3)) else 0
        return -v

    def backprop(n, utility):
        if utility > 0:
            n.U += utility
        if utility == 0:
            n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state = copy.deepcopy(state))
    for _ in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(child.state)
        backprop(child, result)

    max_state = max(root.children, key = lambda p : p.N)

    return root.children.get(max_state)