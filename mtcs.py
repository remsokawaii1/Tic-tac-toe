import numpy as np
import copy

def select_move(cur_state, remain_time):
    # return minimaxCutOff(cur_state)
    return monteCarlo(cur_state)

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