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

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
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