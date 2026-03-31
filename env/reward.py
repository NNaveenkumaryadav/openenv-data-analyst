def compute_reward(state):
    base = max(0, 1 - len(state.dataset)/300)
    penalty = state.current_step * 0.02
    return max(0.0, min(1.0, base - penalty))
