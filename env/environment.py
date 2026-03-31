import pandas as pd
from .models import State, Action, Observation
from .reward import compute_reward

class DataEnv:
    def __init__(self, df):
        self.original = df
        self.reset()

    def reset(self):
        self.df = self.original.copy()
        self.state = State(dataset=self.df.to_dict('records'), current_step=0, max_steps=15)
        return self.state

    def step(self, action: Action):
        self.state.current_step += 1

        if action.type == "filter":
            k, v = action.params["key"], action.params["value"]
            self.df = self.df[self.df[k] == v]

        elif action.type == "clean_missing":
            self.df = self.df.fillna(0)

        elif action.type == "group_by":
            col = action.params["column"]
            self.df = self.df.groupby(col).mean(numeric_only=True).reset_index()

        elif action.type == "sort":
            col = action.params["column"]
            self.df = self.df.sort_values(by=col, ascending=False)

        self.state.dataset = self.df.to_dict('records')

        reward = compute_reward(self.state)
        done = self.state.current_step >= self.state.max_steps

        obs = Observation(preview=self.state.dataset[:5], message="ok")
        return obs, reward, done, {}
