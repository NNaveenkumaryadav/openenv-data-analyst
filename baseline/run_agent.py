import pandas as pd
from env.environment import DataEnv
from env.models import Action

df = pd.read_csv("dataset.csv")
env = DataEnv(df)

env.reset()

actions = [
    Action(type="clean_missing", params={}),
    Action(type="filter", params={"key":"region","value":"South"}),
    Action(type="group_by", params={"column":"category"})
]

done=False
for a in actions:
    obs, r, done, _ = env.step(a)

print("Reward:", r)
print("Preview:", obs.preview)
