from fastapi import FastAPI
import pandas as pd
from env.environment import DataEnv
from env.models import Action

app = FastAPI()

df = pd.read_csv("dataset.csv")
env = DataEnv(df)

# RESET
@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.dict()}

# STEP
@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, _ = env.step(act)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }

# STATE
@app.get("/state")
def get_state():
    return {"state": env.state.dict()}
