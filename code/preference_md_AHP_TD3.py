import argparse, os, yaml, random, math, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ----- CLI arguments -----
parser = argparse.ArgumentParser()
parser.add_argument("--csv",            type=str,   default="portfolio_policy.csv")
parser.add_argument("--ahp_cfg",        type=str,   default="ahp_weights.yml")
parser.add_argument("--pairs",          type=int,   default=20000)
parser.add_argument("--num_envs",       type=int,   default=4)
parser.add_argument("--total_steps",    type=int,   default=300_000)
parser.add_argument("--drift_scale",    type=float, default=1.5,
                    help="Multiplier on annual drifts")
parser.add_argument("--vol_penalty",    type=float, default=0.5,
                    help="Weight on annualized volatility penalty")
parser.add_argument("--dd_penalty",     type=float, default=0.2,
                    help="Weight on max drawdown penalty")
parser.add_argument("--lambda_start",   type=float, default=0.0,
                    help="Start of λ grid (rank weight)")
parser.add_argument("--lambda_end",     type=float, default=1.0,
                    help="End of λ grid (rank weight)")
parser.add_argument("--lambda_steps",   type=int,   default=11,
                    help="Number of λ grid points")
parser.add_argument("--patience",       type=int,   default=5,
                    help="Early-stop patience (evals)")
parser.add_argument("--eval_freq",      type=int,   default=10000,
                    help="Evaluation frequency (timesteps)")
args = parser.parse_args()

assert os.path.exists(args.csv), f"CSV file missing at {args.csv}"
if not os.path.exists(args.ahp_cfg):
    yaml.dump({"matrix":[[1,3,4],[1/3,1,1],[1/4,1,1]]},
              open(args.ahp_cfg,"w"))
AHP = np.array(yaml.safe_load(open(args.ahp_cfg))["matrix"])

# ----- Load & feature engineer -----
df = pd.read_csv(args.csv, parse_dates=["trade_date"])
print("Rows=", len(df), "Trajs=", df.traj_id.nunique())

# helper metrics
def ann_ret(x): return (1 + np.asarray(x)/100).prod()**(252/len(x)) - 1
def ann_vol(x): return np.std(x)*np.sqrt(252)/100
def max_dd(x):
    nav = np.cumprod(1 + np.asarray(x)/100)
    peak = np.maximum.accumulate(nav)
    return ((nav/peak) - 1).min() * 100
def dd_dur(x):
    s   = pd.Series(np.asarray(x))
    nav = (1 + s/100).cumprod(); peak = nav.cummax()
    return ((nav<peak).astype(int)
            .groupby(((nav<peak).astype(int).diff()!=0).cumsum())
            .transform("size").max())
def skew(x):
    a = np.asarray(x); return ((a - a.mean())**3).mean() / (a.std()**3 + 1e-9)

metrics = df.groupby("traj_id").agg(
    ann_ret=("daily_ret", ann_ret),
    ann_vol=("daily_ret", ann_vol),
    max_dd =("daily_ret", max_dd),
    dd_dur =("daily_ret", dd_dur),
    skew   =("daily_ret", skew)
)
mm = lambda c: (c - c.min())/(c.max() - c.min())
eigvec = np.linalg.eig(AHP)[1][:,0].real
w_ahp  = eigvec / eigvec.sum()
metrics["ahp_score"] = (
    w_ahp[0]*mm(metrics.ann_ret) +
    w_ahp[1]*(1-mm(metrics.ann_vol)) +
    w_ahp[2]*(1-mm(metrics.max_dd))
)

# ----- Preference Learning -----
ids     = list(metrics.index)
rand_ids= random.choices(ids, k=2*args.pairs)
pairs   = [(rand_ids[i], rand_ids[i+args.pairs])
           for i in range(args.pairs) if rand_ids[i]!=rand_ids[i+args.pairs]]
X, y, g = [], [], []
for gid,(i,j) in enumerate(pairs):
    Xi = metrics.loc[i, ["ann_ret","ann_vol","max_dd","dd_dur","skew"]].values
    Xj = metrics.loc[j, ["ann_ret","ann_vol","max_dd","dd_dur","skew"]].values
    X += [Xi, Xj]; y += [1, 0]; g += [gid, gid]

dtrain = lgb.Dataset(np.array(X), label=np.array(y), group=np.bincount(g))
ranker = lgb.train({"objective":"lambdarank","metric":"ndcg",
                    "learning_rate":0.05,"num_leaves":31},
                   dtrain, num_boost_round=200)
metrics["R_theta"] = ranker.predict(metrics[["ann_ret","ann_vol","max_dd","dd_dur","skew"]])

# quick NDCG check
n_samp      = min(1000, len(metrics))
s_true      = metrics.ahp_score.sample(n_samp, random_state=42).values.reshape(1,-1)
s_pred      = metrics.R_theta.sample(n_samp, random_state=42).values.reshape(1,-1)
print(f"Sample NDCG@{n_samp} =", ndcg_score(s_true, s_pred))

# ----- Custom Early‑Stop Callback (Sharpe) -----
class EarlyStopCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, patience, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env; self.eval_freq = eval_freq
        self.patience = patience; self.best = -np.inf; self.count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(20):
                obs, _ = self.eval_env.reset()
                done = False; ep_r = 0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = self.eval_env.step(act)
                    ep_r += r; done = term or trunc
                rewards.append(ep_r)
            mean_r, std_r = np.mean(rewards), np.std(rewards)
            sharpe = mean_r / (std_r + 1e-9)
            if self.verbose:
                print(f"[Eval@{self.num_timesteps}] mean={mean_r:.3f} std={std_r:.3f} sharpe={sharpe:.3f}")
            metric = sharpe
            if metric > self.best + 1e-3:
                self.best, self.count = metric, 0
            else:
                self.count += 1
            if self.count >= self.patience:
                if self.verbose:
                    print(f"Early stopping (no Sharpe improvement for {self.patience} evals)")
                return False
        return True

# ----- Portfolio Environment (state includes AHP & rank scores) -----
class PortEnv(gym.Env):
    metadata = {"render_modes":[]}
    def __init__(self, ranker, metrics_df,
                 cost=0.0005, ep_len=60,
                 ahp_alpha=1.0, rank_alpha=1.0,
                 vol_penalty=0.5, dd_penalty=0.2):
        super().__init__()
        self.ranker      = ranker
        self.metrics_df  = metrics_df
        self.cost        = cost
        self.ep_len      = ep_len
        self.ahp_alpha   = ahp_alpha
        self.rank_alpha  = rank_alpha
        self.vol_penalty = vol_penalty
        self.dd_penalty  = dd_penalty
        self.action_space      = spaces.Box(-0.1,0.1,(3,),np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,(6,),np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.traj_id = random.choice(list(self.metrics_df.index))
        self.df      = df[df.traj_id==self.traj_id].reset_index(drop=True)
        w0 = self.df.loc[0, ["w_SP500","w_Treasury","w_Gold","w_Cash"]].values/100
        self.w, self.t, self.rets = w0.astype(np.float32), 0, []
        self.ahp_s  = self.metrics_df.loc[self.traj_id, "ahp_score"]
        self.rank_s = float(self.ranker.predict(
                             [self.metrics_df.loc[self.traj_id, ["ann_ret","ann_vol","max_dd","dd_dur","skew"]].values]
                          )[0])
        return self._obs(), {}

    def _obs(self):
        return np.array([
            self.w[0], self.w[1], self.w[2],
            self.t/self.ep_len,
            self.ahp_s, self.rank_s
        ], dtype=np.float32)

    def step(self, act):
        act = np.clip(act, -0.1,0.1)
        self.w[:3] = np.clip(self.w[:3]+act,0,None)
        self.w[3]  = max(0,1-self.w[:3].sum())
        day_ret     = float(self.df.loc[self.t,"daily_ret"])
        self.rets.append(day_ret)
        reward      = day_ret - self.cost*np.abs(act).sum()*100
        self.t     += 1
        done        = (self.t>=self.ep_len)
        trunc       = False
        if done:
            combo = (self.rank_alpha*self.rank_s +
                     self.ahp_alpha*self.ahp_s)
            vol   = ann_vol(self.rets)
            dd    = max_dd(self.rets)
            reward += math.tanh(combo)
            reward -= self.vol_penalty * vol
            reward -= self.dd_penalty  * dd
        return self._obs(), reward, done, trunc, {}

# ----- Training & λ grid search -----
lambda_vals = np.linspace(args.lambda_start, args.lambda_end, args.lambda_steps)
results = []

for lam in lambda_vals:
    ahp_a  = 1.0 - lam
    rank_a = lam

    make_env_fn = lambda: PortEnv(
        ranker, metrics,
        ahp_alpha=ahp_a, rank_alpha=rank_a,
        vol_penalty=args.vol_penalty, dd_penalty=args.dd_penalty
    )
    vec_env = DummyVecEnv([make_env_fn]*args.num_envs)
    eval_env= PortEnv(
        ranker, metrics,
        ahp_alpha=ahp_a, rank_alpha=rank_a,
        vol_penalty=args.vol_penalty, dd_penalty=args.dd_penalty
    )

    n_act = vec_env.action_space.shape[-1]
    noise = NormalActionNoise(np.zeros(n_act), 0.02*np.ones(n_act))
    model = TD3("MlpPolicy", vec_env,
                learning_rate=3e-4, buffer_size=500_000,
                batch_size=2048, tau=0.005, gamma=0.99,
                train_freq=(1024,"step"), gradient_steps=1024,
                action_noise=noise,
                policy_kwargs={"net_arch":[128,128]},
                verbose=0)

    cb = EarlyStopCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        patience=args.patience,
        verbose=1
    )
    model.learn(total_timesteps=args.total_steps, callback=cb)

    def eval_once(env, mdl, runs=100):
        sc=[]
        for _ in range(runs):
            obs,_=env.reset(); done=False; rsum=0
            while not done:
                act,_=mdl.predict(obs,deterministic=True)
                obs,r,done,tr,_=env.step(act)
                rsum+=r
            sc.append(rsum)
        return np.mean(sc), np.std(sc)

    meanR,stdR = eval_once(eval_env, model, runs=200)
    results.append((lam, meanR, stdR))
    print(f"λ={lam:.2f} → mean={meanR:.3f} ± {stdR:.3f}")

# ----- Plot λ vs. Performance -----
df_res = pd.DataFrame(results, columns=["lambda","mean","std"])
plt.errorbar(df_res["lambda"], df_res["mean"],
             yerr=df_res["std"], marker="o", capsize=5)
plt.xlabel("λ (rank weight)")
plt.ylabel("Final Eval Reward ± Std")
plt.title("λ vs. TD3 Performance")
plt.grid(True)
plt.show()
