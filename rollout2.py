import numpy as np
import os, sys, glob
import gym #openAIGym
from hparams import HyperParams as hp

def rollout():
    env = gym.make("CarRacing-v2")

    #シーケンス長　各エピソード中に収集されるタイムステップの最大数を表す（アクション数やそれに対応する観測値の数を制限する）。
    seq_len = 1000 
    max_ep = hp.n_rollout #最大エピソード数
    feat_dir = hp.data_dir #特徴ディレクトリ（フォルダ）data_dir = 'datasets'

    os.makedirs(feat_dir, exist_ok=True) #指定されたパスのフォルダを作成

    for ep in range(max_ep):

        #観測値、アクション、報酬、次の観測値、エピソードが終了したかどうか(done)
        obs_lst, action_lst, reward_lst, next_obs_lst, done_lst = [], [], [], [], []
        obs = env.reset() #最初の観測値を取得
        done = False
        t = 0
        
        while not done and t < seq_len:  #エピソードが終了するか、シーケンス長に達するまで繰り返す
            action = env.action_space.sample() #環境のアクションスペースからランダムなアクションを選択
            next_obs, reward, done, _ = env.step(action) #選択されたアクションを実行し、次の観測値、報酬、doneを取得

            #タイムステップごとのデータを保存
            np.savez(
                os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep,t)),
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            #データをリストに追加
            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)

            #現在の観測値を更新
            obs = next_obs

            t += 1

        # エピソード全体のデータをファイルに保存
        np.savez(
            os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
            obs=np.stack(obs_lst, axis=0), # (T, H, W, C)
            action=np.stack(action_lst, axis=0), # (T, a)
            reward=np.stack(reward_lst, axis=0), # (T, 1)
            next_obs=np.stack(next_obs_lst, axis=0), #(T, H, W, C)
            done=np.stack(done_lst, axis=0), # (T, 1)
        )

if __name__ == '__main__':
    np.random.seed(hp.seed)
    rollout()
