class Config:
    def __init__(self, **kwargs):
        # コメントアウトされている値は，元実装のハイパーパラメータの値
        # data settings
        self.buffer_size = 100_000  # バッファにためるデータの上限
        self.batch_size = 16  # 50  # 学習時のバッチサイズ
        self.seq_length = 50  # 各バッチの系列長
        self.imagination_horizon = 10  # 15  # 想像上の軌道の系列長

        # model dimensions
        self.state_dim = 20  # 32  # 確率的な状態の次元数
        self.num_classes = 20  # 32  # 確率的な状態のクラス数（離散表現のため）
        self.rnn_hidden_dim = 200  # 600  # 決定論的な状態の次元数
        self.mlp_hidden_dim = 200  # 400  # MLPの隠れ層の次元数

        # learning params
        self.model_lr = 2e-4  # world model(transition / prior / posterior / discount / image predictor)の学習率
        self.actor_lr = 4e-5  # actorの学習率
        self.critic_lr = 1e-4  # criticの学習率
        self.epsilon = 1e-5  # optimizerのepsilonの値
        self.weight_decay = 1e-6  # weight decayの係数
        self.gradient_clipping = 100  # 勾配クリッピング
        self.kl_scale = 0.1  # kl lossのスケーリング係数
        self.kl_balance = 0.8  # kl balancingの係数(fix posterior)
        self.actor_entropy_scale = 1e-3  # entropy正則化のスケーリング係数
        self.slow_critic_update = 100  # target critic networkの更新頻度
        self.reward_loss_scale = 1.0  # reward lossのスケーリング係数
        self.discount_loss_scale = 1.0  # discount lossのスケーリング係数
        self.update_freq = 80  # 4

        # lambda return params
        self.discount = 0.995  # 割引率
        self.lambda_ = 0.95  # lambda returnのパラメータ

        # learning period settings
        self.iter = 600  # 総ステップ数（最初のランダム方策含む）
        self.seed_iter = 300  # 事前にランダム行動で探索する回数
        self.eval_freq = 5  # 評価頻度（エピソード）
        self.eval_episodes = 5  # 評価に用いるエピソード数

cfg = Config()