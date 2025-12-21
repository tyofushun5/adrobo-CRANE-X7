# adrobo-CRANE-X7

## 概要
CRANE-X7 を Genesis 上で動作させ、DreamerV2 で学習させるための実験リポジトリです。CRANE-X7 の公式 URDF とメッシュ、Genesis ベースの環境、DreamerV2 の実装を含みます。

## 動作環境
- Ubuntu : 22.04 LTS / 24.04 LTS
- CUDA : 13.0
- Python : 3.10 系
- 依存ライブラリ: `requirements.txt` を参照

## セットアップ
1. リポジトリ取得
   ```bash
   git clone https://github.com/tyofushun5/adrobo-CRANE-X7.git
   cd adrobo-CRANE-X7
   git submodule update --init --recursive
   ```
2. 依存パッケージのインストール
   ```bash
   python -m pip install -r requirements.txt
   ```
   ※ `requirements.txt` は参考です。環境に合わせて適宜調整してください。

## 使用方法
### 学習済みポリシーの動画保存
学習済みポリシーで環境を 1 エピソード実行し、環境側の録画機能で動画を保存します（デフォルトは `simulation/train/dreamer_agent.pth`）。  
同時に、初期観測から RSSM の潜在状態をロールアウトし、デコーダで生成した想像映像を別ファイルに保存します。  
出力ファイルは `videos/preview.mp4` と `videos/preview_imagined.mp4` です。
```bash
python simulation/tools/record_policy.py
```
設定を変えたい場合は `simulation/tools/record_policy.py` の `DEFAULT_*` を編集してください。

### 学習
DreamerV2 のデフォルト設定は `dreamerv2/config.py` を参照してください。
```bash
python simulation/train/train.py
```
主に調整する設定（`dreamerv2/config.py`）:
- `iter` : 学習イテレーション数
- `device` / `sim_device` : 学習側 / シミュレーション側のデバイス
- `control_mode` : 操作モード（現状は `discrete_xyz`）
- `show_viewer` : Genesis ビューアを表示
- `record` / `video_path` : 学習中のカメラ映像を保存
- `log_dir` : TensorBoard のログ出力先
- `save_path` : チェックポイント保存先（`record_policy.py` の `--checkpoint` と合わせてください）

TensorBoard のログを見る:
```bash
tensorboard --logdir simulation/train/tensorborad
```

## リポジトリ構成
**ディレクトリ**
- `ManiSkill/` — テーブルアセット参照用のサブモジュール（`simulation/entity/table.py`）。
- `crane_x7_description/` — 株式会社アールティ提供の CRANE-X7 の記述パッケージ（非商用ライセンス）。詳細は「ライセンス・利用条件」セクションを参照。
- `simulation/` — Genesis 環境と学習用ツール。
- `dreamerv2/` — 本リポジトリ向けに調整した DreamerV2 の実装。

**主要スクリプト**
- `simulation/envs/custom_env.py` — CRANE-X7 用の Genesis 環境。
- `simulation/train/train.py` — DreamerV2 学習のエントリーポイント。
- `simulation/tools/record_policy.py` — 学習済みポリシーのロールアウト動画と、RSSM で生成した想像映像を保存するユーティリティ。

## ライセンス・利用条件
- 本リポジトリ本体は MIT License（`LICENSE`）です。
- `crane_x7_description/` は株式会社アールティの「非商用使用許諾規約」に基づくデータを含みます。利用前に `crane_x7_description/LICENSE` を必ず確認してください。
- `dreamerv2/` の実装は、東京大学 松尾・岩澤研究室の「世界モデル Deep Learning 応用講座 2025」第1回コンペ「状態空間モデル (1)」を参考にしており、運営側の許可を得て利用しています。参考: https://weblab.t.u-tokyo.ac.jp/lecture/course-list/world-model/

## 参考文献
- DreamerV2 の公式実装: https://github.com/danijar/dreamerv2
- DreamerV2 論文: https://arxiv.org/abs/2010.02193
