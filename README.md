# adrobo-CRANE-X7

CRANE-X7 マニピュレータを ManiSkill3/Genesis 上で動かし、DreamerV2 で学習するための実験リポジトリです。CRANE-X7 の公式 URDF/メッシュ、Genesis ベースの軽量環境、DreamerV2 実装を含みます。

## リポジトリ構成
- `ManiSkill/` — [haosulab/ManiSkill](https://github.com/haosulab/ManiSkill) のサブモジュール。Genesis を含む依存関係のセットアップ手順もこちらを参照。
- `crane_x7_description/` — 株式会社アールティ提供の CRANE-X7 記述（非商用ライセンス）。詳細は「ライセンス」セクションを参照。
- `simulation/` — Genesis 環境とツール類。
  - `envs/custom_env.py` — CRANE-X7 用の軽量 Genesis 環境。
  - `train/train.py` — DreamerV2 での学習エントリポイント。
  - `tools/record_policy.py` — ランダムポリシーのロールアウトを動画保存するユーティリティ。
- `dreamer_v2/` — 本リポジトリ用に調整した DreamerV2 実装。東京大学weblab（松尾・岩澤研究室, Matsuo Lab）の「世界モデル」講座 第1回コンペ（状態空間モデル (1), https://weblab.t.u-tokyo.ac.jp/lecture/course-list/world-model/ ／世界モデル - 東京大学松尾・岩澤研究室（松尾研）- Matsuo Lab）を参考にし、運営許可を得た上で利用しています。
- `docker/` — GPU 前提のベースイメージ (`docker/Dockerfile`)。
- `daydreamer-main/`, `dreamerv2-main/` — 参考実装のスナップショット。

## セットアップ
1. リポジトリ取得
   ```bash
   git clone https://github.com/<your-account>/adrobo-CRANE-X7.git
   cd adrobo-CRANE-X7
   git submodule update --init --recursive
   ```
2. 実行環境を用意  
   - 推奨: 付属の Dockerfile を使う  
     ```bash
     docker build -t adrobo-crane-x7 -f docker/Dockerfile .
     docker run --gpus all -it --rm -v $PWD:/workspace adrobo-crane-x7
     ```  
   - ローカル構築の場合は Python 3.9 相当を用意し、`mani-skill==3.0.0b21`（Dockerfileと同版）、`torch`、`genesis` などを ManiSkill の README に従ってインストールしてください。
3. 必要に応じてシーン資産をダウンロード  
   ```bash
   python -m mani_skill.utils.download_asset --list "scene"
   python -m mani_skill.utils.download_asset ReplicaCAD
   ```
4. 動作確認  
   ランダムポリシーを 1 エピソード実行し動画保存します。  
   ```bash
   python simulation/tools/record_policy.py --output videos/preview.mp4 --device cpu --steps 200
   ```

## 学習の始め方
- DreamerV2 のデフォルト設定は `dreamer_v2/config.py` を参照。
- CRANE-X7 環境で学習を回す例:
  ```bash
  python simulation/train/train.py --iter 1000 --device cuda --image_size 64 --save_path runs/dreamer_agent.pth
  ```
  主なオプション:
  - `--control_mode` : `delta_xy`（水平） / `delta_xyz`（3D 移動）
  - `--show_viewer` : Genesis ビューアを表示
  - `--record` / `--video_path` : 学習中のカメラ映像を保存

## crane_x7_description のライセンスと再配布
- `crane_x7_description/` は株式会社アールティの「非商用使用許諾規約」に基づくデータを含みます。利用前に `crane_x7_description/LICENSE` を必ず確認してください。商用利用が必要な場合は株式会社アールティへお問い合わせください。
- 再配布する場合はオリジナルリポジトリ（https://github.com/rt-net/crane_x7_description）へのクレジットと付属ライセンス文書を同梱してください。
- URDF やメッシュを変更・追加した場合は、変更点を README やコミットメッセージに明記してください。

## コントリビュート
Issue や Pull Request を歓迎します。大きな変更や仕様検討は事前に Issue でディスカッションしてください。
