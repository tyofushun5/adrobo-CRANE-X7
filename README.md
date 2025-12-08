# adrobo-CRANE-X7

CRANE-X7 マニピュレータを ManiSkill ベースでシミュレーション／学習するための実験リポジトリです。Dreamer などの学習コード、ロボット記述ファイル、各種サンプルスクリプトが含まれています。

## リポジトリ構成と依存関係
- `ManiSkill/` — [haosulab/ManiSkill](https://github.com/haosulab/ManiSkill) のサブモジュールです。初回セットアップ時は `git submodule update --init --recursive` を実行してください。
- `robot/crane_x7_description/` — 株式会社アールティが提供する CRANE-X7 のロボット記述一式を本リポジトリ内にベンダリングしています。詳細は以下の「ライセンスと再利用条件」を参照してください。
- `robot/crane_x7.urdf` など、プロジェクト固有に追加した URDF も `robot/` 配下で管理しています。

## セットアップ
```bash
git clone https://github.com/<your-account>/adrobo-CRANE-X7.git
cd adrobo-CRANE-X7
git submodule update --init --recursive
```
Python 依存関係や ManiSkill のビルド方法は ManiSkill サブモジュールの `README.md` を参照してください。

## crane_x7_description のライセンスと再利用条件
- ディレクトリ `robot/crane_x7_description/` には、株式会社アールティの「非商用使用許諾規約」および各種 OSS ライセンスのもとで提供されるデータが含まれます。必ず `robot/crane_x7_description/LICENSE` を確認し、非商用目的でのみ利用してください。商用利用が必要な場合は株式会社アールティへ別途お問い合わせください。
- 本プロジェクトにおいては、上記ディレクトリをサブモジュールではなく通常のファイルとして取り込み、`urdf/crane_x7_*.urdf` などを自由に追加できるようにしています。改変箇所を共有する際は README やコミットメッセージに変更点を明記してください。
- 同ディレクトリを再配布する場合は、オリジナルリポジトリ（https://github.com/rt-net/crane_x7_description）へのクレジットと付属ライセンス文書を必ず同梱してください。

## URDF を追加する際のメモ
1. 追加したい URDF（例: `crane_x7.urdf`）を `robot/crane_x7_description/urdf/` 以下に配置します。
2. 必要に応じて `robot/crane_x7_description/urdf/crane_x7.xacro` などから参照させます。
3. `git add robot/crane_x7_description` を実行して変更をステージし、通常どおりコミットします。

## コントリビュート
Issue や pull request は歓迎です。大きな変更や仕様面の相談がある場合は事前に issue でディスカッションしてください。
