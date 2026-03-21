# TGC Model GitHub リポジトリ整備 — Claude Code 引継ぎサマリー

## 背景

あるまは高校1年生の独立研究者。計算論的精神医学のジャーナル「Computational Psychiatry」にTGC（Thermostatic Gain Control）モデルの理論論文（v29）を投稿済み。論文は確率的カスプ・カタストロフィー理論に基づき、Wilson–Cowan回路からLangevin動力学を導出し、認知過負荷下の突発的パフォーマンス崩壊を説明する。

GitHubリポジトリ（https://github.com/mugityateishoku/TGC-Model）のコードが論文v29と整合しておらず、査読者がリポジトリを見た時に信頼性を損なうリスクがあるため、整備を行った。

## 完了済み作業

### 1. リポジトリ構造の再設計
```
TGC-Model/
├── README.md                    ← v29論文と整合済み（英語）
├── requirements.txt             ← 依存パッケージ（mne, fooof, statsmodels含む）
├── LICENSE                      ← MIT
├── .gitignore                   ← データ・生成物除外
├── simulation/
│   ├── tgc_langevin.py          ← 【新規作成】確率的Langevin動力学（論文Eq.1）
│   └── cusp_deterministic.py    ← 旧simulation.py（docstring更新済み）
├── analysis/
│   ├── study1_ds003838.py       ← kaiseki7.py（英語化済み）
│   ├── study1_pupil.py          ← tgc_pupil_study1.py（パス修正済み）
│   ├── study1_eeg.py            ← tgc_eeg_study1.py（パス修正済み）
│   ├── study2_abide.py          ← kaiseki4.py（docstring英語化・パス修正済み）
│   ├── study2_abide_fdr.py      ← kaiseki6.py
│   ├── study3_sfari.py          ← Tgc-sfari-study.py（docstring英語化・パス修正済み）
│   ├── study4_cogbci.py         ← tgcstudy2.py（docstring英語化・パス修正済み）
│   └── study1_supplementary/
│       ├── gmm_bimodality.py        ← kaiseki.py
│       ├── pupil_single_subject.py  ← kaiseki2.py
│       ├── hysteresis_prev_condition.py ← kaiseki3.py
│       ├── rt_iiv_overheating.py    ← kaiseki5.py
│       └── pipeline_v1.py           ← kaiseki1.py
├── figures/
└── supplementary/
```

### 2. 修正内容
- **README.md**: 旧版（autism-ADHD spectrum, MDP）→ v29論文（general cognitive overload, Langevin dynamics）に完全書き換え
- **全スクリプトのハードコードWindowsパス**: `r"F:\mat\..."` → 環境変数 `os.environ.get(...)` に統一
- **日本語docstring/コメント**: 主要部分を英語化（study1_supplementary/内は未処理）
- **tgc_langevin.py（新規）**: 論文Eq.1のEuler–Maruyamaシミュレーション、3つの図生成
  - Langevin trajectories under load ramp（Predictions 1-3）
  - Overheating demo（Prediction 4: noise-induced transition）
  - Stationary distribution（bistability）

### 3. 未修正（Claude Codeで対応すべき）
- **study1_supplementary/ 内の5ファイル**: 日本語コメントそのまま、パスも`ds003838-download`ハードコード
- **abide_study.py**: 実はEEG study1の別版でABIDEコードではなかった（リポジトリには含めていない）
- **study1_pupil.pyのdocstring**: 途中で切れている（`str_replace`が中断された）

## リポジトリへの反映手順

tarball `/home/claude/TGC-Model-repo.tar.gz` の中身を既存リポジトリにコピーしてpush:

```bash
cd /path/to/TGC-Model
# 既存ファイルをバックアップ
git checkout -b backup-pre-restructure
git push origin backup-pre-restructure
git checkout main

# 新しいファイル構造を展開
tar xzf TGC-Model-repo.tar.gz --strip-components=1

# 旧ファイル削除（必要に応じて）
git rm simulation.py README.md  # 旧ファイル名

git add -A
git commit -m "Restructure repo to match v29 paper

- Update README to reflect v29 paper (general cognitive overload framework)
- Add stochastic Langevin simulation (tgc_langevin.py, Eq. 1)
- Organize analysis scripts by study (1-4) matching paper sections
- Remove hardcoded Windows paths, use env vars
- Translate docstrings and key comments to English
- Add .gitignore, LICENSE, requirements.txt"

git push origin main
```

## 残りのTODO（優先順）

### 最優先（査読者が見る可能性あり）
1. **study1_pupil.pyのdocstring修正** — 途中で切れている
2. **study1_supplementary/ の英語化** — 5ファイルの日本語コメント
3. **全スクリプトの動作確認** — importエラー、パス不整合がないか

### 高優先（論文のData Availabilityセクションで言及）
4. **simulation/identifiability_gate.py** — Section 4のidentifiability gateシミュレーション（parameter recovery r≥0.8, confusion matrix diagonal ≥80%）。論文で言及しているが未実装
5. **simulation/model_comparison.py** — TGC vs DDM vs HMM vs Hopf vs Neural-field の5モデル弁別。READMEに書いたが未実装

### 中優先
6. **supplementary/ の中身** — S1 (Wilson-Cowan導出), S2 (confusion matrices), preregistration templateのMarkdown版。空ディレクトリのまま
7. **figures/ の生成** — tgc_langevin.py を実行して図を生成し、リポジトリに含めるか

## 論文の状態

- **投稿先**: Computational Psychiatry (cpsyjournal.org)
- **投稿状態**: 投稿済み、査読待ち
- **想定ターンアラウンド**: 数週間〜2ヶ月
- **desk reject確率**: 40-50%（"Independent Researcher" + 実証データなし + 小規模ジャーナル）
- **次の論文**: アイデア段階（テーマ未確定）

## 技術的注意

- 環境変数でパスを設定する設計にした:
  - `DS003838_DIR` — Study 1 データ
  - `ABIDE_DATA_DIR` + `ABIDE_PHENOTYPE` — Study 2 データ
  - `SFARI_DATA_DIR` — Study 3 データ
  - `COGBCI_DATA_DIR` — Study 4 データ
  - `TGC_FIGURES_DIR` — 図の出力先
- Study 4の`study4_cogbci.py`は`fooof`（旧名）をインポートしている。`specparam`（新名）に移行する場合はimport文の修正が必要
- `cusp_deterministic.py`は決定論的（σ=0）で、`tgc_langevin.py`が確率的（σ>0）。両方残す設計
