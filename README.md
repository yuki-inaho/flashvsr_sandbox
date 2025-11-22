# FlashVSR Sandbox (v1.1) — Overview & Quick Start

このリポジトリは FlashVSR v1.1 の検証用ワークスペースです。公式コードは `FlashVSR/` サブディレクトリにあり、環境は uv / `.venv` で管理しています。詳細な日本語セットアップ手順は `README_SETUP.md` を参照してください。ここでは最低限の流れをまとめます。

## 1. 環境準備（uv/.venv）

```bash
cd /home/inaho/Project/flashvsr_sandbox
source .venv/bin/activate          # 既存の .venv を使用
python -m ensurepip                # pip がなければ有効化
python -m pip install -e FlashVSR  # diffsynth を開発インストール
python -m pip install modelscope   # diffsynth の downloader 依存
```

依存の全体像は `pyproject.toml`（torch 2.6.0/cu124 スタック）と `uv.lock` を参照。

## 2. モデル配置

Hugging Face から取得済みの v1.1 モデル一式が `FlashVSR/examples/WanVSR/FlashVSR-v1.1/` にあることを確認してください（`.safetensors`, `.ckpt`, `.pth`）。

## 3. Block-Sparse Attention

高速化には Block-Sparse Attention が必要です。CUDA 12.4 / torch 2.6 用のビルド済み whl がリポジトリ直下にあります。

```bash
uv pip install --no-index ./block_sparse_attn-0.0.1+cu124torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# もしくは just:
# BLOCK_SPARSE_WHL=./block_sparse_attn-0.0.1+cu124torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl just install-block-sparse
```

## 4. 推論の実行（v1.1 Tiny）

```bash
python FlashVSR/examples/WanVSR/infer_flashvsr_v1.1_tiny.py
```

出力動画は `FlashVSR/examples/WanVSR/results/` 配下に保存されます。GPU メモリを多く使用します（RTX 3060 で実績あり）。

## 5. 便利スクリプト（pytest や自作スクリプト用）

`scripts/flashvsr_env.py` を呼ぶとパスを自動で通せます。

```python
from scripts.flashvsr_env import add_paths
add_paths()  # sys.path に repo ルートと FlashVSR/ を追加
```

Pytest から `diffsynth` を直接 import したい場合などに利用してください。

## 6. 詳細手順・just タスク

- 詳細: `README_SETUP.md`
- just タスク（例）:
  - `just install-deps` / `just install-block-sparse` / `just download-models`
  - `just run-inference`, `just test-all`, `just info`

## 7. 注意

- FlashVSR は 4× 超解像前提で設計・学習されています。同解像度の復元（デブラー等）は現行モデルではサポートされません。用途に応じて別途モデルを用意してください。
