# FlashVSR v1.1 セットアップ手順書

このドキュメントでは、FlashVSR v1.1をuv環境でセットアップし、CUDA 11.8環境で動作確認する手順を説明します。

## 環境要件

### 必須要件
- **Python**: 3.11
- **CUDA**: 11.8（または互換性のあるバージョン）
- **GPU**: NVIDIA GPU（VRAM 8GB以上推奨）
  - 公式推奨: NVIDIA A100/A800（Ampere アーキテクチャ）
  - 他のGPUでも動作可能ですが、Block-Sparse Attentionの性能が異なる可能性があります
- **Git LFS**: モデルダウンロードに必要

### システム要件
- Linux（Ubuntu 20.04以降推奨）
- 十分なディスク容量（モデルファイル約10GB + ビルド時の一時ファイル）
- ビルド時の十分なメモリ（Block-Sparse Attentionのビルドに必要）

## セットアップ手順

### 1. リポジトリ構造の確認

```bash
cd /home/inaho-omen/Project/flashvsr_sandbox
ls -la
```

以下の構造になっていることを確認：
```
flashvsr_sandbox/
├── FlashVSR/               # FlashVSRリポジトリ（サブディレクトリ）
├── pyproject.toml          # uv環境設定
├── justfile                # タスクランナー設定
├── scripts/                # セットアップスクリプト
│   ├── install_block_sparse_attention.sh
│   ├── download_model_v1.1.sh
│   └── test_inference.py
├── tests/                  # pytestテストコード
│   ├── conftest.py
│   ├── test_environment.py
│   └── test_inference.py
└── README_SETUP.md         # このファイル
```

### 2. クイックスタート（justfile使用）

**just**がインストールされている場合、以下のコマンドで簡単にセットアップできます。

#### 2.1 justのインストール（未インストールの場合）

```bash
# Cargo経由でインストール
cargo install just

# またはシステムパッケージマネージャー
# Ubuntu/Debian
sudo apt install just

# または直接ダウンロード
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
```

#### 2.2 利用可能なコマンドを確認

```bash
just --list
```

#### 2.3 一括セットアップ

```bash
# 完全セットアップ（依存関係 + Block-Sparse + モデル）
just setup

# または手動で段階的に
just install-deps           # Python依存関係
just install-block-sparse   # Block-Sparse Attention
just download-models        # v1.1モデル
```

#### 2.4 セットアップ状況の確認

```bash
# セットアップ状況をチェック
just check
```

#### 2.5 テストと推論実行

```bash
# 全テスト実行
just test-all

# 推論テスト実行
just run-inference

# 環境情報表示
just info
```

**以降のセクションは手動でのセットアップ手順です。justを使用する場合は不要です。**

---

### 3. uv環境のセットアップ（手動）

#### 3.1 uvのインストール（未インストールの場合）

```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# パスを通す
source $HOME/.cargo/env
```

または **just を使用**:
```bash
just install-deps
```

#### 3.2 Python環境の作成と依存関係のインストール（手動）

```bash
# プロジェクトルートに移動
cd /home/inaho-omen/Project/flashvsr_sandbox

# uv環境を作成（Python 3.11を使用）
uv venv --python 3.11

# 環境をアクティベート
source .venv/bin/activate

# 依存関係をインストール（CUDA 11.8用）
uv sync
```

**注意**: `pyproject.toml`でCUDA 11.8用のPyTorchを指定しています。インストールには数分かかる場合があります。

#### 3.3 インストールの確認

```bash
# Pythonバージョン確認
python --version  # Python 3.11.x と表示されるはず

# PyTorchとCUDAの確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

出力例：
```
PyTorch: 2.6.0+cu118
CUDA available: True
CUDA version: 11.8
```

### 4. Block-Sparse Attentionのビルドとインストール

Block-Sparse AttentionはFlashVSRの高速化に必須のコンポーネントです。

**just を使用**:
```bash
just install-block-sparse
```

**手動**:
```bash
# ビルドスクリプトに実行権限を付与
chmod +x scripts/install_block_sparse_attention.sh

# Block-Sparse Attentionをビルド・インストール
bash scripts/install_block_sparse_attention.sh
```

**注意**:
- ビルドには10〜30分程度かかります
- ビルド中はメモリを大量に消費します（16GB以上推奨）
- エラーが発生した場合は、メモリ不足の可能性があります

#### 4.1 ビルドの確認

```bash
# インポートテスト
python -c "import block_sparse_attn; print('✓ Block-Sparse Attention successfully installed!')"
```

### 5. FlashVSR v1.1モデルのダウンロード

**just を使用**:
```bash
just download-models
```

**手動**:
```bash
# ダウンロードスクリプトに実行権限を付与
chmod +x scripts/download_model_v1.1.sh

# v1.1モデルをダウンロード
bash scripts/download_model_v1.1.sh
```

**注意**:
- Git LFSが必要です（スクリプトが自動でインストールを試みます）
- モデルファイルは合計約10GBあります
- ダウンロードには数分〜数十分かかります

#### 5.1 ダウンロードの確認

**just を使用**:
```bash
just test-models
```

**手動**:
```bash
# モデルファイルの確認
ls -lh FlashVSR/examples/WanVSR/FlashVSR-v1.1/
```

以下のファイルが存在することを確認：
- `LQ_proj_in.ckpt`
- `TCDecoder.ckpt`
- `Wan2.1_VAE.pth`
- `diffusion_pytorch_model_streaming_dmd.safetensors`

### 6. 環境テスト（pytest）

pytestを使用して環境が正しくセットアップされているか確認します。

**just を使用**:
```bash
# セットアップ状況チェック
just check

# 全テスト実行
just test-all

# 環境情報表示
just info
```

**手動**:

#### 6.1 基本的な環境テスト（GPUなし）

```bash
# 環境テストを実行（GPUテストを除外）
pytest tests/test_environment.py -v -m "not gpu and not model"
```

#### 6.2 GPU環境テスト

```bash
# GPUテストを含む環境テスト
pytest tests/test_environment.py -v -m "not model"
```

#### 6.3 モデルファイルテスト

```bash
# モデルファイルの存在確認テスト
pytest tests/test_environment.py::TestModelFiles -v
```

#### 6.4 全テスト実行（推論テストは除く）

```bash
# 全テストを実行（遅いテストを除外）
pytest tests/ -v -m "not slow"
```

**テストのマーカー**:
- `gpu`: GPU必須のテスト
- `model`: モデルファイル必須のテスト
- `slow`: 時間のかかるテスト

### 7. 動作確認（推論テスト）

簡易版の推論スクリプトで動作確認を行います。

**just を使用**:
```bash
# 推論テスト実行
just run-inference

# GPU状態確認
just gpu-stats
```

**手動**:
```bash
# 推論テストスクリプトに実行権限を付与
chmod +x scripts/test_inference.py

# 推論テストを実行
python scripts/test_inference.py
```

**このスクリプトの動作**:
- 最小サイズのサンプル動画を使用
- フレーム数を16フレームに制限（メモリ節約）
- FlashVSR Tinyパイプラインで推論実行
- 結果を`results/`ディレクトリに保存

#### 7.1 出力の確認

```bash
# 出力動画の確認
ls -lh results/
```

出力例：
```
test_output_example4_seed0.mp4
```

### 8. 本格的な推論実行

動作確認が完了したら、FlashVSRの公式スクリプトで本格的な推論を実行できます。

**just を使用**:
```bash
# v1.1 Tinyモデル（メモリ効率版）
just run-official-tiny

# v1.1 Fullモデル（VAE含む完全版）
just run-official-full

# v1.1 Long Videoモデル（長尺動画用）
just run-official-long
```

**手動**:
```bash
# FlashVSRのWanVSRディレクトリに移動
cd FlashVSR/examples/WanVSR

# v1.1 Tinyモデルで推論
python infer_flashvsr_v1.1_tiny.py
```

**利用可能なスクリプト**:
- `infer_flashvsr_v1.1_tiny.py`: メモリ効率版（推奨）
- `infer_flashvsr_v1.1_full.py`: 完全版（VAE含む）
- `infer_flashvsr_v1.1_tiny_long_video.py`: 長尺動画用

## トラブルシューティング

### CUDA関連のエラー

```
RuntimeError: CUDA out of memory
```

**対処法**:
1. 推論時のフレーム数を減らす
2. 解像度を下げる
3. `sparse_ratio`を調整（デフォルト: 2.0）
4. より大きなVRAMを持つGPUを使用

### Block-Sparse Attentionのビルドエラー

```
error: command 'gcc' failed
```

**対処法**:
1. ビルドツールをインストール:
   ```bash
   sudo apt-get install build-essential
   ```
2. CUDA Toolkitがインストールされているか確認
3. メモリが十分か確認（16GB以上推奨）

### モデルダウンロードエラー

```
Git LFS not found
```

**対処法**:
```bash
# Git LFSをインストール
sudo apt-get install git-lfs
git lfs install

# 再度ダウンロード
bash scripts/download_model_v1.1.sh
```

### importエラー

```
ImportError: cannot import name 'FlashVSRTinyPipeline'
```

**対処法**:
1. FlashVSRがPythonパスに含まれているか確認
2. 依存関係が正しくインストールされているか確認:
   ```bash
   uv sync --reinstall
   ```

## GPU メモリ使用量の目安

推論時のおおよそのVRAM使用量：

| 解像度（出力） | フレーム数 | VRAM使用量（推定） |
|--------------|-----------|------------------|
| 512x512      | 16        | 6-8 GB           |
| 768x768      | 16        | 10-12 GB         |
| 1024x1024    | 16        | 14-18 GB         |
| 1280x768     | 16        | 12-16 GB         |

**注意**:
- 実際の使用量はGPUの種類、ドライバ、その他のプロセスにより変動します
- Block-Sparse Attentionの効果でメモリ使用量は削減されています

## パラメータ調整

### sparse_ratio（スパース比率）
- デフォルト: `2.0`
- 推奨範囲: `1.5` - `2.0`
- `1.5`: 高速だが安定性が低い
- `2.0`: 安定した結果（推奨）

### local_range（ローカルレンジ）
- デフォルト: `11`
- 推奨値: `9` または `11`
- `9`: シャープな詳細
- `11`: より安定した結果（推奨）

### scale（倍率）
- デフォルト: `4.0`（4倍超解像）
- **変更非推奨**: FlashVSRは4倍超解像に最適化されています

## 参考リンク

- **FlashVSR GitHub**: https://github.com/OpenImagingLab/FlashVSR
- **FlashVSR v1.1 Model**: https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
- **Block-Sparse Attention**: https://github.com/mit-han-lab/Block-Sparse-Attention
- **uv Documentation**: https://docs.astral.sh/uv/
- **just Documentation**: https://just.systems/

## ライセンスと引用

FlashVSRを使用する場合は、公式リポジトリのライセンスとcitation情報を確認してください。

---

**作成日**: 2025-11-21
**環境**: CUDA 11.8, Python 3.11, uv環境
**FlashVSR Version**: v1.1
