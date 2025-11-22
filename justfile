# FlashVSR v1.1 セットアップ & テスト用 justfile

# デフォルトレシピ（ヘルプ表示）
default:
    @just --list

# 完全セットアップ（依存関係 + Block-Sparse + モデル）
setup: install-deps install-block-sparse download-models
    @echo ""
    @echo "========================================="
    @echo "✓ セットアップ完了！"
    @echo "========================================="
    @echo ""
    @echo "次のステップ:"
    @echo "  just test-all        - テストを実行"
    @echo "  just run-inference   - 推論テストを実行"

# Python依存関係のインストール
install-deps:
    @echo "========================================="
    @echo "Python依存関係をインストール中..."
    @echo "========================================="
    uv sync
    @echo "✓ 依存関係のインストール完了"

# Block-Sparse Attentionのビルド・インストール
install-block-sparse:
    @echo "========================================="
    @echo "Block-Sparse Attentionをローカルwhlからインストール中..."
    @echo "BLOCK_SPARSE_WHL 環境変数で whl のパスを指定してください。"
    @echo "========================================="
    @if [ -z "$${BLOCK_SPARSE_WHL:-}" ]; then \
        echo "✗ BLOCK_SPARSE_WHL が未指定です。例:"; \
        echo "  BLOCK_SPARSE_WHL=./block_sparse_attn-0.0.1+cu124torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl just install-block-sparse"; \
        exit 1; \
    fi
    @if [ ! -f "$${BLOCK_SPARSE_WHL}" ]; then \
        echo "✗ BLOCK_SPARSE_WHL で指定されたファイルが存在しません: $${BLOCK_SPARSE_WHL}"; \
        exit 1; \
    fi
    uv pip install --no-index "$${BLOCK_SPARSE_WHL}"

# FlashVSR v1.1モデルのダウンロード
download-models:
    @echo "========================================="
    @echo "FlashVSR v1.1モデルをダウンロード中..."
    @echo "（数分〜数十分かかります）"
    @echo "========================================="
    chmod +x scripts/download_model_v1.1.sh
    bash scripts/download_model_v1.1.sh

# 基本テスト（GPU/モデル不要）
test:
    @echo "========================================="
    @echo "基本テストを実行中..."
    @echo "========================================="
    pytest tests/ -v -m "not gpu and not model and not slow"

# 環境テスト
test-env:
    @echo "========================================="
    @echo "環境テストを実行中..."
    @echo "========================================="
    pytest tests/test_environment.py -v -m "not model"

# GPUテスト
test-gpu:
    @echo "========================================="
    @echo "GPUテストを実行中..."
    @echo "========================================="
    pytest tests/test_environment.py::TestCUDAEnvironment -v
    pytest tests/test_environment.py::TestPyTorchEnvironment -v

# 全テスト（遅いテスト除く）
test-all:
    @echo "========================================="
    @echo "全テストを実行中（遅いテスト除く）..."
    @echo "========================================="
    pytest tests/ -v -m "not slow"

# モデルファイル存在確認テスト
test-models:
    @echo "========================================="
    @echo "モデルファイルテストを実行中..."
    @echo "========================================="
    pytest tests/test_environment.py::TestModelFiles -v

# 推論テスト実行
run-inference:
    @echo "========================================="
    @echo "推論テストを実行中..."
    @echo "========================================="
    chmod +x scripts/test_inference.py
    python scripts/test_inference.py

# カスタムフレーム数で推論テスト
run-inference-frames frames="16":
    @echo "========================================="
    @echo "推論テスト実行（フレーム数: {{frames}}）..."
    @echo "========================================="
    chmod +x scripts/test_inference.py
    python scripts/test_inference.py --max-frames {{frames}}

# クリーンアップ
clean:
    @echo "========================================="
    @echo "クリーンアップ中..."
    @echo "========================================="
    rm -rf build/Block-Sparse-Attention/build
    rm -rf build/Block-Sparse-Attention/dist
    rm -rf build/Block-Sparse-Attention/*.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    @echo "✓ クリーンアップ完了"

# 完全クリーンアップ（ビルドディレクトリも削除）
clean-all: clean
    @echo "完全クリーンアップ中..."
    rm -rf build/
    rm -rf results/
    rm -rf .pytest_cache/
    @echo "✓ 完全クリーンアップ完了"

# 環境情報表示
info:
    @echo "========================================="
    @echo "環境情報"
    @echo "========================================="
    @echo "Python version:"
    @python --version
    @echo ""
    @echo "PyTorch version:"
    @python -c "import torch; print(f'  {torch.__version__}')" 2>/dev/null || echo "  未インストール"
    @echo ""
    @echo "CUDA available:"
    @python -c "import torch; print(f'  {torch.cuda.is_available()}')" 2>/dev/null || echo "  確認不可"
    @echo ""
    @echo "CUDA version:"
    @python -c "import torch; print(f'  {torch.version.cuda}')" 2>/dev/null || echo "  確認不可"
    @echo ""
    @echo "Block-Sparse Attention:"
    @python -c "import block_sparse_attn; print('  インストール済み')" 2>/dev/null || echo "  未インストール"
    @echo ""
    @echo "GPU Device:"
    @python -c "import torch; print(f'  {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "  確認不可"
    @echo ""
    @echo "GPU Memory:"
    @python -c "import torch; print(f'  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')" 2>/dev/null || echo "  確認不可"
    @echo ""
    @echo "モデルファイル:"
    @ls -lh FlashVSR/examples/WanVSR/FlashVSR-v1.1/ 2>/dev/null | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' || echo "  未ダウンロード"

# FlashVSR公式スクリプト実行（Tiny版）
run-official-tiny:
    @echo "========================================="
    @echo "FlashVSR公式スクリプト実行（Tiny版）"
    @echo "========================================="
    cd FlashVSR/examples/WanVSR && python infer_flashvsr_v1.1_tiny.py

# FlashVSR公式スクリプト実行（Full版）
run-official-full:
    @echo "========================================="
    @echo "FlashVSR公式スクリプト実行（Full版）"
    @echo "========================================="
    cd FlashVSR/examples/WanVSR && python infer_flashvsr_v1.1_full.py

# FlashVSR公式スクリプト実行（Long Video版）
run-official-long:
    @echo "========================================="
    @echo "FlashVSR公式スクリプト実行（Long Video版）"
    @echo "========================================="
    cd FlashVSR/examples/WanVSR && python infer_flashvsr_v1.1_tiny_long_video.py

# GPU メモリ使用状況を表示
gpu-stats:
    @echo "========================================="
    @echo "GPU メモリ使用状況"
    @echo "========================================="
    nvidia-smi

# セットアップ状況チェック
check:
    @echo "========================================="
    @echo "セットアップ状況チェック"
    @echo "========================================="
    @echo ""
    @echo "✓ Python環境:"
    @python --version || echo "  ✗ Python未インストール"
    @echo ""
    @echo "✓ uv:"
    @uv --version || echo "  ✗ uv未インストール"
    @echo ""
    @echo "✓ PyTorch:"
    @python -c "import torch; print(f'  バージョン: {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch未インストール"
    @echo ""
    @echo "✓ CUDA:"
    @python -c "import torch; print(f'  利用可能: {torch.cuda.is_available()}'); print(f'  バージョン: {torch.version.cuda}')" 2>/dev/null || echo "  ✗ CUDA確認不可"
    @echo ""
    @echo "✓ Block-Sparse Attention:"
    @python -c "import block_sparse_attn; print('  インストール済み')" 2>/dev/null || echo "  ✗ 未インストール（just install-block-sparse を実行）"
    @echo ""
    @echo "✓ FlashVSRリポジトリ:"
    @test -d FlashVSR && echo "  存在する" || echo "  ✗ 存在しない"
    @echo ""
    @echo "✓ モデルファイル:"
    @test -d FlashVSR/examples/WanVSR/FlashVSR-v1.1 && echo "  ダウンロード済み" || echo "  ✗ 未ダウンロード（just download-models を実行）"
    @echo ""
    @test -d FlashVSR/examples/WanVSR/FlashVSR-v1.1 && ( \
        test -f FlashVSR/examples/WanVSR/FlashVSR-v1.1/LQ_proj_in.ckpt && echo "  ✓ LQ_proj_in.ckpt" || echo "  ✗ LQ_proj_in.ckpt"; \
        test -f FlashVSR/examples/WanVSR/FlashVSR-v1.1/TCDecoder.ckpt && echo "  ✓ TCDecoder.ckpt" || echo "  ✗ TCDecoder.ckpt"; \
        test -f FlashVSR/examples/WanVSR/FlashVSR-v1.1/Wan2.1_VAE.pth && echo "  ✓ Wan2.1_VAE.pth" || echo "  ✗ Wan2.1_VAE.pth"; \
        test -f FlashVSR/examples/WanVSR/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors && echo "  ✓ diffusion_pytorch_model_streaming_dmd.safetensors" || echo "  ✗ diffusion_pytorch_model_streaming_dmd.safetensors" \
    ) || true
