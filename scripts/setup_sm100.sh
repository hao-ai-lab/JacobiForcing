python3 -m venv cllm2
source cllm2/bin/activate

pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu128

cd /home/lah003/workspace/flash-attention
MAX_JOBS=8 python3 setup.py install

pip install -r requirements_gb200.txt