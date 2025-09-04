sudo apt-get update -y

python3 -m venv cllm2

pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu128

python3.10 -m venv cllm2
source cllm2/bin/activate
pip install -r requirements_gb200.txt