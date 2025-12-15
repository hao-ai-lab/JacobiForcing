sudo apt-get update -y
sudo apt install python3.12-venv

python3 -m venv cllm2_venv

pip uninstall torch -y && pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

source cllm2_venv/bin/activate
pip install -r requirements_gb200.txt
