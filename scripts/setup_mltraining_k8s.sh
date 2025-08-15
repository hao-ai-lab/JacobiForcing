sudo apt-get update -y
sudo apt-get install python3.10 python3.10-venv -y
python3.10 -m venv cllm2
source cllm2/bin/activate
pip install -r requirements.txt