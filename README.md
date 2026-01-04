# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation

# Cài đặt git
apt update
apt install git -y

git clone -b claude/integrate-yescale-llm-01Eij6gMg1uSwfaLizjNQ1ih \
https://github.com/NgocMinh000/GFM.git
# Cài miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Thêm conda vào PATH thủ công
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
# Accept Terms of Service của Conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda init bash
source ~/.bashrc
# Tạo env 
conda create -n gfm-rag python=3.12 -y
conda activate gfm-rag
# Tải các tài nguyên tính toán 
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1 # Replace with your desired CUDA version
pip install gfmrag
# Gỡ faiss cũ
pip uninstall -y faiss faiss-cpu faiss-gpu
conda remove --force faiss -y
# Cài faiss 1.9.0 (hỗ trợ Python 3.12)
pip install faiss-cpu==1.9.0

# Chạy workflow tạo BKG
cd ~/GFM
python -m gfmrag.workflow.stage1_index_dataset
