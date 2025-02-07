conda create -n ppillars python=3.8 -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y
conda install numba::numba -y
conda install conda-forge::opencv -y
pip install open3d -U