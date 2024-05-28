FROM ubuntu:20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/home/user/.local/bin:${PATH}"
ENV nnUNet_results="/opt/algorithm/checkpoint/"
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw_data_base"
ENV nnUNet_preprocessed="/opt/algorithm/preproc"
ENV MKL_SERVICE_FORCE_INTEL=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    zip \
    unzip \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Add user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} user && useradd -u ${UID} -g user -m --no-log-init -r -g user user

# Create necessary directories and set permissions
RUN mkdir -p /opt/app /input /output /opt/algorithm/checkpoint/nnUNet \
    && chown -R user:user /opt/app /input /output /opt/algorithm/checkpoint/nnUNet

# Switch to user
USER user
WORKDIR /opt/app

# Install Python packages
RUN python3 -m pip install --user -U pip
RUN python3 -m pip install --user pip-tools
RUN python3 -m pip install --upgrade pip

# Install PyTorch and related packages
RUN python3 -m pip install --user torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Copy nnUNet and install
COPY --chown=user:user nnUNet/ /opt/app/nnUNet/
RUN python3 -m pip install --user -e nnUNet

# Copy requirements and install
COPY --chown=user:user requirements.txt /opt/app/
RUN python3 -m pip install --user -r requirements.txt

# Copy checkpoint and extract
COPY --chown=user:user nnUNetTrainer__nnUNetPlans__3d_fullres.zip /opt/algorithm/checkpoint/nnUNet/
RUN python3 -c "import zipfile; import os; zipfile.ZipFile('/opt/algorithm/checkpoint/nnUNet/nnUNetTrainer__nnUNetPlans__3d_fullres.zip').extractall('/opt/algorithm/checkpoint/nnUNet/')"

# Copy custom scripts
COPY --chown=user:user custom_algorithm.py /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user calc_dice.py /opt/app/

# Launch the script
ENTRYPOINT ["python3", "-m", "process"]
