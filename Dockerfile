FROM pytorch/pytorch

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools && python -m pip install --upgrade pip
COPY  --chown=user:user  nnUNet/ /opt/app/nnUNet/
RUN python -m pip install -e nnUNet
#RUN python -m pip uninstall -y scipy
#RUN python -m pip install --user --upgrade scipy

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt


# This is the checkpoint file, uncomment the line below and modify /local/path/to/the/checkpoint to your needs
COPY --chown=user:user nnUNetTrainer__nnUNetPlans__3d_fullres.zip /opt/algorithm/checkpoint/nnUNet/
RUN python -c "import zipfile; import os; zipfile.ZipFile('/opt/algorithm/checkpoint/nnUNet/nnUNetTrainer__nnUNetPlans__3d_fullres.zip').extractall('/opt/algorithm/checkpoint/nnUNet/')"

COPY --chown=user:user custom_algorithm.py /opt/app/
COPY --chown=user:user process.py /opt/app/

# COPY --chown=user:user weights /opt/algorithm/checkpoint
ENV nnUNet_results="/opt/algorithm/checkpoint/"
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw_data_base"
ENV nnUNet_preprocessed="/opt/algorithm/preproc"
# ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=64(nope!)
# ENV nnUNet_def_n_proc=1

#ENTRYPOINT [ "python3", "-m", "process" ]

ENV MKL_SERVICE_FORCE_INTEL=1

# Launches the script
ENTRYPOINT python -m process $0 $@
