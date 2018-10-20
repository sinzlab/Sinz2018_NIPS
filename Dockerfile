FROM sinzlab/pytorch:v0.3.1-cuda9.1

RUN pip install git+https://github.com/circstat/pycircstat.git

RUN pip install jupyterlab && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

RUN pip install "git+https://github.com/atlab/attorch.git@pytorch0.3.1"

ADD . /src/Sinz2018_NIPS
RUN pip install -e /src/Sinz2018_NIPS

#WORKDIR /src
#RUN git clone https://github.com/atlab/tuna.git && \
#    pip3 install -e tuna/

RUN pip3 install --no-cache-dir --upgrade datajoint && \
    pip3 install nose

WORKDIR /notebooks

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]
