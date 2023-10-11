FROM kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.6.1



# copy
COPY --chown=root:root . /workspace
WORKDIR /workspace
# RUN useradd -ms /bin/bash admin
# RUN chmod 755 /workspace
# RUN chown -R root:root /workspace
EXPOSE 8000
EXPOSE 8888

USER root
# install pip
RUN pip install -r /workspace/requirements.txt
RUN pip install -U sentence-transformers
RUN pip install protobuf==3.20.*
# RUN pip install typing-extensions --upgrade
# RUN pip install lightning==2.0.1
# default os environment
ENV MODEL_URL=https://storage.itss.io.vn/models/fe-ir/modelv1.0.zip
ENV MODEL_PATH=./assets/models/48
ENV FULL_TEXT_PATH=./assets/models/48/fulltext_0.csv

# stop jupyter notebook
# run uvicorn main:app --reload
CMD ["uvicorn", "main:app", "--reload","--host","0.0.0.0"]