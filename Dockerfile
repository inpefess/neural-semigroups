FROM jupyter/base-notebook:python-3.9.7
USER root
RUN mkdir -p /kaggle/input/smallsemi
RUN chown -R jovyan /kaggle/input/smallsemi
COPY examples /home/jovyan/examples
RUN chown -R jovyan /home/jovyan/examples
USER jovyan
ENV PATH /home/jovyan/.local/bin:${PATH}
RUN pip install neural-semigroups ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
