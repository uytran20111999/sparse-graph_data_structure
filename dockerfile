FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    graphviz \
    python3-pip \
    python3-tk \
    && pip3 install -r "requirements.txt" \
    && pip install tk

RUN apt-get install -y libx11-dev
WORKDIR /usr/src/app
COPY demo_graph/ ./demo_graph
COPY graph_viz_render/ ./graph_viz_render
COPY __init__.py .
COPY demo.py .
COPY graph_obj.py .
COPY my_util.py .