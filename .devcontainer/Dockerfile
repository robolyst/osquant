FROM python:3.12-slim-bookworm

RUN apt update
RUN apt install -y npm hugo wget
RUN npm install -g markdownlint-cli

RUN pip3 install numpy scipy matplotlib jupyter

# Install vale
RUN wget -q -O /tmp/vale.tar.gz https://github.com/errata-ai/vale/releases/download/v2.28.0/vale_2.28.0_Linux_64-bit.tar.gz \
  && tar -xvzf /tmp/vale.tar.gz -C bin \
  && rm /tmp/vale.tar.gz

# Setup oh my zsh for dev
RUN apt install -y git zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV SHELL /usr/bin/zsh

# Install quarto
RUN wget -q -O /tmp/quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.526/quarto-1.4.526-linux-arm64.deb \
# RUN wget -q -O /tmp/quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.527/quarto-1.4.527-linux-arm64.deb \ NO
# RUN wget -q -O /tmp/quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.536/quarto-1.4.536-linux-arm64.deb \ NO
  && dpkg -i /tmp/quarto.deb \
  && rm /tmp/quarto.deb