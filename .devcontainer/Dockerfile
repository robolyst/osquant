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

# WORKS
ENV QUARTO_VERSION=1.4.526

# DOES NOT WORK
# ENV QUARTO_VERSION=1.4.527

# Install quarto
RUN wget -q -O /tmp/quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v$QUARTO_VERSION/quarto-$QUARTO_VERSION-linux-arm64.deb \
  && dpkg -i /tmp/quarto.deb \
  && rm /tmp/quarto.deb