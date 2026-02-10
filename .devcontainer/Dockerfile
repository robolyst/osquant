FROM python:3.12-slim-bookworm

RUN apt update

# Install hugo for local development.
# This takes up a lot of space
RUN apt install -y hugo

# Markdown linting.
RUN apt install -y npm
RUN npm install -g markdownlint-cli

# Install vale
RUN apt install -y npm hugo wget
RUN wget -q -O /tmp/vale.tar.gz https://github.com/errata-ai/vale/releases/download/v2.28.0/vale_2.28.0_Linux_64-bit.tar.gz \
  && tar -xvzf /tmp/vale.tar.gz -C bin \
  && rm /tmp/vale.tar.gz

# Setup oh my zsh for dev
RUN apt install -y git zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV SHELL=/usr/bin/zsh

# Install quarto
ENV QUARTO_VERSION=1.7.31
RUN pip3 install numpy scipy matplotlib jupyter pandas pyarrow
# pick correct arch at build time
RUN QUARTO_ARCH=$(dpkg --print-architecture 2>/dev/null || echo amd64) \
  && wget -q -O /tmp/quarto.deb "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${QUARTO_ARCH}.deb" \
  && dpkg -i /tmp/quarto.deb \
  && rm /tmp/quarto.deb