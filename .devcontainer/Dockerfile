FROM debian:bookworm-slim

RUN apt update \
    && apt install -y hugo=0.111.3-1 npm wget git zsh --no-install-recommends \
    && apt-get clean

# Markdown linting.
RUN npm install -g markdownlint-cli

# Install vale
RUN wget -q -O /tmp/vale.tar.gz https://github.com/errata-ai/vale/releases/download/v2.28.0/vale_2.28.0_Linux_64-bit.tar.gz \
  && tar -xvzf /tmp/vale.tar.gz -C bin \
  && rm /tmp/vale.tar.gz

# Setup oh my zsh for dev
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV SHELL=/usr/bin/zsh
COPY .zshrc /root/.zshrc
