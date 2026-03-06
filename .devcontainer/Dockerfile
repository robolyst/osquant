FROM debian:forky-slim

RUN apt update \
    && apt install -y hugo=0.157.0-3 npm wget git zsh --no-install-recommends \
    && apt-get clean

# Markdown linting.
RUN npm install -g markdownlint-cli

# Setup oh my zsh for dev
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV SHELL=/usr/bin/zsh
COPY .zshrc /root/.zshrc
