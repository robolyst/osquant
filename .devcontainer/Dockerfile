FROM python:3.12-slim-bookworm

RUN apt update
RUN apt install -y npm hugo wget
RUN npm install -g markdownlint-cli

RUN pip3 install numpy scipy matplotlib jupyter

RUN wget -q -O /tmp/quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-arm64.deb \
  && dpkg -i /tmp/quarto.deb \
  && rm /tmp/quarto.deb

# Setup oh my zsh for dev
RUN apt install -y git zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV SHELL /usr/bin/zsh