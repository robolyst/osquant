FROM alpine:3.17.0

RUN apk add npm
RUN npm install -g markdownlint-cli

WORKDIR /app