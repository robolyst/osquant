---
title: "How to monitor a trading system"
summary: "
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer odio neque, volutpat vel nunc
    ut. Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui,
    a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris
    pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.
"

date: "2022-12-04"
type: paper
katex: false
authors:
    - Adrian Letchford
draft: true
tags:
    - engineering
---

* Common approach is to use text logs
* Better approach is to use json logs and a metric system
* First, switch your app over to JSON logs
* Second, setup Elastic Search and a log shipper
* Third, setup Grafana