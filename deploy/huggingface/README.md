---
title: India PM2.5 Predictor
emoji: 🌫️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: unknown
---

# India PM2.5 Predictor

One-day-ahead PM2.5 forecast for Indian cities, with the CPCB PM2.5 sub-index category.

This Space uses the Docker SDK. The container listens on port **7860**.

- Interactive API docs: `/docs`
- Health check: `GET /health`
- Forecast: `POST /predict`

The model was trained on the public CPCB city-day file (2015-01-01 to 2020-07-01). It is a historical forecast model, not a live monitor.

This repository does not declare a software license, so the card license is `unknown`. The training file is the public CPCB extract documented as CC0 in the GitHub repository.

Auto-deployed from `main` on [Vineet-shukl/aqi-predictor](https://github.com/Vineet-shukl/aqi-predictor).

Author: Vineet Shukla
