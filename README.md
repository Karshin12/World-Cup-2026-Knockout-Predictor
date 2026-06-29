# FIFA World Cup 2026™ Live Knockout Predictor Engine

An industry-grade, data-driven web application that simulates and predicts the single-elimination knockout bracket of the FIFA World Cup 2026™. Powered by a dynamic, chronologically trained **Elo Rating System**, this engine processes live-updating international football datasets to generate precise algorithmic match probabilities in real time.


## Analytics Architecture & Methodology

Unlike basic prediction models that rely on static, unweighted goal averages, this application implements a production level predictive pipeline built on standard sports-tech quantitative principles:

1. **Chronological Elo Ingestion Engine:** The model ingests a comprehensive historical dataset containing over a century of international football history. It tracks and calculates team strength sequentially down to the present day, using a custom logistic probability curve transformation.
2. **Opponent Adjusted Value:** The mathematical framework scales rating shifts based on quality of competition. Defeating a top-tier opponent increases a team's latent variable profile far more than defeating a lower ranked squad, dynamically self-correcting for strength of schedule.
3. **Live Bracket State Synchronization:** Utilizing data logs via automated pipeline connections, the platform automatically intercepts real-world match resolutions as they settle, instantly freezing completed bracket blocks while keeping upcoming pathway branches fully interactive.


## Features

* A clean, optimized central layout engineered specifically for mobile and desktop scannability.
* Custom UI logic built in Streamlit that seamlessly updates labels from upcoming (`Who advances from X vs Y?`) to finalized configurations (`Advanced from X vs Y:`) when live match commits are processed.
* Fully functional dynamic routing tracking consolation bracket paths alongside the main championship line.


## Technology Stack

* **Language:** Python 3.9+
* **Framework:** Streamlit (Dynamic Web UI)
* **Data Engineering:** Pandas, NumPy
* **Analytics Layer:** Logistical Elo Rating Engine
* **CI/CD Deployment:** Streamlit Community Cloud (Auto-tracked GitHub Commits)
