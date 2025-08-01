## Project: Reddit Sentiment ⇒ Stock Alpha

This repo tests whether crowd sentiment from r/WallStreetBets
adds incremental forecasting power over daily stock returns.

* **Data**  
  * WallStreetBets posts 2015-2024 (Kaggle).  
  * NASDAQ daily prices (Kaggle).

* **Key result**  
  * Mean weekly IC = **+0.026**  
  * Newey–West t = **3.4** (lag 3) → clears the +0.02 hurdle.

* **Next steps**  
  * Sector-neutralise signals.  
  * Deploy live with Deephaven + Papermill.