# stock_prediction_LSTM
- Final Project for NTOU 2023 Data Analysis Course
- 量化分析 台積電、聯發科、鴻海、中華電信、台達電 5間科技公司股票，並使用LSTM模型預測股票走向
- 參考至[📊Stock Market Analysis 📈 + Prediction using LSTM](https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm#5.-How-much-value-do-we-put-at-risk-by-investing-in-a-particular-stock?)

## 架構與過程
1. 取得5間公司的股價資訊
    - 使用 yahoo finance API 下載 2008/01至今 5間公司的股價資訊
    - 使用 talib 計算個別的 20ma, 50ma, 120ma, 14rsi
2. 切割資料集
    - 訓練資料集為 2008/07/01-2023/06/30 存進 train.csv
    - 測試資料集為 2023/07/01-2023/12/28 存進 test.csv
    - csv檔屬性為Date, Open, High, Low, Close, Adj Close, Volume, 20ma, 50ma, 120ma, 14rsi, Company 
3. 繪畫 (訓練資料集) k線圖+120ma疊圖 (共5張) 存進/result/公司股票代號.jpg
4. 不同科技公司的相關性分析，繪畫熱力圖 存進/result/relation.jpg
5. 使用LSTM模型預測股票
    - 訓練模型 (2008/07/01-2023/06/30)
        - x_train : y_train = 60:1
        - 2個LSTM層和2個Dense層
    - 測試模型 (2023/07/01-2023/12/28)
        - x_test : y_test = 60:1
        - 繪畫 prediction和y_test 存進/result/公司股票代號_pred.jpg
        - 5間公司的rmse = [554.5585249161572, 873.914044926729, 101.34767305749004, 117.08556435189753, 314.33293707140325]
