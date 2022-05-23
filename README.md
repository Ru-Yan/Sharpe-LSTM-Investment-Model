# Sharpe-LSTM Investment Model
The research provides effective management strategies for different asset portfolios in the financial sector by building models. The VMD-LSTM-PSO model is developed for daily financial market price forecasting, where the time series are decomposed by VMD and the sub-series are used as LSTM input units to carry out forecasting, and then the network parameters are adjusted by PSO to improve the forecasting accuracy, and the Huber-loss of the model is 1.0481e-04. For the daily portfolio strategy, EEG is used to construct a system of investment risk indicators, which is optimized by incorporating the risk indicators into the Sharpe index, and the objective function is analyzed by using GA to derive the optimal daily asset share that maximizes the investor's return with minimal risk. The results of the empirical analysis show that the model provides strategies with good robustness.

## Requirements

Install requirements.txt file to make sure correct versions of libraries are being used.

* Python 3.5.x
* TensorFlow 1.10.0
* Numpy 1.15.0
* Keras 2.2.2
* Matplotlib 2.2.2

