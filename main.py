"""
TSE Wang Pok
Date: 2026-02-06
Personal Website: https://jerrytseee.github.io/

According HKMA, average risk free rate from 2022 to 2025 is around 3.4%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


class HSITradingStrategy:
    def __init__(self, file_path):
        self.raw_data = self.load_data(file_path) # keep original data unchanged
        self.data = self.raw_data.copy()
        self.prepare_data()
        self.scaler = StandardScaler()
        self.logreg_model = None # for logistic regression model!
    
    def load_data(self, file_path):
        print("loading HSI data ...")
        df = pd.read_excel(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sentiment'] = df['Up votes'] - df['Down votes']
        df['Sentiment_Strength'] = abs(df['Sentiment'])
        df['Return'] = df['Close'].pct_change() * 100
        return df
    
    def prepare_data(self):
        self.raw_data['MA_20'] = self.raw_data['Close'].rolling(window=20).mean()
        self.raw_data['MA_50'] = self.raw_data['Close'].rolling(window=50).mean()
        self.raw_data['RSI'] = self.calculate_rsi(self.raw_data['Close'])
        self.raw_data['Volatility'] = self.raw_data['Return'].rolling(window=10).std()
        # momentum indicator
        self.raw_data['Momentum_5'] = self.raw_data['Close'] - self.raw_data['Close'].shift(5)
        self.raw_data['Momentum_10'] = self.raw_data['Close'] - self.raw_data['Close'].shift(10)
        # price volatility ratio within a day
        self.raw_data['High_Low_Ratio'] = (self.raw_data['High'] - self.raw_data['Low']) / self.raw_data['Close'] * 100
        self.raw_data['Prediction'] = np.where(self.raw_data['Up votes'] > 0.5, 1, -1)
        
        self.data = self.raw_data.copy()
    
    def calculate_rsi(self, prices, period=14):
        # according to the RSI formula
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data_for_logistic_regression(self):
        """data for logistic regression model training"""
        logreg_data = self.raw_data.copy()

        feature_cols = ['Sentiment', 'Sentiment_Strength', 'Return', 'RSI', 'Volatility', 
                       'MA_20', 'MA_50', 'Momentum_5', 'Momentum_10', 'High_Low_Ratio']
        
        # target for logistic regression: 1 if next day return > 0, else 0
        logreg_data['Target'] = (logreg_data['Return'].shift(-1) > 0).astype(int)
        
        # keep only the rows with all features available
        logreg_data = logreg_data.dropna(subset=feature_cols + ['Target'])
        
        return logreg_data, feature_cols
    
    def train_logistic_regression(self, train_ratio=0.7):
        logreg_data, feature_cols = self.prepare_data_for_logistic_regression()
        
        X = logreg_data[feature_cols].copy()
        y = logreg_data['Target'].copy()
        
        # split data
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # training model!!!!
        self.logreg_model = LogisticRegression(
            C=1.0, # regularization strength
            penalty='l2', # L2 regularization
            solver='liblinear', # good for small datasets
            max_iter=1000,
            random_state=42
        )
        self.logreg_model.fit(X_train_scaled, y_train)
        
        X_all = self.scaler.transform(self.raw_data[feature_cols].fillna(0))
        probabilities = self.logreg_model.predict_proba(X_all)[:, 1]
        self.raw_data['LogReg_Probability'] = probabilities
        
        # make predictions
        y_train_pred = self.logreg_model.predict(X_train_scaled)
        y_test_pred = self.logreg_model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Logistic Regression Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"Logistic Regression Testing Accuracy: {test_accuracy*100:.2f}%")
        
        return train_accuracy, test_accuracy
    
    def analyze_sentiment_accuracy(self):
        # reset data to original raw data copy
        self.data = self.raw_data.copy()
        self.data['Actual_Direction'] = np.where(self.data['Return'] > 0, 1, -1)
        accuracy = (self.data['Prediction'] == self.data['Actual_Direction']).mean() * 100
        return accuracy
    
    def trading_strategy(self, strategy_type, use_data=None):
        """
        four types of strategy:
        1. sentiment_ma: combine sentiment and moving averages
        2. sentiment_rsi: combine sentiment and RSI
        3. sentiment_only: use sentiment only
        4. logistic_regression: use logistic regression predictions
        """

        if use_data is not None:
            self.data = use_data.copy()
        else:
            self.data = self.raw_data.copy()
        
        if strategy_type == 'sentiment_only':
            self.data['Signal'] = self.data['Prediction']

        elif strategy_type == 'sentiment_ma':
            ma_signal = np.where(self.data['Close'] > self.data['MA_20'], 1, -1)
            self.data['Signal'] = np.where(
                (self.data['Prediction'] == 1) & (ma_signal == 1), 1,
                np.where((self.data['Prediction'] == -1) & (ma_signal == -1), -1, 0)
            )

        elif strategy_type == 'sentiment_rsi':
            rsi_signal = np.where(self.data['RSI'] < 30, 1, np.where(self.data['RSI'] > 70, -1, 0))
            self.data['Signal'] = np.where(
                (self.data['Prediction'] == 1) & (rsi_signal != -1), 1,
                np.where((self.data['Prediction'] == -1) & (rsi_signal != 1), -1, 0)
            )
        
        elif strategy_type == 'logistic_regression':
            if self.logreg_model is None:
                print('Training logistic regression model ...')
                self.train_logistic_regression()
            
            # Ensure we're using raw_data which has the LogReg_Probability column
            if 'LogReg_Probability' not in self.data.columns:
                # If not in self.data, get from raw_data
                if 'LogReg_Probability' in self.raw_data.columns:
                    self.data['LogReg_Probability'] = self.raw_data['LogReg_Probability']
                else:
                    # If still not there, calculate probabilities
                    logreg_data, feature_cols = self.prepare_data_for_logistic_regression()
                    X_all = self.scaler.transform(logreg_data[feature_cols].fillna(0))
                    probabilities = self.logreg_model.predict_proba(X_all)[:, 1]
                    self.data['LogReg_Probability'] = probabilities
            
            prob_threshold = 0.55
            self.data['Signal'] = np.where(
                self.data['LogReg_Probability'] > prob_threshold, 1,
                np.where(self.data['LogReg_Probability'] < (1 - prob_threshold), -1, 0)
            )
            # if self.logreg_model is None:
            #     print('Training logistic regression model ...')
            #     self.train_logistic_regression()
            
            # prob_threshold = 0.55 # set a threshold for making buy/sell decisions
            # self.data['Signal'] = np.where(
            #     self.data['LogReg_Probability'] > prob_threshold, 1,
            #     np.where(self.data['LogReg_Probability'] < (1 - prob_threshold), -1, 0)
            # )
        
        # calculate positions (buy = 1, sell = -1, hold = 0)
        self.data['Position'] = self.data['Signal'].shift(1)
        self.data['Strategy_Return'] = self.data['Position'] * self.data['Return']
        
        # drop NaN rows (only those caused by strategy calculation)
        self.data = self.data.dropna(subset=['Strategy_Return'])
        
        return self.data
    
    def backtest_strategy(self, strategy_type):
        """backtest specified strategy"""
        # use a separate copy of raw data for each strategy
        strategy_data = self.raw_data.copy()
        strategy_data = self.trading_strategy(strategy_type, use_data=strategy_data)
        
        # calculate backtest metrics
        strategy_data['Cumulative_Market_Return'] = (1 + strategy_data['Return']/100).cumprod()
        strategy_data['Cumulative_Strategy_Return'] = (1 + strategy_data['Strategy_Return']/100).cumprod()
        
        # calculate portfolio value
        initial_capital = 100000
        strategy_data['Portfolio_Value'] = initial_capital * strategy_data['Cumulative_Strategy_Return']
        
        # calculate trading metrics
        trades = strategy_data['Position'].diff().fillna(0)
        num_trades = abs(trades[trades != 0]).sum() / 2
        
        # calculate sharpe ratio
        excess_return = strategy_data['Strategy_Return'] - 0.034/252
        sharpe_ratio = np.sqrt(252) * (excess_return.mean() / excess_return.std())
        
        # calculate win rate
        winning_trades = strategy_data[strategy_data['Strategy_Return'] > 0]['Strategy_Return']
        total_trades = strategy_data[strategy_data['Strategy_Return'] != 0]['Strategy_Return']
        win_rate = len(winning_trades) / len(total_trades) * 100 if len(total_trades) > 0 else 0
        
        # final results
        results = {
            'Initial_Capital': initial_capital,
            'Final_Portfolio_Value': strategy_data['Portfolio_Value'].iloc[-1],
            'Total_Return': (strategy_data['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100,
            'Cumulative_Market_Return': (strategy_data['Cumulative_Market_Return'].iloc[-1] - 1) * 100,
            'Sharpe_Ratio': sharpe_ratio,
            'Win_Rate': win_rate,
            'Number_of_Trades': num_trades,
            'Avg_Daily_Return': strategy_data['Strategy_Return'].mean(),
            'Std_Daily_Return': strategy_data['Strategy_Return'].std()
        }
        
        return results
    
    def generate_plots(self):
        # Create a figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Ensure we have the necessary data
        plot_data = self.raw_data.copy()
        
        # Plot 1: HSI Price with Moving Averages
        axes[0, 0].plot(plot_data['Date'], plot_data['Close'], label='HSI Close', linewidth=2)
        axes[0, 0].plot(plot_data['Date'], plot_data['MA_20'], label='20-day MA', alpha=0.7)
        axes[0, 0].plot(plot_data['Date'], plot_data['MA_50'], label='50-day MA', alpha=0.7)
        axes[0, 0].set_title('HSI Index Price with Moving Averages')
        axes[0, 0].set_ylabel('Close Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Sentiment Distribution
        axes[0, 1].hist(plot_data['Sentiment'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Sentiment Scores')
        axes[0, 1].set_xlabel('Sentiment (Up Votes - Down Votes)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Neutral')
        axes[0, 1].legend()
        
        # Plot 3: RSI with Overbought/Oversold levels
        axes[1, 0].plot(plot_data['Date'], plot_data['RSI'], label='RSI', linewidth=2)
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        axes[1, 0].axhline(y=50, color='y', linestyle='--', alpha=0.3, label='Neutral (50)')
        axes[1, 0].set_title('Relative Strength Index (RSI)')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Daily Returns Distribution
        axes[1, 1].hist(plot_data['Return'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Daily Returns')
        axes[1, 1].set_xlabel('Daily Return (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=plot_data['Return'].mean(), color='r', linestyle='--', 
                        alpha=0.5, label=f'Mean: {plot_data["Return"].mean():.3f}%')
        axes[1, 1].legend()
        
        # Plot 5: Volume/Volatility Analysis (if volume data exists)
        if 'High_Low_Ratio' in plot_data.columns:
            axes[2, 0].plot(plot_data['Date'], plot_data['High_Low_Ratio'], 
                        label='Daily Range %', linewidth=1, alpha=0.7)
            axes[2, 0].set_title('Daily Price Range (High-Low) as % of Close')
            axes[2, 0].set_ylabel('Range (%)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            # Alternative: Plot Momentum
            axes[2, 0].plot(plot_data['Date'], plot_data['Momentum_5'], 
                        label='5-Day Momentum', linewidth=1, alpha=0.7)
            axes[2, 0].plot(plot_data['Date'], plot_data['Momentum_10'], 
                        label='10-Day Momentum', linewidth=1, alpha=0.7)
            axes[2, 0].set_title('Price Momentum Indicators')
            axes[2, 0].set_ylabel('Momentum')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # Plot 6: Logistic Regression Probability (if available)
        if 'LogReg_Probability' in plot_data.columns:
            axes[2, 1].plot(plot_data['Date'], plot_data['LogReg_Probability'], 
                        label='Positive Return Probability', linewidth=2, color='purple')
            axes[2, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Boundary (0.5)')
            axes[2, 1].axhline(y=0.7, color='g', linestyle='--', alpha=0.3, label='High Confidence (0.7)')
            axes[2, 1].axhline(y=0.3, color='b', linestyle='--', alpha=0.3, label='Low Confidence (0.3)')
            axes[2, 1].set_title('Logistic Regression Prediction Probability')
            axes[2, 1].set_ylabel('Probability')
            axes[2, 1].set_ylim(0, 1)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        elif self.logreg_model is not None:
            # Show feature importance instead
            # Get feature columns
            _, feature_cols = self.prepare_data_for_logistic_regression()
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': self.logreg_model.coef_[0]
            })
            feature_importance = feature_importance.sort_values(by='Coefficient', key=abs, ascending=False)

            axes[2, 1].barh(feature_importance['Feature'], feature_importance['Coefficient'])
            axes[2, 1].set_title('Logistic Regression Feature Importance')
            axes[2, 1].set_xlabel('Coefficient Value')
            axes[2, 1].grid(True, alpha=0.3, axis='x')
        else:
            # Alternative plot: Volatility
            if 'Volatility' in plot_data.columns:
                axes[2, 1].plot(plot_data['Date'], plot_data['Volatility'], 
                            label='10-day Volatility', linewidth=2, color='orange')
                axes[2, 1].set_title('Market Volatility (10-day rolling std)')
                axes[2, 1].set_ylabel('Volatility')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('HSI_Strategy_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        
        # 基础统计（使用原始数据）
        self.data = self.raw_data.copy()
        print("\n1. DATA SUMMARY:")
        print(f"   • Time Period: {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
        print(f"   • Total Trading Days: {len(self.data)}")
        print(f"   • Average HSI Close: {self.data['Close'].mean():.2f}")
        print(f"   • Average Daily Return: {self.data['Return'].mean():.4f}%")
        print(f"   • Daily Return Volatility: {self.data['Return'].std():.4f}%")
        accuracy = self.analyze_sentiment_accuracy()
        print(f"   • Sentiment Prediction Accuracy: {accuracy:.2f}%")
        
        # 策略回测
        print("\n" + "="*60)
        print("STRATEGY BACKTESTING RESULTS")
        print("="*60)
        
        strategies = ['sentiment_only', 'sentiment_ma', 'sentiment_rsi', 'logistic_regression']
        strategy_names = ['Pure Sentiment', 'Sentiment + MA', 'Sentiment + RSI', 'Logistic Regression']
        
        all_results = []
        
        for strategy, name in zip(strategies, strategy_names):
            print(f"\n{name} Strategy:")
            print("-" * 40)
            
            # backtest strategy (use separate data copy for each)
            results = self.backtest_strategy(strategy)
            all_results.append(results)

            print(f"  Total Return: {results['Total_Return']:.2f}%")
            print(f"  Cumulative Market Return: {results['Cumulative_Market_Return']:.2f}%")
            print(f"  Alpha (vs Market): {results['Total_Return'] - results['Cumulative_Market_Return']:.2f}%")
            print(f"  Sharpe Ratio: {results['Sharpe_Ratio']:.3f}")
            print(f"  Win Rate: {results['Win_Rate']:.1f}%")
            print(f"  Number of Trades: {results['Number_of_Trades']:.0f}")
            print(f"  Final Portfolio Value: ${results['Final_Portfolio_Value']:,.2f}")
        

        # Generate visualizations
        self.generate_plots()

        print("\n" + "="*60)
        print("RECOMMENDED STRATEGY")
        print("="*60)
        
        # Find best strategy based on Sharpe Ratio !
        best_idx = np.argmax([r['Sharpe_Ratio'] for r in all_results])
        best_strategy = strategy_names[best_idx]
        print(f"\nRecommended: {best_strategy} Strategy")


def main():
    file_path = "HSI.xlsx"
    strategy_analyzer = HSITradingStrategy(file_path)
    print(strategy_analyzer.raw_data.head())
    strategy_analyzer.generate_report()

if __name__ == "__main__":
    main()

