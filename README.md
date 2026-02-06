# Quant_Trading_Strategy_with_SocialMedia_Sentiment  
<img width="943" height="249" alt="image" src="https://github.com/user-attachments/assets/b9e2e2ef-0772-4e11-9285-ffe888cdefa8" />  
  
This report presents a comprehensive backtesting analysis of various trading strategies applied to the Hang Seng Index (HSI) for the period from February 24, 2022, to March 12, 2025. The analysis evaluates four distinct strategies:   
  
• Pure Sentiment  
• Sentiment combined with Moving Averages (MA)  
• Sentiment combined with Relative Strength Index (RSI)  
• Logistic Regression-based approach  
  
The Sentiment + MA Strategy emerged as the top-performing approach, delivering a total return of 4.15%, outperforming the market"s cumulative return of 3.05%.  
  
# Comments  
Prediction Vote analysis: The HSI Up/Down votes from the social media are not a good enough predictor of market direction. The social media sentiment prediction is only slightly better than a coin flip (53.41%).  
  
Trend Filtering is Crucial: Sentiment signals alone are noisy, combining them with trend indicators like Moving Averages significantly enhances reliability. However, the strategy is still not good enough since the Sharp ratio is just 0.165.  
  
Risk Management: The Sentiment + MA strategy reduced the number of trades by nearly 43% compared to the Pure Sentiment strategy, leading to lower transaction costs and better signal selection.  
  
Future Work: Further optimization of the Logistic Regression model and exploring other technical indicators could potentially enhance the alpha generation, such as LSTM model, GAN model, etc. And more strategies like “Rubber Band Trading Strategy”, “MFI Indicator Strategy” can be implemented.
  
# Visualization  
<img width="4469" height="4466" alt="HSI_Strategy_Analysis" src="https://github.com/user-attachments/assets/6cd1d129-c80c-4924-a6ea-e9e9100b540a" />
