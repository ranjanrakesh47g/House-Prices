# House-Prices

## Data description: 
6.3 MB of training data, containing around 80 features like no. of rooms, house area, garage quality (if present), kitchen quality, electrical connections’ quality etc, and the goal was to predict sale price of the house.


## Data exploration and preparation: 
- Imputation
- Dummies from categorical features
- Feature extraction, like age in place of year when built, etc
- Standardization


## Models used:
Tree-models:	
- XGBoost
- GBM
- ExtraTrees
- RandomForest
- AdaBoost
- Bagging

Linear-models:	
- Ridge
- Lasso
- LassoCV
- LassoLars
- ElasticNet
- BayesianRidge
		

## Hyperparameter tuning:
- 10 fold CV + GridSearch;
- 10 fold CV + GridSearch + RandomSearch, always better than GridSearch;
- 10 fold CV + GridSearch + Genetic algorithm:  stuck in local optima, sometimes better than RandomSearch;                       
- 10 fold CV + GridSearch + Bayesian optimisation:  performed the best in most cases.


## Final ensemble:
- Stacking


## Result:
- R2 score on train set: 0.889743
- R2 score on test set: 0.87435
- Kaggle rank: 1212 (Top 36%), as on 14th Jan 2017
