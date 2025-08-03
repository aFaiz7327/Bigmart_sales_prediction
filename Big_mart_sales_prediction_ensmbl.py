import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, HuberRegressor, TheilSenRegressor,
    ElasticNet, Lasso
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_data(train, final_test_data):
    """Load datasets with validation"""
    print("Loading BigMart datasets...")
    
    train_data = train.copy()
    test_data = final_test_data.copy()
    
    print(f"   - Training data: {train_data.shape}")
    print(f"   - Test data: {test_data.shape}")
    
    return train_data, test_data


def handle_missing_values(data):
    """Intelligent missing value handling"""
    print("\nHandling missing values intelligently...")
    
    # Item_Weight: Multiple strategies
    print("   - Item_Weight missing values...")
    
    # Strategy 1: By Item_Identifier
    weight_by_item = data.groupby('Item_Identifier')['Item_Weight'].mean()
    data['Item_Weight'] = data.apply(
        lambda x: weight_by_item.get(x['Item_Identifier'], x['Item_Weight']) if pd.isna(x['Item_Weight']) else x['Item_Weight'], 
        axis=1
    )
    
    # Strategy 2: By Item_Type
    weight_by_type = data.groupby('Item_Type')['Item_Weight'].mean()
    data['Item_Weight'] = data.apply(
        lambda x: weight_by_type[x['Item_Type']] if pd.isna(x['Item_Weight']) else x['Item_Weight'], 
        axis=1
    )
    
    # Strategy 3: Overall mean
    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
    
    # Outlet_Size: Context-aware imputation
    print("   - Outlet_Size missing values...")
    
    # Use mode by outlet type and location
    for outlet_type in data['Outlet_Type'].unique():
        for location in data['Outlet_Location_Type'].unique():
            mask = (data['Outlet_Type'] == outlet_type) & (data['Outlet_Location_Type'] == location)
            subset = data[mask]
            
            if len(subset) > 0 and not subset['Outlet_Size'].mode().empty:
                mode_size = subset['Outlet_Size'].mode()[0]
                null_mask = mask & data['Outlet_Size'].isna()
                data.loc[null_mask, 'Outlet_Size'] = mode_size
    
    # Fill remaining nulls with mode by outlet type
    for outlet_type in data['Outlet_Type'].unique():
        type_mask = data['Outlet_Type'] == outlet_type
        if not data[type_mask]['Outlet_Size'].mode().empty:
            mode_size = data[type_mask]['Outlet_Size'].mode()[0]
            null_mask = type_mask & data['Outlet_Size'].isna()
            data.loc[null_mask, 'Outlet_Size'] = mode_size
    
    return data


def clean_and_standardize(data):
    """Clean and standardize data"""
    print("\nCleaning and standardizing...")
    
    # Standardize Item_Fat_Content
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular',
        'LOW FAT': 'Low Fat', 'REGULAR': 'Regular'
    })
    
    return data


def engineer_features(data):
    """Core feature engineering"""
    print("\nEngineering core features...")
    
    # Time-based features
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
    data['Outlet_Age_Category'] = pd.cut(data['Outlet_Years'], 
                                       bins=[0, 8, 15, 30], 
                                       labels=['New', 'Established', 'Mature'])
    
    # Item identifier features
    data['Item_Category'] = data['Item_Identifier'].str[:2]
    # Extract numeric part from Item_Identifier, handle non-numeric cases
    data['Item_Number'] = pd.to_numeric(data['Item_Identifier'].str[2:], errors='coerce').fillna(0).astype(int)
    
    # Price and weight features
    data['Price_per_Weight'] = data['Item_MRP'] / data['Item_Weight']
    
    # Visibility features
    data['Has_Visibility'] = (data['Item_Visibility'] > 0).astype(int)
    
    # Handle zero visibility
    visibility_means = data[data['Item_Visibility'] > 0].groupby('Item_Type')['Item_Visibility'].mean()
    data['Item_Visibility_Adjusted'] = data.apply(
        lambda x: visibility_means[x['Item_Type']] if x['Item_Visibility'] == 0 else x['Item_Visibility'], 
        axis=1
    )
    
    # Binning features
    data['MRP_Category'] = pd.qcut(data['Item_MRP'], q=5, 
                                 labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'])
    data['Weight_Category'] = pd.qcut(data['Item_Weight'], q=4, 
                                    labels=['Light', 'Medium', 'Heavy', 'VeryHeavy'])
    
    return data


def create_advanced_features(data):
    """Create advanced features"""
    print("\nCreating advanced features...")
    
    # Store performance features (for training data)
    train_mask = data['source'] == 'train'
    if train_mask.sum() > 0:
        store_stats = data[train_mask].groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg([
            'mean', 'std', 'count'
        ])
        store_stats.columns = ['Outlet_Avg_Sales', 'Outlet_Sales_Std', 'Outlet_Item_Count']
        
        data = data.merge(store_stats, left_on='Outlet_Identifier', right_index=True, how='left')
        
        # Fill test data
        overall_mean = data[train_mask]['Item_Outlet_Sales'].mean()
        overall_std = data[train_mask]['Item_Outlet_Sales'].std()
        overall_count = data[train_mask].groupby('Outlet_Identifier').size().mean()
        
        data['Outlet_Avg_Sales'].fillna(overall_mean, inplace=True)
        data['Outlet_Sales_Std'].fillna(overall_std, inplace=True)
        data['Outlet_Item_Count'].fillna(overall_count, inplace=True)
    
    # Market positioning
    data['Market_Position'] = 'Standard'
    
    high_mrp_mask = data['Item_MRP'] > data['Item_MRP'].quantile(0.8)
    low_fat_mask = data['Item_Fat_Content'] == 'Low Fat'
    data.loc[high_mrp_mask & low_fat_mask, 'Market_Position'] = 'Premium'
    
    low_mrp_mask = data['Item_MRP'] < data['Item_MRP'].quantile(0.2)
    data.loc[low_mrp_mask, 'Market_Position'] = 'Budget'
    
    # Interaction features
    data['Outlet_Item_Combo'] = data['Outlet_Type'] + '_' + data['Item_Type']
    data['Size_Location_Combo'] = data['Outlet_Size'].fillna('Unknown') + '_' + data['Outlet_Location_Type']
    
    return data


def encode_categoricals(data):
    """Encode categorical variables"""
    print("\nEncoding categorical features...")
    
    categorical_cols = [
        'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 
        'Outlet_Type', 'Item_Category', 'Outlet_Age_Category', 'MRP_Category', 
        'Weight_Category', 'Market_Position', 'Outlet_Item_Combo', 'Size_Location_Combo'
    ]
    
    label_encoders = {}
    
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            # Convert to string first, then handle nulls
            data[col] = data[col].astype(str)
            data[col] = data[col].replace('nan', 'Unknown')
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
            print(f"   - {col}: {len(le.classes_)} categories")
    
    return data, label_encoders


def preprocess_data(train_data, test_data):
    """Complete preprocessing pipeline"""
    print("\n" + "="*70)
    print("ADVANCED PREPROCESSING & FEATURE ENGINEERING")
    print("="*70)
    
    # Combine for consistent preprocessing
    train_copy = train_data.copy()
    test_copy = test_data.copy()
    
    train_copy['source'] = 'train'
    test_copy['source'] = 'test'
    test_copy['Item_Outlet_Sales'] = 0
    
    combined = pd.concat([train_copy, test_copy], ignore_index=True)
    print(f"Combined dataset: {combined.shape}")
    
    # Preprocessing steps
    combined = handle_missing_values(combined)
    combined = clean_and_standardize(combined)
    combined = engineer_features(combined)
    combined = create_advanced_features(combined)
    combined, label_encoders = encode_categoricals(combined)
    
    print("Advanced preprocessing completed!")
    
    return combined, label_encoders


def initialize_models():
    """Initialize model suite"""
    print("\n" + "="*70)
    print("INITIALIZING MODERN MODEL SUITE")
    print("="*70)
    
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.005,
            random_state=42,
            n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=300, learning_rate=0.01, random_state=42),
        'LinearRegression': LinearRegression(),
        'MLPRegressor': MLPRegressor(
            hidden_layer_sizes=(100, 75), max_iter=500, random_state=70,
            learning_rate_init=0.005, early_stopping=True
        ),
        'Ridge': Ridge(alpha=5.0),
        'HuberRegressor': HuberRegressor(epsilon=1.35, max_iter=200),
        'TheilSenRegressor': TheilSenRegressor(random_state=42, max_subpopulation=1000),
        'RandomForest': RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'KNeighbors': KNeighborsRegressor(n_neighbors=8, weights='distance'),
        # Additional models
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_samples_split=2, min_samples_leaf=1, random_state=42
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=200, max_depth=12, min_samples_split=2,
            random_state=42, n_jobs=-1
        ),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'Lasso': Lasso(alpha=0.01, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    }
    
    print(f"Initialized {len(models)} models:")
    for i, name in enumerate(models.keys(), 1):
        print(f"   {i:2d}. {name}")
    
    return models


def prepare_features(data):
    """Prepare features for modeling"""
    print(f"\nPreparing features...")
    
    # Core feature set
    feature_cols = [
        # Basic features
        'Item_Weight', 'Item_Fat_Content', 'Item_Visibility_Adjusted', 'Item_Type', 
        'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 
        'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Years',
        
        # Engineered features
        'Item_Category', 'Item_Number', 'Price_per_Weight', 'Has_Visibility',
        'MRP_Category', 'Weight_Category', 'Market_Position', 'Outlet_Age_Category',
        
        # Advanced features
        'Outlet_Avg_Sales', 'Outlet_Sales_Std', 'Outlet_Item_Count',
        'Outlet_Item_Combo', 'Size_Location_Combo'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in data.columns]
    
    # Split data
    train_data = data[data['source'] == 'train']
    test_data = data[data['source'] == 'test']
    
    X_train = train_data[available_features]
    y_train = train_data['Item_Outlet_Sales']
    X_test = test_data[available_features]
    
    print(f"   - Features: {len(available_features)}")
    print(f"   - Training: {X_train.shape}")
    print(f"   - Test: {X_test.shape}")
    
    return X_train, y_train, X_test, available_features


def tune_model(model_name, model, X_train, y_train):
    """Tune hyperparameters for a specific model"""
    print(f"Tuning {model_name}...")
    
    # Define parameter grids for different models
    param_grids = {
        'XGBoost': {
            'n_estimators': [500, 1000, 2000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [6, 8, 10],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 10, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Ridge': {
            'alpha': [0.1, 1.0, 5.0, 10.0]
        },
        'ElasticNet': {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'KNeighbors': {
            'n_neighbors': [5, 8, 10, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    }
    
    # Use default parameters for models not in the grid
    if model_name not in param_grids:
        print(f"   - No tuning parameters defined for {model_name}, using defaults")
        return model
    
    # Use RandomizedSearchCV for efficiency
    cv = RandomizedSearchCV(
        model, 
        param_distributions=param_grids[model_name],
        n_iter=10,  # Number of parameter settings sampled
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit on a smaller subset for efficiency if dataset is large
    sample_size = min(5000, X_train.shape[0])
    if X_train.shape[0] > sample_size:
        indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_sample, y_sample = X_train.iloc[indices], y_train.iloc[indices]
    else:
        X_sample, y_sample = X_train, y_train
    
    try:
        cv.fit(X_sample, y_sample)
        print(f"   - Best parameters: {cv.best_params_}")
        print(f"   - Best score: {-cv.best_score_:.4f} RMSE")
        return cv.best_estimator_
    except Exception as e:
        print(f"   - Error during tuning: {str(e)}")
        print(f"   - Using default parameters")
        return model


def train_and_evaluate(models, X_train, y_train):
    """Train and evaluate all models with improved robustness"""
    print(f"\nTraining and evaluating models...")
    
    # Validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    results = {}
    tuned_models = {}
    
    # First, identify top performing models with default parameters
    for name, model in models.items():
        print(f"\nInitial evaluation of {name}...")
        
        try:
            # Train with cross-validation for more robust evaluation
            cv_scores = cross_val_score(
                model, X_train_split, y_train_split, 
                cv=3, 
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            mean_cv_rmse = -np.mean(cv_scores)
            std_cv_rmse = np.std(cv_scores)
            
            print(f"   - CV RMSE: {mean_cv_rmse:.2f} ± {std_cv_rmse:.2f}")
            
            # Train on split for validation metrics
            model.fit(X_train_split, y_train_split)
            
            # Predict
            train_pred = model.predict(X_train_split)
            val_pred = model.predict(X_val_split)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_split, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
            val_r2 = r2_score(y_val_split, val_pred)
            val_mae = mean_absolute_error(y_val_split, val_pred)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'cv_rmse': mean_cv_rmse
            }
            
            print(f"   - Train RMSE: {train_rmse:8.2f}")
            print(f"   - Val RMSE:   {val_rmse:8.2f}")
            print(f"   - Val R²:     {val_r2:8.4f}")
            print(f"   - Val MAE:    {val_mae:8.2f}")
            
        except Exception as e:
            print(f"   - Error: {str(e)}")
            continue
    
    # Sort models by validation RMSE
    if results:
        sorted_models = sorted(results.items(), key=lambda x: x[1]['val_rmse'])
        
        # Tune top 3 models if we have at least 3
        top_models_to_tune = min(3, len(sorted_models))
        print(f"\nTuning top {top_models_to_tune} models...")
        
        for i in range(top_models_to_tune):
            name, result = sorted_models[i]
            print(f"\n{i+1}. Tuning {name}...")
            
            try:
                # Tune the model
                tuned_model = tune_model(name, models[name], X_train, y_train)
                
                # Evaluate tuned model
                tuned_model.fit(X_train_split, y_train_split)
                val_pred = tuned_model.predict(X_val_split)
                tuned_val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
                
                print(f"   - Original Val RMSE: {result['val_rmse']:.2f}")
                print(f"   - Tuned Val RMSE:    {tuned_val_rmse:.2f}")
                
                # Update if tuned model is better
                if tuned_val_rmse < result['val_rmse']:
                    print(f"   - Improvement: {result['val_rmse'] - tuned_val_rmse:.2f}")
                    tuned_models[name] = tuned_model
                else:
                    print(f"   - No improvement, keeping original model")
            except Exception as e:
                print(f"   - Error during tuning: {str(e)}")
                continue
        
        # Update results with tuned models
        for name, tuned_model in tuned_models.items():
            # Retrain and evaluate
            tuned_model.fit(X_train_split, y_train_split)
            train_pred = tuned_model.predict(X_train_split)
            val_pred = tuned_model.predict(X_val_split)
            
            train_rmse = np.sqrt(mean_squared_error(y_train_split, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
            val_r2 = r2_score(y_val_split, val_pred)
            val_mae = mean_absolute_error(y_val_split, val_pred)
            
            results[name] = {
                'model': tuned_model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'tuned': True
            }
        
        # Get top two models
        sorted_models = sorted(results.items(), key=lambda x: x[1]['val_rmse'])
        top_model_name, top_model_results = sorted_models[0]
        second_model_name, second_model_results = sorted_models[1]
        
        top_model = top_model_results['model']
        second_model = second_model_results['model']
        
        print(f"\nTop two models selected for weighted ensemble:")
        print(f"1. {top_model_name}: RMSE = {top_model_results['val_rmse']:.2f}")
        print(f"2. {second_model_name}: RMSE = {second_model_results['val_rmse']:.2f}")
        
        # Retrain both models on full training data
        print(f"Retraining top two models on full dataset...")
        top_model.fit(X_train, y_train)
        second_model.fit(X_train, y_train)
        
        # Create ensemble info
        ensemble_info = {
            'top_model': {
                'name': top_model_name,
                'model': top_model,
                'weight': 0.7
            },
            'second_model': {
                'name': second_model_name,
                'model': second_model,
                'weight': 0.3
            }
        }
        
        return results, ensemble_info
    else:
        return results, None


def predict_with_ensemble(ensemble_info, X_test):
    """Generate predictions using weighted ensemble of top two models"""
    if ensemble_info is None:
        return None
    
    top_model = ensemble_info['top_model']['model']
    second_model = ensemble_info['second_model']['model']
    top_weight = ensemble_info['top_model']['weight']
    second_weight = ensemble_info['second_model']['weight']
    
    print(f"\nGenerating ensemble predictions...")
    print(f"   - {ensemble_info['top_model']['name']} (weight: {top_weight:.1f})")
    print(f"   - {ensemble_info['second_model']['name']} (weight: {second_weight:.1f})")
    
    # Generate predictions from both models
    top_preds = top_model.predict(X_test)
    second_preds = second_model.predict(X_test)
    
    # Weighted average
    weighted_preds = (top_weight * top_preds) + (second_weight * second_preds)
    
    # Ensure non-negative predictions
    return np.maximum(weighted_preds, 0)


def create_submission(predictions, test_data):
    """Create submission file"""
    print(f"\nCreating submission...")
    
    submission = pd.DataFrame({
        'Item_Identifier': test_data['Item_Identifier'],
        'Outlet_Identifier': test_data['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    submission.to_csv('weighted_ensemble_submission_final.csv', index=False)
    print(f"   - Saved: weighted_ensemble_submission.csv")
    
    print(f"\nSample predictions:")
    print(submission.head(10).to_string(index=False))
    
    return submission


def print_final_summary(results, ensemble_info):
    """Print final summary"""
    print(f"\n" + "="*80)
    print("ADVANCED ENSEMBLE SYSTEM - PIPELINE COMPLETED!")
    print("="*80)
    
    # Model ranking
    print(f"\nMODEL RANKING:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['val_rmse'])
    
    for i, (name, scores) in enumerate(sorted_models, 1):
        tuned_status = " (tuned)" if scores.get('tuned', False) else ""
        print(f"   {i}. {name:<20}{tuned_status} | RMSE: {scores['val_rmse']:>7,.0f} | R²: {scores['val_r2']:>6.3f}")
    
    # Ensemble information
    print(f"\nWEIGHTED ENSEMBLE:")
    print(f"   - Primary model ({ensemble_info['top_model']['weight']:.1f}): {ensemble_info['top_model']['name']}")
    print(f"   - Secondary model ({ensemble_info['second_model']['weight']:.1f}): {ensemble_info['second_model']['name']}")


def run_bigmart_pipeline(train, final_test_data):
    """Execute complete pipeline"""
    print("="*80)
    print("BIGMART SALES PREDICTION - ADVANCED WEIGHTED ENSEMBLE SYSTEM")
    print("="*80)
    
    # Step 1: Load data
    train_data, test_data = load_data(train, final_test_data)
    
    # Step 2: Preprocessing
    processed_data, label_encoders = preprocess_data(train_data, test_data)
    
    # Step 3: Model training
    models = initialize_models()
    X_train, y_train, X_test, feature_names = prepare_features(processed_data)
    model_results, ensemble_info = train_and_evaluate(models, X_train, y_train)
    
    # Step 4: Ensemble predictions
    print(f"\n" + "="*70)
    print("GENERATING WEIGHTED ENSEMBLE PREDICTIONS")
    print("="*70)
    
    predictions = predict_with_ensemble(ensemble_info, X_test)
    
    print(f"Generated {len(predictions):,} predictions")
    print(f"   - Range: {predictions.min():,.2f} - {predictions.max():,.2f}")
    print(f"   - Mean: {predictions.mean():,.2f}")
    
    # Step 5: Create submission
    submission = create_submission(predictions, test_data)
    
    # Step 6: Final summary
    print_final_summary(model_results, ensemble_info)
    
    return submission


def main(train, final_test_data):
    """Main execution function"""
    try:
        print("Starting BigMart Advanced Weighted Ensemble Prediction System...")
        
        # Run pipeline
        submission = run_bigmart_pipeline(train, final_test_data)
        
        print(f"\nSystem completed successfully!")
        return submission
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train = pd.read_csv('train_v9rqX0R.csv')
    final_test_data = pd.read_csv('test_AbJTz2l.csv')
    main(train, final_test_data)