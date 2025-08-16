import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from typing import Tuple, Optional

class SampleDataManager:
    """Manages sample datasets for demonstration and testing."""
    
    def load_sample_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Load a sample dataset by name."""
        try:
            if dataset_name == "iris":
                return self._load_iris()
            elif dataset_name == "titanic":
                return self._load_titanic()
            elif dataset_name == "house_prices":
                return self._load_house_prices()
            else:
                return None, "Unknown dataset"
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"
    
    def _load_iris(self) -> Tuple[pd.DataFrame, str]:
        """Load the Iris dataset."""
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df, "Iris Flower Dataset"
    
    def _load_titanic(self) -> Tuple[pd.DataFrame, str]:
        """Create a synthetic Titanic-like dataset."""
        np.random.seed(42)
        n_samples = 891
        
        # Generate synthetic Titanic data
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.616, 0.384]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 14, n_samples).clip(0.42, 80),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.002]),
            'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.015, 0.004, 0.004, 0.001]),
            'Fare': np.random.exponential(15, n_samples).clip(0, 512),
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.086, 0.724])
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values to make it realistic
        missing_age_indices = np.random.choice(df.index, size=int(0.2 * n_samples), replace=False)
        df.loc[missing_age_indices, 'Age'] = np.nan
        
        missing_embarked_indices = np.random.choice(df.index, size=2, replace=False)
        df.loc[missing_embarked_indices, 'Embarked'] = np.nan
        
        return df, "Titanic Passenger Dataset (Synthetic)"
    
    def _load_house_prices(self) -> Tuple[pd.DataFrame, str]:
        """Create a synthetic house prices dataset."""
        np.random.seed(42)
        n_samples = 1460
        
        # Generate synthetic house data
        data = {
            'Id': range(1, n_samples + 1),
            'LotArea': np.random.normal(10000, 3000, n_samples).clip(1000, 50000),
            'YearBuilt': np.random.randint(1872, 2011, n_samples),
            'YearRemodAdd': np.random.randint(1950, 2011, n_samples),
            'GrLivArea': np.random.normal(1500, 500, n_samples).clip(334, 5642),
            'BedroomAbvGr': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.02, 0.15, 0.44, 0.31, 0.07, 0.01]),
            'FullBath': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.02, 0.35, 0.52, 0.10, 0.01]),
            'HalfBath': np.random.choice([0, 1, 2], n_samples, p=[0.59, 0.37, 0.04]),
            'TotalRooms': np.random.randint(3, 15, n_samples),
            'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.06, 0.18, 0.56, 0.18, 0.02]),
            'Neighborhood': np.random.choice([
                'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown',
                'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards',
                'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
                'Blueste'
            ], n_samples),
            'HouseStyle': np.random.choice([
                '1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'
            ], n_samples, p=[0.5, 0.3, 0.08, 0.07, 0.025, 0.02, 0.003, 0.002]),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate price based on features (with some noise)
        price_base = (
            df['GrLivArea'] * 100 +
            df['LotArea'] * 2 +
            (2024 - df['YearBuilt']) * -100 +
            df['BedroomAbvGr'] * 5000 +
            df['FullBath'] * 8000 +
            df['GarageCars'] * 7000 +
            df['TotalRooms'] * 2000
        )
        
        # Add neighborhood premium/discount
        neighborhood_multiplier = np.random.normal(1.0, 0.2, n_samples).clip(0.7, 1.5)
        df['SalePrice'] = (price_base * neighborhood_multiplier + 
                          np.random.normal(0, 15000, n_samples)).clip(34900, 755000)
        
        # Add some missing values
        missing_garage_indices = np.random.choice(df.index, size=int(0.06 * n_samples), replace=False)
        df.loc[missing_garage_indices, 'GarageCars'] = np.nan
        
        return df, "House Prices Dataset (Synthetic)"
    
    def get_available_datasets(self) -> list:
        """Get list of available sample datasets."""
        return [
            {
                'name': 'iris',
                'display_name': 'Iris Flower Dataset',
                'description': 'Classic dataset with flower measurements and species classification',
                'features': 'Sepal/Petal dimensions, Species',
                'size': '150 rows × 5 columns'
            },
            {
                'name': 'titanic',
                'display_name': 'Titanic Passenger Dataset',
                'description': 'Passenger information from the Titanic with survival data',
                'features': 'Age, Class, Sex, Fare, Survival status',
                'size': '891 rows × 9 columns'
            },
            {
                'name': 'house_prices',
                'display_name': 'House Prices Dataset',
                'description': 'Real estate data with house features and sale prices',
                'features': 'Living area, Year built, Bedrooms, Neighborhood',
                'size': '1460 rows × 12 columns'
            }
        ]
    
    def create_custom_dataset(self, dataset_type: str, n_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
        """Create a custom synthetic dataset."""
        np.random.seed(42)
        
        if dataset_type == "sales":
            data = {
                'Date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
                'Product': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
                'Sales': np.random.exponential(100, n_samples),
                'Quantity': np.random.poisson(5, n_samples),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
                'Customer_Type': np.random.choice(['New', 'Returning'], n_samples, p=[0.3, 0.7])
            }
            return pd.DataFrame(data), "Sales Dataset (Synthetic)"
        
        elif dataset_type == "financial":
            dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
            price_start = 100
            returns = np.random.normal(0.001, 0.02, n_samples)
            prices = [price_start]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = {
                'Date': dates,
                'Price': prices,
                'Volume': np.random.exponential(1000000, n_samples),
                'High': np.array(prices) * (1 + np.random.uniform(0, 0.05, n_samples)),
                'Low': np.array(prices) * (1 - np.random.uniform(0, 0.05, n_samples)),
                'Returns': [0] + list(returns[1:])
            }
            return pd.DataFrame(data), "Financial Time Series (Synthetic)"
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
