# Models Package - Modular Architecture

## Overview

The models package has been restructured to provide a modular, extensible architecture for machine learning models. This new design separates concerns, improves maintainability, and makes it easier to add new models.

## Architecture

```
models/
├── __init__.py              # Package initialization and exports
├── base/                    # Base classes and interfaces
│   ├── __init__.py
│   ├── base_model.py        # Abstract base class for all models
│   ├── interfaces.py        # Protocol definitions and interfaces
│   └── metrics.py           # Common evaluation metrics
├── clustering/              # Clustering models
│   ├── __init__.py
│   └── kmeans_model.py      # K-Means implementation
├── classification/          # Classification models
│   ├── __init__.py
│   ├── knn_model.py         # K-Nearest Neighbors
│   ├── decision_tree_model.py # Decision Tree
│   └── naive_bayes_model.py # Naive Bayes
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── model_registry.py    # Model registration system
│   └── model_factory.py     # Model creation factory
├── register_models.py       # Model registration script
└── new_model_trainer.py     # New trainer using modular architecture
```

## Key Components

### 1. Base Classes (`base/`)

- **`BaseModel`**: Abstract base class that all models inherit from
- **`ModelInterface`**: Protocol defining the interface for all models
- **`ModelMetrics`**: Common evaluation metrics and comparison functions

### 2. Model Categories

- **Clustering**: Unsupervised learning models (K-Means, etc.)
- **Classification**: Supervised learning models (KNN, Decision Tree, Naive Bayes)
- **Deep Learning**: Neural network models (planned for future)

### 3. Utility Modules (`utils/`)

- **`ModelRegistry`**: Manages available models and their metadata
- **`ModelFactory`**: Creates model instances dynamically

## Usage

### Basic Model Creation

```python
from models.utils.model_factory import model_factory

# Create a K-Means model
kmeans = model_factory.create_model('kmeans', n_clusters=5)

# Create a KNN model
knn = model_factory.create_model('knn', n_neighbors=3)
```

### Training and Testing

```python
from models.new_model_trainer import NewModelTrainer

trainer = NewModelTrainer()

# Train and test a specific model
y_pred, accuracy, report = trainer.train_and_test_model(
    'kmeans', X_train, y_train, X_test, y_test
)

# Train and test all models
results = trainer.train_and_test_all_models(
    X_train, y_train, X_test, y_test
)
```

### Model Information

```python
# Get available models
available_models = model_factory.get_available_models()

# Get model metadata
model_info = model_factory.get_model_info('kmeans')

# Get model categories
categories = model_factory.get_model_categories()
```

## Adding New Models

### 1. Create Model Class

```python
from models.base.base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
        
    def fit(self, X, y):
        # Implement fitting logic
        pass
        
    def predict(self, X):
        # Implement prediction logic
        pass
```

### 2. Register the Model

```python
from models.utils.model_registry import model_registry

model_registry.register_model(
    'new_model',
    NewModel,
    {
        'category': 'classification',
        'task_type': 'supervised',
        'description': 'Description of the new model'
    }
)
```

## Benefits

1. **Separation of Concerns**: Each model is in its own file
2. **Easy Extension**: Add new models without touching existing code
3. **Better Testing**: Test each model independently
4. **Plugin Architecture**: Models can be loaded dynamically
5. **Consistent Interface**: All models follow the same pattern
6. **Metadata Management**: Rich information about each model

## Migration from Old Architecture

The old `models.py` file is still available for backward compatibility. To migrate:

1. **Immediate**: Use `NewModelTrainer` instead of `ModelTrainer`
2. **Gradual**: Replace direct model usage with factory pattern
3. **Complete**: Update all imports to use new modular structure

## Testing

Run the test script to verify the new architecture:

```bash
python test_new_architecture.py
```

This will test:
- Model registration
- Model factory functionality
- Model creation
- Basic model operations

## Future Enhancements

- **Deep Learning Models**: BERT, LSTM, Transformer models
- **Hyperparameter Tuning**: Automated optimization framework
- **Model Persistence**: Save/load trained models
- **Performance Monitoring**: Built-in benchmarking tools
- **AutoML**: Automatic model selection and tuning
