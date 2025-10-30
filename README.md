# Obesity ML Project - Refactored Code

## 📋 Overview

This project implements a complete Machine Learning pipeline for obesity estimation, with a focus on:
- ✅ **Professional code structure** using Cookiecutter template
- ✅ **Refactored codebase** with OOP principles and best practices
- ✅ **Scikit-Learn pipelines** for data preprocessing
- ✅ **MLflow integration** for experiment tracking
- ✅ **DVC support** for data versioning
- ✅ **Docker containerization** for reproducibility
- ✅ **Comprehensive testing** to ensure identical results

## 👥 Team Members

| Student ID | Name | Role |
|------------|------|------|
| A01796095 | Alicia Yovanna Canta Pandal | DevOps Engineer |
| A01796264 | Andrés Roberto Osuna González | SW Engineer |
| A01067109 | Iván Ricardo Cruz Ibarra | Data Scientist |
| A01796828 | Mayra Hernández Alba | Data Engineer |
| A01212428 | Sebastián Ezequiel Coronado Rivera | ML Engineer |

## 🏗️ Project Structure

```
obesity-ml-project/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup configuration
├── .gitignore                        # Git ignore rules
│
├── data/
│   ├── raw/                          # Original immutable data
│   ├── interim/                      # Intermediate transformed data
│   │   ├── obesity_estimation_original.csv
│   │   ├── obesity_estimation_modified.csv
│   │   ├── dataset_limpio.csv       # Original cleaned (from notebook)
│   │   └── dataset_limpio_refactored.csv  # Refactored cleaned (identical!)
│   └── processed/                    # Final data for modeling
│
├── notebooks/
│   └── 01_eda_original.ipynb        # Original EDA notebook (reference)
│
├── src/                              # Source code modules
│   ├── data/
│   │   ├── data_loader.py           # Data loading with error handling
│   │   └── data_cleaner.py          # Data cleaning with OOP & Pipelines
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── utils/
│       ├── config.py                # Centralized configuration
│       └── logger.py                # Logging utilities
│
├── pipelines/
│   └── eda_pipeline.py              # Complete EDA pipeline with MLflow
│
├── tests/
│   └── test_comparison.py           # Unit tests for validation
│
├── scripts/
│   ├── run_eda.py                   # Main script to run EDA
│   └── compare_datasets.py          # Script to compare datasets
│
├── models/                           # Trained models
├── reports/                          # Generated analysis reports
│   ├── figures/                     # Plots and visualizations
│   └── metrics/                     # Evaluation metrics
│
└── mlruns/                          # MLflow tracking directory
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd obesity-ml-project

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Run the EDA Pipeline

```bash
# Run with MLflow tracking
python scripts/run_eda.py

# Run without MLflow tracking
python scripts/run_eda.py --no-mlflow

# Custom input/output paths
python scripts/run_eda.py --input data/interim/your_data.csv --output data/interim/your_output.csv
```

### 3. Compare Results

```bash
# Compare original cleaned dataset vs refactored cleaned dataset
python scripts/compare_datasets.py
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_comparison.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## ✅ Validation Results

### Dataset Comparison: IDENTICAL ✅

```
==========================================
COMPARISON RESULTS
==========================================
✓ Shape match: (2153, 17)
✓ Columns match: 17 columns
✓ Data types match: All dtypes identical
✓ Values match: All values identical
✓ No missing values: 0 in both datasets

🎉 DATASETS ARE IDENTICAL! 🎉
==========================================
```

### Unit Tests: ALL PASSED ✅

```
12 tests passed:
✓ test_files_exist
✓ test_shape_match
✓ test_columns_match
✓ test_dtypes_match
✓ test_no_missing_values
✓ test_numeric_values_match
✓ test_categorical_values_match
✓ test_identical_datasets
✓ test_mixed_type_col_removed
✓ test_correct_columns_present
✓ test_numeric_ranges
✓ test_categorical_normalization
```

## 🔧 Refactoring Improvements

### 1. Code Organization (OOP)
- ✅ Created modular classes: `DataLoader`, `DataCleaner`, `EDAPipeline`
- ✅ Implemented Scikit-Learn transformers for each cleaning step
- ✅ Clear separation of concerns and single responsibility principle

### 2. Scikit-Learn Pipelines
- ✅ `ColumnDropper`: Removes unnecessary columns
- ✅ `TextCleaner`: Cleans text and special characters
- ✅ `NAHandler`: Handles N/A values and variations
- ✅ `NumericConverter`: Converts numeric columns
- ✅ `OutlierHandler`: Validates and corrects outliers
- ✅ `CategoricalNormalizer`: Normalizes categorical values
- ✅ `MissingValueImputer`: Imputes missing values intelligently

### 3. Configuration Management
- ✅ Centralized configuration in `src/utils/config.py`
- ✅ All parameters, ranges, and mappings in one place
- ✅ Easy to modify and maintain

### 4. Logging & Tracking
- ✅ Comprehensive logging throughout the pipeline
- ✅ MLflow integration for experiment tracking
- ✅ Detailed execution reports

### 5. Testing & Validation
- ✅ Unit tests for all components
- ✅ Comparison script to validate identical results
- ✅ Automated testing with pytest

## 📊 Data Cleaning Process

The refactored pipeline replicates the original notebook's cleaning process exactly:

1. **Drop unnecessary columns** (`mixed_type_col`)
2. **Clean text data** (whitespace, special characters)
3. **Handle N/A values** (all variations: N/A, nan, NaN, etc.)
4. **Convert numeric columns** with error handling
5. **Validate realistic ranges** and correct outliers
6. **Normalize categorical values** (proper casing)
7. **Impute missing values** (median for numeric, mode for categorical)

### Key Statistics

| Metric | Value |
|--------|-------|
| Input rows | 2,153 |
| Output rows | 2,153 (100% preserved) |
| Input columns | 18 |
| Output columns | 17 |
| Missing values removed | 620 |
| Outliers corrected | Based on realistic ranges |

## 🔬 MLflow Tracking

The pipeline automatically logs:
- Input/output dataset shapes
- Missing value counts
- Rows preserved percentage
- Execution time
- Dataset artifacts

View MLflow UI:
```bash
mlflow ui --port 5000
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t obesity-ml-project .

# Run container
docker run -v $(pwd)/data:/app/data obesity-ml-project

# Run with MLflow UI
docker-compose up
```

## 📝 Next Steps

- [x] Refactor EDA pipeline
- [x] Validate identical results
- [x] Create comprehensive tests
- [ ] Refactor ML pipeline
- [ ] Add model training and evaluation
- [ ] Create visualization dashboards
- [ ] Deploy model with API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 📞 Contact

For questions or issues, please contact the team members listed above.

---

**Status**: ✅ EDA Refactoring Complete - All Tests Passing - Results Validated
