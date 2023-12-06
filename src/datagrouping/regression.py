# Import necessary libraries
import pandas as pd  # Pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Matplotlib for data visualization
import seaborn as sns  # Seaborn for statistical data visualization
from sklearn.model_selection import (
    train_test_split,
)  # For splitting the data into training and testing sets
from sklearn.linear_model import (
    LinearRegression,
)  # Linear Regression model from scikit-learn
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)  # Evaluation metrics for regression models
from sklearn.preprocessing import (
    OneHotEncoder,
)  # One-hot encoding for categorical features
from sklearn.compose import (
    ColumnTransformer,
)  # For applying transformations to specific columns
from sklearn.pipeline import (
    Pipeline,
)  # For creating a pipeline to streamline the workflow
import warnings

# Ignore FutureWarnings to reduce clutter in the output
warnings.filterwarnings("ignore", category=FutureWarning)


class LinearRegressionModel:
    """
    This class provides methods for training and evaluating a linear regression model on
    a dataset provided during class instantiation. The dataset is expected to be a Pandas
    DataFrame with features and a target (Amount) for predicting continuous values. It
    also includes methods for plotting diagnostic charts related to the regression model
    performance.

    Methods
    -------
    trainAndEvaluateModel(self):
        Trains the linear regression model using a pipeline that includes preprocessing
        with OneHotEncoding for categorical features and evaluation metrics computation.
        Stores training and testing data, predictions, and feature importance internally.

    plotActualVsPredicted(self):
        Plots a scatterplot comparing actual vs. predicted values obtained from the
        model along with the regression line, if the model has been trained.

    plotResiduals(self):
        Plots a histogram of residuals (differences between the actual and the predicted
        values), if the model has been trained.

    plotDistribution(self):
        Plots the distribution of the predictions and the actual target values as kernel
        density estimations to assess how well they align, if the model has been trained.

    plotFeatureImportance(self):
        Plots the feature importances derived from the regression model's coefficients if
        available (model should have the attribute coef_).

    Attributes
    ----------
    data : DataFrame
        The input data used for model training and evaluation.
    XTrain : DataFrame
        The features of the training data.
    XTest : DataFrame
        The features of the testing data.
    yTrain : Series
        The target variable of the training data.
    yTest : Series
        The target variable of the testing data.
    predictions : Series or array-like
        The predicted values produced by the model.
    featureImportance : Series
        A pandas Series holding the importance of each feature according to the trained
        linear regression model.

    """

    def __init__(self, data):
        """
        Initialize the LinearRegressionModel instance.

        Parameters:
        - data (pd.DataFrame): The input data for model training and evaluation.
        """
        self.data = data
        self.XTrain = None
        self.XTest = None
        self.yTrain = None
        self.yTest = None
        self.predictions = None
        self.featureImportance = None

    def trainAndEvaluateModel(self):
        """
        Train and evaluate the linear regression model.

        Splits the data into training and testing sets, preprocesses categorical
        features using OneHotEncoder, and computes evaluation metrics. Stores training
        and testing sets, predictions, and feature importance internally.
        """
        # Split the data into training and testing sets
        X = self.data.drop(columns=["Amount"])
        y = self.data["Amount"]
        XTrain, XTest, yTrain, yTest = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Store training and testing sets
        self.XTrain = XTrain
        self.XTest = XTest
        self.yTrain = yTrain
        self.yTest = yTest

        # Preprocessing: Use OneHotEncoder for categorical variables
        categoricalFeatures = [
            "Gender",
            "Age Group",
            "Marital_Status",
            "State",
            "Zone",
            "Occupation",
            "Product_Category",
        ]
        numericFeatures = ["Age", "Orders"]

        # Create a preprocessor using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numericFeatures),
                ("cat", OneHotEncoder(), categoricalFeatures),
            ]
        )

        # Create a pipeline with preprocessing and Linear Regression model
        model = Pipeline(
            [("preprocessor", preprocessor), ("regressor", LinearRegression())]
        )

        # Train the model
        model.fit(XTrain, yTrain)

        # Make predictions on the test set
        predictions = model.predict(XTest)

        # Store predictions
        self.predictions = predictions

        # Evaluate the model
        mse = mean_squared_error(yTest, predictions)
        r2 = r2_score(yTest, predictions)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        # Extract feature importance if the model has coefficients
        if hasattr(model.named_steps["regressor"], "coef_"):
            featureImportance = model.named_steps["regressor"].coef_

            # Get one-hot encoded feature names
            oneHotEncoder = model.named_steps["preprocessor"].named_transformers_["cat"]
            featureNames = numericFeatures + list(
                oneHotEncoder.get_feature_names_out(categoricalFeatures)
            )

            # Combine numeric and categorical feature names
            self.featureImportance = pd.Series(featureImportance, index=featureNames)

    def plotActualVsPredicted(self):
        """
        Plot a scatterplot comparing actual vs. predicted values.

        Plots a scatterplot of actual vs. predicted values along with the regression
        line, if the model has been trained.
        """
        # Check if predictions and actual values are available
        if self.yTest is None or self.predictions is None:
            print("Model has not been trained and evaluated.")
            return

        # Set a Seaborn style for better aesthetics
        sns.set(style="whitegrid")

        # Create a scatter plot of actual vs. predicted values using Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=self.yTest, y=self.predictions, alpha=0.5, label="Actual vs. Predicted"
        )
        sns.regplot(
            x=self.yTest,
            y=self.predictions,
            scatter=False,
            color="blue",
            label="Regression Line",
        )

        plt.title("Actual vs. Predicted Values with Regression Line")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.show()

    def plotResiduals(self):
        """
        Plot a histogram of residuals.

        Plots a histogram of residuals (differences between the actual and the predicted
        values), if the model has been trained.
        """
        # Check if predictions and actual values are available
        if self.yTest is None or self.predictions is None:
            print("Model has not been trained and evaluated.")
            return

        # Create a residual plot
        residuals = self.yTest - self.predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color="blue", bins=30)
        plt.title("Residual Plot")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.axvline(
            x=0, color="red", linestyle="--", linewidth=2, label="Zero Residuals Line"
        )
        plt.legend()
        plt.show()

    def plotDistribution(self):
        """
        Plot the distribution of predictions and actual values.

        Plots the distribution of the predictions and the actual target values as kernel
        density estimations to assess how well they align, if the model has been trained.
        """
        # Check if predictions and actual values are available
        if self.yTest is None or self.predictions is None:
            print("Model has not been trained and evaluated.")
            return

        # Create a distribution plot of predictions and actual values
        plt.figure(figsize=(12, 6))
        sns.kdeplot(self.predictions, label="Predictions", fill=True)
        sns.kdeplot(self.yTest, label="Actual Values", fill=True)
        plt.title("Distribution of Predictions and Actual Values")
        plt.xlabel("Amount")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def plotFeatureImportance(self):
        """
        Plot the feature importance.

        Plots the feature importances derived from the regression model's coefficients if
        available (model should have the attribute coef_).
        """
        # Check if feature importance is available
        if self.featureImportance is None:
            print("Feature importance is not available.")
            return

        # Sort feature importance values and names
        sortedFeatureImportance = self.featureImportance.abs().sort_values(
            ascending=False
        )
        sortedFeatureNames = sortedFeatureImportance.index

        # Set colors for positive and negative coefficients
        colors = [
            "green" if coef > 0 else "red"
            for coef in self.featureImportance[sortedFeatureNames]
        ]

        # Create a bar plot of feature importance
        plt.figure(figsize=(15, 10))
        sns.barplot(x=sortedFeatureImportance, y=sortedFeatureNames, palette=colors)
        plt.title("Feature Importance")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Features")
        plt.show()
