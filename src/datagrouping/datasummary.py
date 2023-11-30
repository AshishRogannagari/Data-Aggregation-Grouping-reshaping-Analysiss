class DataSummary:
    def __init__(self, df):
        """
        Create an object by passing in a `pandas.DataFrame` of data.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        """
        self.data = df

    def show(self):
        """
        Display the first 10 rows of the dataframe.

        Returns:
        - pd.DataFrame: The first 10 rows of the dataframe.
        """
        return self.data.head(10)

    def shape(self):
        """
        Get the shape of the dataframe.

        Returns:
        - tuple: The shape (number of rows, number of columns) of the dataframe.
        """
        return self.data.shape

    def dataTypes(self):
        """
        Display the data types of each attribute in the dataset.

        Returns:
        - pd.Series: Data types for each attribute.
        """
        print("Data Types for Each Attribute:")
        return self.data.dtypes

    def checkNull(self):
        """
        Check for missing values in the dataset.

        Returns:
        - pd.Series: The number of null values in each column.
        """
        return self.data.isnull().sum()

    def printColumns(self):
        """
        Print the column names of the dataset.

        Returns:
        - list: The column names of the dataset.
        """
        print("Column Names:")
        return self.data.columns

    def removeColumn(self):
        """
        Remove specific columns from the dataset.

        Returns:
        - pd.DataFrame: The dataset after removing specified columns.
        """
        self.data.drop(columns=["Unnamed: 13", "Unnamed: 14"], inplace=True)
        return self.data
