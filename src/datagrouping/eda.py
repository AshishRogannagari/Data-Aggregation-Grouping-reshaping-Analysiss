# importing the required library's
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


class EdaVisualizer:
    """
    Class for visualizing exploratory data analysis (EDA) of a dataset.
    """

    def __init__(self, data):
        """
        Constructor for EdaVisualizer class.

        Parameters:
        - data (pd.DataFrame): The dataset to be visualized.
        """
        self.data = data
        sns.set(style="whitegrid")

    def plotAgeDistributionMatplotlib(self):
        """
        Plot the distribution of ages in the dataset using matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(
            self.data["Age"], bins=20, color="skyblue", edgecolor="black", alpha=0.7
        )
        plt.title("Distribution of Age (matplotlib)")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

    def plotAgeDistributionSeaborn(self):
        """
        Plot the distribution of ages in the dataset using seaborn.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data["Age"], bins=20, kde=True, color="skyblue")
        plt.title("Distribution of Age (seaborn)")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

    def plotTopStatesCustomersMatplotlib(self):
        """
        Plot the top 10 states by the number of customers using matplotlib.
        """
        top_states_counts = self.data["State"].value_counts().nlargest(10)

        plt.figure(figsize=(12, 6))
        plt.bar(top_states_counts.index, top_states_counts.values, color="skyblue")

        plt.title("Top 10 States by Number of Customers (matplotlib)")
        plt.xlabel("State")
        plt.ylabel("Number of Customers")

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45, ha="right")

        plt.show()

    def plotTopStatesCustomersSeaborn(self):
        """
        Plot the top 10 states by the number of customers using seaborn.
        """
        top_states_counts = self.data["State"].value_counts().nlargest(10)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x=top_states_counts.index, y=top_states_counts.values, palette="viridis"
        )

        ax.set_title("Top 10 States by Number of Customers (seaborn)")
        ax.set_xlabel("State")
        ax.set_ylabel("Number of Customers")

        # Rotate x-axis labels for better visibility
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        plt.show()

    def highestSalesForProductsMatplotlib(self):
        """
        Plot the top 10 products by the amount spent using matplotlib.
        """
        top_products = (
            self.data.groupby("Product_Category")["Amount"].sum().reset_index()
        )
        top_products = top_products.sort_values(by="Amount", ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        plt.barh(
            top_products["Product_Category"], top_products["Amount"], color="green"
        )

        plt.title("Top 10 Products by Amount Spent (matplotlib)")
        plt.xlabel("Amount Spent")
        plt.ylabel("Product Category")

        plt.show()

    def highestSalesForProductsSeaborn(self):
        """
        Plot the top 10 products by the amount spent using seaborn.
        """
        top_products = (
            self.data.groupby("Product_Category")["Amount"].sum().reset_index()
        )
        top_products = top_products.sort_values(by="Amount", ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x=top_products["Amount"],
            y=top_products["Product_Category"],
            palette="viridis",
        )

        ax.set_title("Top 10 Products by Amount Spent (seaborn)")
        ax.set_xlabel("Amount Spent")
        ax.set_ylabel("Product Category")

        plt.show()

    def plotAmountSpentMatplotlib(self):
        """
        Plot the distribution of the amount spent in the dataset using matplotlib.
        """
        plt.figure(figsize=(15, 5))
        plt.hist(
            self.data["Amount"], bins=50, color="salmon", edgecolor="black", alpha=0.7
        )
        plt.title("Distribution of Amount Spent (matplotlib)")
        plt.xlabel("Amount Spent")
        plt.ylabel("Frequency")
        plt.show()

    def plotAmountSpentSeaborn(self):
        """
        Plot the distribution of the amount spent in the dataset using seaborn.
        """
        plt.figure(figsize=(15, 5))
        sns.histplot(self.data["Amount"], bins=50, kde=True, color="salmon")
        plt.title("Distribution of Amount Spent (seaborn)")
        plt.xlabel("Amount Spent")
        plt.ylabel("Frequency")
        plt.show()

    def generateProductsWordCloud(self):
        """
        Generate and display a Word Cloud for product categories in the dataset.
        """
        # Concatenate all product names
        product_names = " ".join(self.data["Product_Category"].astype(str))

        # Generate WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            product_names
        )

        # Display the WordCloud using matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud for Products")
        plt.show()
