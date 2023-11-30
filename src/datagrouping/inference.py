import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class DatasetEDA:
    """
    Class for performing exploratory data analysis (EDA) on a given dataset.
    """

    def __init__(self, data):
        """
        Constructor for DatasetEDA class.

        Parameters:
        - data (pd.DataFrame): The dataset to be analyzed.
        """
        self.data = data

    def displaySummary(self):
        """
        Display summary statistics of the dataset.
        """
        print(self.data.describe())

    def plotAmountByAgeGroup(self):
        """
        Plot the relationship between customer age group and the amount spent on products.
        """
        sns.boxplot(x="Age Group", y="Amount", data=self.data)
        plt.title("Amount Spent by Age Group")
        plt.show()

    def plotAmountByMaritalStatus(self):
        """
        Plot the distribution of product amounts based on marital status.
        """
        g = sns.FacetGrid(self.data, col="Marital_Status", height=4, col_wrap=3)
        g.map(sns.histplot, "Amount", bins=20, kde=True)
        g.set_axis_labels("Amount", "Density")
        g.set_titles(col_template="{col_name}")
        plt.show()

    def plotAmountByGender(self):
        """
        Plot the difference in product spending between males and females using a violin plot.

        Additionally, annotate the highest spender on the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            x="Gender", y="Amount", data=self.data, inner="quartile", palette="Pastel1"
        )
        highest_spender = self.data.loc[self.data["Amount"].idxmax()]
        plt.annotate(
            f'Highest Spender: {highest_spender["Amount"]:.2f}',
            xy=(highest_spender.name, highest_spender["Amount"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="red",
        )
        plt.title("Amount Spent by Gender")
        plt.show()

    def plotAmountByOccupation(self):
        """
        Plot the average amount spent by different occupations, highlighting the highest spender.

        The plot is a bar chart with occupations on the x-axis and the average amount spent on the y-axis.
        """
        plt.figure(figsize=(12, 6))
        top_occupation = self.data.groupby("Occupation")["Amount"].mean().idxmax()
        average_amount_by_occupation = (
            self.data.groupby("Occupation")["Amount"].mean().reset_index()
        )
        average_amount_by_occupation = average_amount_by_occupation.sort_values(
            by="Amount", ascending=True
        )
        sns.barplot(
            x="Occupation",
            y="Amount",
            data=average_amount_by_occupation,
            palette="viridis",
            order=average_amount_by_occupation["Occupation"],
        )
        highest_spender_value = self.data[self.data["Occupation"] == top_occupation][
            "Amount"
        ].mean()
        plt.annotate(
            f" : {top_occupation}\nAverage Amount: {highest_spender_value:.2f}",
            xy=(
                average_amount_by_occupation["Occupation"][
                    average_amount_by_occupation["Occupation"] == top_occupation
                ].index[0],
                highest_spender_value,
            ),
            xytext=(0, 10),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color="blue",
            weight="bold",
        )
        plt.title("Average Amount Spent by Occupation")
        plt.xlabel("Occupation")
        plt.ylabel("Average Amount Spent")
        plt.xticks(rotation=45, ha="right")
        plt.show()

    def plotAmountByState(self):
        """
        Plot the average amount spent by different states or zones.

        The plot is a bar chart with states on the y-axis and the average amount spent on the x-axis.
        """
        plt.figure(figsize=(14, 6))
        average_amount_by_state = (
            self.data.groupby("State")["Amount"].mean().reset_index()
        )
        average_amount_by_state = average_amount_by_state.sort_values(by="Amount")
        sns.barplot(
            x="Amount", y="State", data=average_amount_by_state, palette="viridis"
        )
        plt.title("Amount Spent by State")
        plt.xlabel("Average Amount Spent")
        plt.ylabel("State")
        plt.show()

    def plotAmountByProductCategory(self):
        """
        Plot the average amount spent on different product categories.

        The plot is a bar chart with product categories on the x-axis and the average amount spent on the y-axis.
        """
        plt.figure(figsize=(12, 6))
        average_amount_by_category = (
            self.data.groupby("Product_Category")["Amount"].mean().reset_index()
        )
        average_amount_by_category = average_amount_by_category.sort_values(
            by="Amount", ascending=False
        )
        sns.barplot(
            x="Product_Category",
            y="Amount",
            data=average_amount_by_category,
            palette="husl",
            order=average_amount_by_category["Product_Category"],
        )
        plt.title("Average Amount Spent by Product Category ")
        plt.xlabel("Product Category")
        plt.ylabel("Average Amount Spent")
        plt.xticks(rotation=45, ha="right")
        plt.show()

    def plotAmountVsOrders(self):
        """
        Plot the relationship between the number of orders placed and the total amount spent by a customer.

        The plot is a joint plot with the number of orders on the x-axis and the amount spent on the y-axis.
        """
        plt.figure(figsize=(10, 8))
        sns.jointplot(x="Orders", y="Amount", data=self.data, kind="reg", height=8)
        plt.suptitle("Total Amount Spent vs Number of Orders", y=1.02)
        plt.show()

    def plotAmountByAgeAndGender(self):
        """
        Plot the average amount spent by different age groups and genders.

        The plot is a bar chart with age groups on the x-axis, average amount spent on the y-axis,
        and different colors for different genders.
        """
        plt.figure(figsize=(12, 8))
        avg_amount_by_age_gender = (
            self.data.groupby(["Age Group", "Gender"])["Amount"].mean().reset_index()
        )
        sorted_data = avg_amount_by_age_gender.sort_values(
            by="Age Group", key=lambda x: x.map({"18-34": 1, "35-54": 2, "55+": 3})
        )
        sns.barplot(
            x="Age Group",
            y="Amount",
            hue="Gender",
            data=sorted_data,
            ci="sd",
            capsize=0.2,
            palette="Set3",
        )
        plt.title("Average Amount Spent by Age Group and Gender")
        plt.xlabel("Age Group")
        plt.ylabel("Average Amount Spent")
        plt.show()
