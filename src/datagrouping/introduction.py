import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataIntro:
    def __init__(self, df):
        self.df = df

    def Data_Details(self):
        print(
            f"""The dataset provided contains information about customers and their purchasing behavior.
            .The fields include User_ID, Cust_name, Product_ID, Gender, Age Group, Age, Marital_Status, State, Zone, Occupation, Product_Category, Orders, Amount, Status, and an unnamed column.
            .The data seems to capture diverse demographic and transactional details, offering an opportunity for valuable insights into customer preferences and market trends.
            .Upon initial analysis, several interesting patterns emerge.
            .The dataset encompasses a range of ages, from minors (0-17 years old) to seniors (55+ years old).
            .The majority falls within the 26-35 age group, suggesting a broad customer base.
            .The gender distribution is fairly balanced, with male and female customers both actively participating in transactions.
             """
        )
        return
