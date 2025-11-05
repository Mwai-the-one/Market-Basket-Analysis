🛒 Market Basket Analysis using Apriori Algorithm

This project demonstrates how to perform Market Basket Analysis — a popular data mining technique used to uncover relationships between items in transactional data — using the Apriori algorithm.

It identifies frequent itemsets and generates association rules (e.g., “If a customer buys Bread, they are likely to also buy Eggs”).

Overview

The main objective of this project is to:

Identify frequent combinations of products purchased together.

Derive association rules to describe relationships between items.

Demonstrate how to use Python’s mlxtend library for Apriori-based association rule mining.

This technique is widely used in retail analytics, recommendation systems, and consumer behavior modeling.


🧰 Technologies Used

Library	Purpose

pandas	Data manipulation and analysis

mlxtend	Provides tools for machine learning extensions, including Apriori and association rules

TransactionEncoder	Converts lists of transactions into a one-hot encoded DataFrame

apriori	Finds frequent itemsets based on support threshold

association_rules	Generates rules based on confidence, lift, and support

⚙️ How It Works

Data Preparation – Represent each shopping basket as a list of items.

One-Hot Encoding – Convert transactions into a binary (True/False) format where each column corresponds to an item.

Frequent Itemset Generation – Use the Apriori algorithm to find item combinations that appear frequently together.

Association Rule Mining – Extract rules such as “If item A is bought, item B is likely to be bought too”, with measures like:

Support: Frequency of the itemset in all transactions.

Confidence: Likelihood that a consequent is bought when the antecedent is bought.

Lift: Strength of the rule compared to random chance.

🛍️ Dataset
I used this simple database for the demonstration:
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Dipers', 'Beer', 'Eggs'],
    ['Milk', 'Dipers', 'Bread', 'Cola'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Dipers', 'Eggs'],
    ['Cola', 'Dipers', 'Beer'],
    ['Milk', 'Bread', 'Cola'],
    ['Bread', 'Dipers', 'Eggs']
]

Each inner list represents a single shopping basket (a transaction).

🧩 Code Walkthrough
1. Import Libraries🧩:

[import pandas as pd

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori, association_rules]

2. Encode Transactions:
[te = TransactionEncoder()

te_data = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_data, columns=te.columns_)]

TransactionEncoder transforms the list of lists into a binary matrix.

Each column represents an item, with True meaning the item is in the basket

3. Generate Frequent Itemsets:
[frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)]

min_support=0.6 means only itemsets appearing in at least 60% of all transactions are considered frequent.

use_colnames=True displays actual item names instead of column indices.

4. Generate Association Rules:

[rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)]

metric="confidence": filter rules by confidence level.

min_threshold=0.7: only keep rules with at least 70% confidence.

📊 Output Interpretation

Support: Frequency of occurrence across all transactions.

Confidence: Probability that the consequent is bought when the antecedent is bought.

Lift: Indicates rule strength:

Lift > 1 → Positive correlation (items tend to be bought together).

Lift = 1 → No correlation.

Lift < 1 → Negative correlation.
