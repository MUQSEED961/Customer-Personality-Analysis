# Imports
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture
from mlxtend.frequent_patterns import apriori, association_rules


# Load Data
data = pd.read_csv(r"intership\marketing_campaign.csv", sep=",")
# using default comma


# ——— HEADER CLEANUP (Step B): start here ———
print("✴️ Raw columns:", data.columns.tolist())

# Fix typo if present (you mentioned 'Year_Birtth' before)
data.rename(columns={'Year_Birtth':'Year_Birth'}, inplace=True)

# Standardize all column names to lowercase with underscores
data.columns = data.columns.astype(str) \
    .str.strip() \
    .str.replace(' ', '_') \
    .str.lower()

print("🛠️ Normalized cols:", list(data.columns))

# Rename snake_case to title case if code uses TitlеCase consistently
snake_to_title = {
    'year_birth': 'Year_Birth',
    'dt_customer': 'Dt_Customer',
    'recency': 'Recency',
    'marital_status': 'Marital_Status',
    'education': 'Education',
    'mntwines': 'MntWines',
    'mntfruits': 'MntFruits',
    'mntmeatproducts': 'MntMeatProducts',
    'mntfishproducts': 'MntFishProducts',
    'mntsweetproducts': 'MntSweetProducts',
    'mntgoldprods': 'MntGoldProds',
    'numwebpurchases': 'NumWebPurchases',
    'numcatalogpurchases': 'NumCatalogPurchases',
    'numstorepurchases': 'NumStorePurchases'
}
snake_to_title.update({
    'kidhome': 'Kidhome',
    'teenhome': 'Teenhome',
    'income': 'Income'
})
# Only rename columns that actually exist
data.rename(columns={k:v for k, v in snake_to_title.items() if k in data.columns}, inplace=True)

print("✅ Renamed columns:", [col for col in snake_to_title.values() if col in data.columns])

# Fail early if critical columns are still missing
required = ['Year_Birth', 'Dt_Customer', 'Recency', 'Marital_Status', 'Education']
missing = [c for c in required if c not in data.columns]
if missing:
    raise KeyError(f"❗ Required columns missing after cleanup: {missing}")

# ——— HEADER CLEANUP: end here ———

# -----------------
# Feature Engineering
# -----------------
# Age
data['Age'] = 2014 - data['Year_Birth']

# Total Spending
product_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
data['Spending'] = data[product_cols].sum(axis=1)

# Seniority in months
last_date = date(2014, 10, 4)
data['Seniority'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True, errors='coerce')
data['Seniority'] = data['Seniority'].apply(lambda x: (last_date - x.date()).days if pd.notnull(x) else np.nan)/30

# Standardize column names
data = data.rename(columns={
    'NumWebPurchases': "Web",
    'NumCatalogPurchases': 'Catalog',
    'NumStorePurchases': 'Store'
})

# Group Marital and Education levels
data['Marital_Status'] = data['Marital_Status'].replace({
    'Divorced': 'Alone', 'Single': 'Alone', 'Married': 'In couple', 'Together': 'In couple',
    'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone'
})
data['Education'] = data['Education'].replace({
    'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate',
    'Graduation': 'Postgraduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'
})

# Children and child flags
data['Children'] = data['Kidhome'] + data['Teenhome']
data['Has_child'] = np.where(data['Children'] > 0, 'Has child', 'No child')

# Human-readable children count
data['Children'].replace({3: "3 children",2:'2 children',1:'1 child',0:"No child"}, inplace=True)

# More user-friendly product columns
data.rename(columns={
    'MntWines': 'Wines', 'MntFruits': 'Fruits',
    'MntMeatProducts': 'Meat', 'MntFishProducts': 'Fish',
    'MntSweetProducts': 'Sweets', 'MntGoldProds': 'Gold'
}, inplace=True)

# Prepare clustering input
df_cluster = data[['Income', 'Seniority', 'Spending']].copy()

# Remove outliers & missing
df_cluster = df_cluster.dropna(subset=['Income','Seniority','Spending'])
df_cluster = df_cluster[df_cluster['Income'] < 600000]

# Normalize for clustering
scaler = StandardScaler()
X_std = scaler.fit_transform(df_cluster)
X = normalize(X_std, norm='l2')

# -----------------
# Clustering: Gaussian Mixture
# -----------------
gmm = GaussianMixture(n_components=4, covariance_type='spherical', max_iter=2000, random_state=5).fit(X)
labels = gmm.predict(X)

# Cluster labels
segments = {0:'Stars', 1:'Need attention', 2:'High potential', 3:'Leaky bucket'}
df_cluster['Cluster'] = [segments.get(label, label) for label in labels]

# Merge labels back
data = data.loc[df_cluster.index]  # match indices after dropping
data['Cluster'] = df_cluster['Cluster'].values

# ---------------
# Segments (Age/Income/Seniority bins)
# ---------------
# Create Age group
cut_labels_Age = ['Young', 'Adult', 'Mature', 'Senior']
cut_bins = [0, 30, 45, 65, 120]
data['Age_group'] = pd.cut(data['Age'], bins=cut_bins, labels=cut_labels_Age)

# Income group (quartiles)
cut_labels_Income = ['Low income', 'Low to medium income', 'Medium to high income', 'High income']
data['Income_group'] = pd.qcut(data['Income'], q=4, labels=cut_labels_Income)

# Seniority group (quartiles)
cut_labels_Seniority = ['New customers', 'Discovering customers', 'Experienced customers', 'Old customers']
data['Seniority_group'] = pd.qcut(data['Seniority'], q=4, labels=cut_labels_Seniority)

# Drop original values if desired
data = data.drop(columns=['Age', 'Income', 'Seniority'])

# ---------------
# Product Consumption Segments (qcut on >0 buyers)
# ---------------
cut_labels_product = ['Low consumer', 'Frequent consumer', 'Biggest consumer']
for prod in ['Wines','Fruits','Meat','Fish','Sweets','Gold']:
    seg_name = f'{prod}_segment'
    data[seg_name] = 'Non consumer'
    buyers = data[prod] > 0
    data.loc[buyers, seg_name] = pd.qcut(data.loc[buyers, prod], q=[0, .25, .75, 1], labels=cut_labels_product).astype(str)

# Replace nans left behind in segment labels
data = data.astype(object)
data.fillna('Non consumer', inplace=True)


# Drop numeric product columns if not needed for Apriori
data.drop(columns=['Spending','Wines','Fruits','Meat','Fish','Sweets','Gold'], inplace=True)

# Set all columns as object types (category)
data = data.astype(object)

# ---------------
# Prepare for Apriori
# ---------------
# ... (your data cleaning / segmentation code above)

# ---------------
# Prepare for Apriori
# ---------------

# ---------------
# Mine Frequent Itemsets (with FP‑Growth for speed)
# ---------------

min_support = 0.08
max_len     = 10

# ⚡ Critical: cast to Boolean
df_apriori = pd.get_dummies(data).astype(bool)

# Use FP‑Growth instead of Apriori for large or wide data
from mlxtend.frequent_patterns import fpgrowth

frequent_items = fpgrowth(
    df_apriori,
    min_support=min_support,
    use_colnames=True,
    max_len=max_len + 1,
    verbose=1
)

rules = association_rules(
    frequent_items,
    metric="lift",
    min_threshold=1
)


# Find rules for 'Wines_segment_Biggest consumer'
product = 'Wines'
segment = 'Biggest consumer'
target = f"Wines_segment_{segment}"
target_mask = rules['consequents'].astype(str).str.contains(target, na=False)
results = rules[target_mask].sort_values(by='confidence', ascending=False)

# Display best association rules for 'Biggest consumer' of wines
print(results[['antecedents','consequents','support','confidence','lift']].head(10))
