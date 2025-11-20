import json
import random
import time
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # base_url="https://cmu.litellm.ai",
    base_url="https://ai-gateway.andrew.cmu.edu/")

# ------------------------------------------
# DOMAIN-SPECIFIC COLUMN SCHEMAS
# ------------------------------------------
SCHEMAS = {
    "finance": [
        "transaction_amount", "merchant_category", "card_type",
        "country", "is_fraud", "account_balance", "income"
    ],
    "medical": [
        "age", "blood_pressure", "heart_rate", "diagnosis_code",
        "cholesterol", "treatment_duration", "is_readmitted"
    ],
    "ecommerce": [
        "price", "category", "product_rating", "num_reviews",
        "user_id", "purchase_time", "is_returned"
    ],
    "climate": [
        "temperature", "humidity", "co2_ppm", "wind_speed",
        "precipitation", "region", "day_of_year"
    ]
}

DOMAINS = list(SCHEMAS.keys())

# ------------------------------------------
# TEMPLATES FOR EACH AGENT
# ------------------------------------------
TEMPLATES = {
    "cleaning": [
        "Remove missing values in [col1] and fill missing values in [col2] with median.",
        "Convert [col1] to numeric and strip whitespace from [col2].",
        "Handle outliers using IQR",
        "Remove rows where [col1] is negative.",
        "Drop duplicate rows based on [col1] and [col2]."
        "Strip whitespace from string columns",
        "Convert [col1] to datetime format.",
        "Rename columns [col2] into snake_case",
        "Drop irrelevant columns",
        "Fill all missing values using median or mode"
    ],

    "eda": [
        "Compute descriptive statistics for [col1], [col2], and [col3].",
        "Generate value counts for [col1].",
        "Group by [col1] and compute mean of [col2].",
        "Find correlation between [col1] and [col2].",
        "List top 5 categories of [col1] by count.",
        "Compute correlations",
        "Identify missing values per column",
        "Summarize unique values in [col1]",
        "Group by categorical column [col1] and summarize",
        "Produce dataset shape and column info",
        "Identify numeric vs categorical columns"
    ],

    "visualization": [
        "Plot histogram of [col1].",
        "Create scatter plot of [col1] vs [col2] colored by [col3].",
        "Plot correlation heatmap of numeric features.",
        "Plot boxplot of [col1] grouped by [col2].",
        "Visualize category frequencies using bar chart.",
        "Plot pairplot for [col1], [col2], [col3].",
        "Plot time series line chart."
    ],

    "feature_engineering": [
        "Create new feature [col1]_ratio as [col1] divided by [col2].",
        "One-hot encode the column [col1].",
        "Normalize [col1] using MinMaxScaler.",
        "Extract year, month, and day from [col1].",
        "Bucketize [col1] into groups.",
        "Generate binary flag feature",
    ],

    "modeling": [
        "Train a logistic regression model to predict [target] using [col1], [col2], [col3].",
        "Train a random forest classifier using numeric columns.",
        "Train linear regression model.",
        "Split dataset into train and test using [col1] as target.",
        "Train a decision tree and compute accuracy.",
        "Evaluate model on [target] using F1 score.",
        "Perform cross-validation with 5 folds.",
        "Train a linear regression model predicting [target] from [feature_list]."
    ],

    "statistics": [
        "Compute Pearson correlation between [col1] and [col2].",
        "Perform t-test between two groups",
        "Perform chi-square test between [col1] and [col2].",
        "Compute ANOVA for [col1] grouped by [col2].",
        "Calculate covariance matrix for numeric columns.",
        "Generate summary of central tendency.",
    ]
}

SYSTEM_PROMPT = """
You are a Python data science expert. 
Only output Python code (no text).
Use pandas, numpy, seaborn, matplotlib, sklearn, and scipy.
Assume the dataframe is named df.
"""

def fill_template(template, domain):
    cols = SCHEMAS[domain]
    col1 = random.choice(cols)
    col2 = random.choice(cols)
    col3 = random.choice(cols)
    target = random.choice(cols)
    feature_list = ", ".join(random.sample(cols, k=min(3, len(cols))))

    return (template
            .replace("[col1]", col1)
            .replace("[col2]", col2)
            .replace("[col3]", col3)
            .replace("[target]", target)
            .replace("[feature_list]", feature_list)
            .replace("[domain]", domain)
    )

def generate_example(agent):
    domain = random.choice(DOMAINS)
    template = random.choice(TEMPLATES[agent])
    instruction = fill_template(template, domain)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ],
        temperature=0.5
    )

    return {
        "instruction": instruction,
        "domain": domain,
        "input": "df",
        "output_code": response.choices[0].message.content.strip()
    }

def generate(agent, n=50):
    filename = f"{agent}_synthetic.jsonl"
    with open(filename, "w") as f:
        for _ in range(n):
            ex = generate_example(agent)
            f.write(json.dumps(ex) + "\n")
            time.sleep(0.2)
    print(f"Generated {n} examples for {agent} in {filename}")

if __name__ == "__main__":
    for agent in ["cleaning", "eda", "visualization", "feature_engineering", "modeling", "statistics"]:
        generate(agent, n=2)