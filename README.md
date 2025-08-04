# Customer Personality Clustering & Rule-Mining

**Overview**  
This Python script builds a robust customer segmentation and association-rule analysis pipeline using the “Customer Personality Analysis” dataset. It:

- Cleans and standardizes raw data (e.g. fixes headers, handles typos, imputes missing values)
- Engineers features like Age, Spending, Seniority, and child indicators
- Segments customers via **Gaussian Mixture Model** into four behavioral clusters (Stars, High Potential, Need Attention, Leaky Bucket)
- Bins demographic and consumption behavior into tiered segments (e.g. wine spending, age, income)
- Converts segments into a boolean one-hot matrix and mines association rules using fast **FP‑Growth** (instead of slower Apriori) to identify which feature combinations predict being a *biggest wine consumer*

Execution completes in minutes even on wide transaction datasets, thanks to FP‑Growth and boolean encoding optimizations.

---

## 🛠 Installation

Create and activate a virtual environment, then run:

```bash
pip install numpy pandas scikit-learn matplotlib mlxtend
```

- **mlxtend** provides frequent-pattern mining (FP‑Growth and association_rules) and must be installed (via `pip install mlxtend`) — it is indispensable for this workflow :contentReference[oaicite:1]{index=1}.
- Compatible with Python 3.7–3.11. mlxtend version 0.23.4 is recommended for stability :contentReference[oaicite:2]{index=2}.

---

## 🧪 Usage

```bash
python intership1.py
```

Make sure the `marketing_campaign.csv` file is located under `./intership/`. Adjust thresholds (`min_support`, `max_len`) in the README as needed to customize rule patterns.

---

## License

This project is released under the **MIT License**—feel free to adapt and extend. Mention me if you repurpose or improve the script!
