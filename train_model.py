import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pk

FILE = 'Cardetails.csv'  # change path here if needed

# ---------- Diagnostics: file existence and size ----------
if not os.path.exists(FILE):
    print(f"❌ File not found at: {os.path.abspath(FILE)}")
    sys.exit(1)

size = os.path.getsize(FILE)
if size == 0:
    print(f"❌ File '{FILE}' is empty (0 bytes). Re-download or provide a valid CSV.")
    sys.exit(1)

print(f"✅ Found file: {os.path.abspath(FILE)} (size: {size} bytes)")

# ---------- Try reading CSV with safe options ----------
def try_read_csv(path):
    # Try default read first
    try:
        df = pd.read_csv(path)
        print("✅ read_csv succeeded with default settings.")
        return df
    except Exception as e:
        print("⚠️ read_csv default failed:", repr(e))

    # Try python engine and auto sep detection
    try:
        df = pd.read_csv(path, engine='python', sep=None)
        print("✅ read_csv succeeded with engine='python' and sep=None.")
        return df
    except Exception as e:
        print("⚠️ read_csv with engine='python' failed:", repr(e))

    # If file may be an Excel file renamed to .csv, try read_excel
    try:
        df = pd.read_excel(path)
        print("✅ read_excel succeeded (file might be xlsx renamed to csv).")
        return df
    except Exception as e:
        print("⚠️ read_excel failed too:", repr(e))

    raise ValueError("Unable to read file. Check file format/encoding.")

cars_data = try_read_csv(FILE)

# ---------- Quick peek at file structure ----------
print("\n--- File peek ---")
print("Shape:", cars_data.shape)
print("Columns:", list(cars_data.columns[:40]))
print("Dtypes:\n", cars_data.dtypes)
print("\nFirst 5 rows:\n", cars_data.head().to_string(index=False))

# ---------- Ensure required columns exist ----------
required_cols = {'name','year','selling_price','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'}
missing = required_cols - set(cars_data.columns.str.lower())  # using lowercase to be tolerant
if missing:
    print(f"\n⚠️ Required columns missing (case-insensitive): {missing}")
    print("Please make sure CSV has headers exactly like:", required_cols)
    # continue anyway — maybe user will fix and rerun
else:
    print("\n✅ All required columns appear present (case-insensitive check).")

# ---------- Standardize column names to lowercase (helps avoid capitalization issues) ----------
cars_data.columns = [c.strip() for c in cars_data.columns]
cols_lower = {c:c.lower() for c in cars_data.columns}
cars_data.rename(columns=cols_lower, inplace=True)

# ---------- Drop 'torque' safely only if present ----------
if 'torque' in cars_data.columns:
    cars_data.drop(columns=['torque'], inplace=True)
    print("Dropped 'torque' column.")
else:
    print("'torque' column not found — skipping drop.")

# ---------- Drop NA and duplicates (but show counts first) ----------
print(f"\nRows before dropna/drop_duplicates: {len(cars_data)}")
na_counts = cars_data.isna().sum()
print("Null counts (top 10):\n", na_counts.sort_values(ascending=False).head(10))

cars_data = cars_data.dropna()
cars_data = cars_data.drop_duplicates()
print(f"Rows after dropping NA & duplicates: {len(cars_data)}")

# ---------- Safe cleaning functions ----------
def get_brand_name(val):
    try:
        s = str(val).strip()
        if s == '' or s.lower() == 'nan':
            return 'Unknown'
        return s.split(' ')[0].strip()
    except Exception:
        return 'Unknown'

def clean_data_num(value):
    try:
        if pd.isna(value):
            return 0.0
        s = str(value).strip()
        # If value like "18 kmpl" or "1200 cc", take first token and remove commas
        token = s.split(' ')[0].replace(',','')
        if token == '' or token.lower() == 'nan':
            return 0.0
        return float(token)
    except Exception:
        return 0.0

# ---------- Apply cleaning (but protect if column missing) ----------
if 'name' in cars_data.columns:
    cars_data['name'] = cars_data['name'].apply(get_brand_name)
else:
    print("Warning: 'name' column missing.")

for c in ['mileage','max_power','engine']:
    if c in cars_data.columns:
        cars_data[c] = cars_data[c].apply(clean_data_num)
    else:
        print(f"Warning: '{c}' column missing.")

# ---------- Encode categorical values with mappings (safe replacement) ----------
brand_list = ['Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault',
       'Mahindra','Tata','Chevrolet','Datsun','Jeep','Mercedes-Benz',
       'Mitsubishi','Audi','Volkswagen','BMW','Nissan','Lexus',
       'Jaguar','Land','MG','Volvo','Daewoo','Kia','Fiat','Force',
       'Ambassador','Ashok','Isuzu','Opel']

brand_map = {b: i+1 for i,b in enumerate(brand_list)}
# default unknown brands to 0
cars_data['name'] = cars_data['name'].map(lambda x: brand_map.get(x, 0))

# Safe mapping for other categorical columns
if 'transmission' in cars_data.columns:
    cars_data['transmission'] = cars_data['transmission'].map({'Manual':1,'Automatic':2}).fillna(0)
if 'seller_type' in cars_data.columns:
    cars_data['seller_type'] = cars_data['seller_type'].map({'Individual':1,'Dealer':2,'Trustmark Dealer':3}).fillna(0)
if 'fuel' in cars_data.columns:
    cars_data['fuel'] = cars_data['fuel'].map({'Diesel':1,'Petrol':2,'LPG':3,'CNG':4}).fillna(0)
if 'owner' in cars_data.columns:
    cars_data['owner'] = cars_data['owner'].map({'First Owner':1,'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4,'Test Drive Car':5}).fillna(0)

# ---------- Reset index ----------
cars_data.reset_index(drop=True, inplace=True)

# ---------- Check that selling_price exists and is numeric ----------
if 'selling_price' not in cars_data.columns:
    print("❌ 'selling_price' column not found. Cannot train model without the target column.")
    sys.exit(1)

# Convert selling_price to numeric (strip spaces, currency signs if any)
def to_numeric_price(x):
    try:
        s = str(x).strip().replace(',','')
        # remove currency symbols if present
        s = s.replace('₹','').replace('rs','').replace('Rs','').strip()
        return float(s)
    except Exception:
        return np.nan

cars_data['selling_price'] = cars_data['selling_price'].apply(to_numeric_price)
print("selling_price nulls after conversion:", cars_data['selling_price'].isna().sum())
cars_data.dropna(subset=['selling_price'], inplace=True)

# ---------- Prepare X and y ----------
feature_cols = [c for c in cars_data.columns if c != 'selling_price']
X = cars_data[feature_cols]
y = cars_data['selling_price']

print("\nFinal dataset shape:", X.shape)
print("Feature columns:\n", X.columns.tolist())

# ---------- Train/test split ----------
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain shape: {x_train.shape}, Test shape: {x_test.shape}")

# ---------- Train model ----------
model = LinearRegression()
model.fit(x_train, y_train)
print("✅ Model trained.")

# ---------- Example prediction ----------
example = pd.DataFrame(
    [[5, 2022, 12000, 1, 1, 1, 1, 12.99, 2494.0, 100.6, 5.0]],
    columns=feature_cols[:11] if len(feature_cols) >= 11 else feature_cols
)
# Make sure example has same columns -- if not, create zeros for missing columns
for c in feature_cols:
    if c not in example.columns:
        example[c] = 0.0
example = example[X.columns]  # reorder
try:
    print("Example prediction:", model.predict(example)[0])
except Exception as e:
    print("Warning: example prediction failed:", repr(e))

# ---------- Save the model ----------
pk.dump(model, open('model.pkl', 'wb'))
print("✅ Model saved to model.pkl")
