import pandas as pd
file_path = r"C:\Users\sivar\OneDrive\Desktop\ML\lab2\Lab Session Data(thyroid0387_UCI).csv"
df = pd.read_csv(file_path)

# ─── 1. Select the first 2 rows ───
row1 = df.iloc[0]
row2 = df.iloc[1]

# ─── 2. Identify binary columns (values only 0 or 1) ───
binary_cols = []
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        binary_cols.append(col)

print(f"Binary Columns Used: {binary_cols}")

# ─── 3. Extract binary attribute values from first 2 rows ───
vec1 = row1[binary_cols].astype(int)
vec2 = row2[binary_cols].astype(int)

# ─── 4. Compute f11, f00, f10, f01 ───
f11 = sum((vec1 == 1) & (vec2 == 1))
f00 = sum((vec1 == 0) & (vec2 == 0))
f10 = sum((vec1 == 1) & (vec2 == 0))
f01 = sum((vec1 == 0) & (vec2 == 1))

# ─── 5. Calculate JC and SMC ───
jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

# ─── 6. Print Results ───
print("\n🔍 Similarity Comparison between First Two Binary Vectors:")
print(f"f11 = {f11}, f00 = {f00}, f10 = {f10}, f01 = {f01}")
print(f"Jaccard Coefficient (JC) = {jc:.4f}")
print(f"Simple Matching Coefficient (SMC) = {smc:.4f}")

# ─── 7. Observation ───
print("\n📌 Observation:")
if jc > smc:
    print("JC > SMC → JC emphasizes on positive matches (1s), better for sparse binary data.")
elif smc > jc:
    print("SMC > JC → SMC considers both 0 and 1 matches, useful when 0s are also meaningful.")
else:
    print("JC and SMC are equal — rare, implies balanced match across 1s and 0s.")
