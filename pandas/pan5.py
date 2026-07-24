import pandas as pd

data = {
    "Marks": [80, 90, 75, 88, 95]
}

df = pd.DataFrame(data)

print(df.describe())