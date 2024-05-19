import numpy as np
import pandas as pd
from fastapi import FastAPI
from pyDecision.algorithm import promethee_ii
from pydantic import BaseModel
from typing import List

# Load the data
data = pd.read_excel("skincaredata.xlsx")
data["ProductID"] = range(1, len(data) + 1)

# Initialize FastAPI app
app = FastAPI()


# Function to modify price values based on user input
def adjust_price_values(df, price_preference):
    if price_preference == "Harga bukanlah faktor utama":
        df["Harga"] = df["Harga"].apply(lambda x: 6 - x)
    return df


# Function to modify skin type values based on user input
def adjust_skin_type_values(df, skin_type_preference):
    if skin_type_preference == "Kulit Sensitif":
        df["Jenis Kulit"] = df["Jenis Kulit"].apply(
            lambda x: 7 - x if x == 2 or x == 5 else x
        )
    elif skin_type_preference == "Kulit Berminyak":
        df["Jenis Kulit"] = df["Jenis Kulit"].apply(
            lambda x: 9 - x if x == 4 or x == 5 else x
        )
    elif skin_type_preference == "Kulit Kering":
        df["Jenis Kulit"] = df["Jenis Kulit"].apply(
            lambda x: 8 - x if x == 3 or x == 5 else x
        )
    return df


# Function to perform PROMETHEE II analysis and return top 3 products
def get_top_3_products(price_preference: str, skin_type_preference: str):
    df = data.copy()
    df = adjust_price_values(df, price_preference)
    df = adjust_skin_type_values(df, skin_type_preference)

    decision_matrix = df.iloc[:, 5:].values
    criteria_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Q = [0.1, 0.1, 0.1, 0.1, 0.1]
    S = [0.1, 0.1, 0.1, 0.1, 0.1]
    P = [0.2, 0.2, 0.2, 0.2, 0.2]
    F = ["t2", "t2", "t2", "t2", "t2"]

    result = promethee_ii(decision_matrix, criteria_weights, Q, S, P, F)
    ranking = result[:, 1]
    ranked_indices = result[:, 0].astype(int) - 1
    sorted_indices = np.argsort(ranking)[::-1]

    top_3 = []
    for rank, sorted_index in enumerate(sorted_indices[:3], start=1):
        original_index = ranked_indices[sorted_index]
        product = {
            "ProductID": int(df["ProductID"].iloc[original_index]),
            "ProductName": df["Merk"].iloc[original_index],
            "Image": df["Gambar"].iloc[original_index],
            "RealPrice": int(df["Harga Asli"].iloc[original_index]),
            "RealSkinType": df["Jenis Kulit Asli"].iloc[original_index],
            "Rating": int(df["Rating"].iloc[original_index]),
        }
        top_3.append(product)

    return top_3


# Define request model
class PreferenceRequest(BaseModel):
    price_preference: str
    skin_type_preference: str


# Define response model
class ProductResponse(BaseModel):
    ProductID: int
    ProductName: str
    Image: str
    RealPrice: int
    RealSkinType: str
    Rating: int


@app.post("/recommendations", response_model=List[ProductResponse])
def recommend_products(preferences: PreferenceRequest):
    top_3_products = get_top_3_products(
        preferences.price_preference, preferences.skin_type_preference
    )
    return top_3_products


@app.get("/products", response_model=List[ProductResponse])
def get_all_products():
    all_products = []
    for index, row in data.iterrows():
        product = {
            "ProductID": int(row["ProductID"]),
            "ProductName": row["Merk"],
            "Image": row["Gambar"],
            "RealPrice": int(row["Harga Asli"]),
            "RealSkinType": row["Jenis Kulit Asli"],
            "Rating": int(row["Rating"]),
        }
        all_products.append(product)
    return all_products


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
