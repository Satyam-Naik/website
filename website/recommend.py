import numpy as np
import pulp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def nutritional_values(kg,calories):
    protein_calories = kg*4
    carb_calories = calories/2.
    fat_calories = calories-carb_calories-protein_calories
    res = {'Protein Grams':protein_calories/4,'Carbohydrates Grams':carb_calories/4,'Fat Grams':fat_calories/9}
    return res

def linear_optimisation(wt,cal,perc,data):
    data = data.sample(frac=1).sample(frac=1).reset_index().drop('index',axis=1)
    opt = pulp.LpProblem("Diet",pulp.LpMinimize)
    G = nutritional_values(wt,cal)
    E = G['Carbohydrates Grams']
    F = G['Fat Grams']
    P = G['Protein Grams']
    data = data[data.calories!=0]
    food = data.name.tolist()
    c = data.calories.tolist()
    x = pulp.LpVariable.dicts( "x", indices = food, lowBound=0, upBound=1.5, cat='Continuous', indexStart=[] )
    e = data.carbohydrate.tolist()
    f = data.total_fat.tolist()
    p = data.protein.tolist()
    opt += pulp.lpSum( [x[food[i]]*c[i]*np.random.uniform(0.9, 1.1) for i in range(len(food))]  )
    opt += pulp.lpSum( [x[food[i]]*e[i] for i in range(len(x)) ] )>=E * perc
    opt += pulp.lpSum( [x[food[i]]*f[i] for i in range(len(x)) ] )>=F * perc
    opt += pulp.lpSum( [x[food[i]]*p[i] for i in range(len(x)) ] )>=P * perc
    opt.solve(pulp.PULP_CBC_CMD(msg=0))

    food_list = []
    for v in opt.variables():
        value = v.varValue
        if value > 0:
            value = np.round(value, 2)
            food_name = v.name[2:].replace('_', ' ')
            food_list.append({'Food': food_name, 'Quantity': value * 100})
    
    # print(food_list)
    return food_list

def knn_model(wt,cal,perc,data):
    G = nutritional_values(wt,cal)
    E = G['Carbohydrates Grams']
    F = G['Fat Grams']
    P = G['Protein Grams']

    df = data.copy()
    data = data.reset_index(drop=True)
    scaler = StandardScaler()
    df[['calories', 'carbohydrate', 'total_fat', 'protein']] = scaler.fit_transform(df[['calories', 'carbohydrate', 'total_fat', 'protein']])

    user_requirements = {
        "calories": cal * perc,
        "carbohydrate": E * perc,
        "total_fat": F * perc,
        "protein": P * perc
    }

    # Normalize user requirements
    user_requirements_normalized = scaler.transform([[user_requirements['calories'], user_requirements['carbohydrate'], user_requirements['total_fat'], user_requirements['protein']]])

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df[['calories', 'carbohydrate', 'total_fat', 'protein']])

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(user_requirements_normalized)

    # Randomly sample from the neighbors
    random_indices = np.random.choice(indices[0], size=5, replace=False)  # Change 'size' to the desired number of recommendations

    food_list = []
    for index in random_indices:
        if index < len(data):
            try:
                food_item = {
                    'name': data.loc[index, 'name'],
                    'calories': data.loc[index, 'calories'],
                    'carbohydrate': data.loc[index, 'carbohydrate'],
                    'total_fat': data.loc[index, 'total_fat'],
                    'protein': data.loc[index, 'protein']
                }
                food_list.append(food_item)
            except KeyError:
                print(f"KeyError occurred at index {index}")
    return food_list



