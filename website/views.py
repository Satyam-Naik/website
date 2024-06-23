from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from .recommend import linear_optimisation, knn_model
from django.conf import settings

def get_details(request):
    if "user_data" in request.session:
        return redirect('website:home')
    if request.method == "POST":
        name = request.POST.get('name').capitalize()
        age = int(request.POST.get('age'))
        weight = int(request.POST.get('weight'))
        height = int(request.POST.get('height'))
        pref = request.POST.get('preference')
        gender = request.POST.get('gender')
        activity = request.POST.get('activity')
        diseases = request.POST.getlist('disease')
        if gender == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        if activity == "sedentary":
            cal = bmr * 1.2
        elif activity == "lightly_active":
            cal = bmr * 1.375
        elif activity == "moderately_active":
            cal = bmr * 1.55
        elif activity == "very_active":
            cal = bmr * 1.725
        else:
            cal = bmr * 1.9

        request.session['user_data'] = {
                'name': name,
                'age': age,
                'weight': weight,
                'height': height,
                'preference': pref,
                'gender': gender,
                'activity': activity,
                'diseases': diseases,
                'calories': np.round(cal, 2),
            }
        return redirect('website:home')
    
    return render(request, 'website/details.html')

def home(request):
    if "user_data" not in request.session:
        return render(request, 'website/details.html')
    
    return render(request, 'website/home.html', {
        "user_data": request.session["user_data"],
    })

def optimisation(request):
    if "user_data" not in request.session:
        return redirect('website:home') 
    dataset = pd.read_csv("final_data.csv")
    user = request.session.get("user_data")
    wt       = user['weight']
    cal      = user['calories']
    diseases = user['diseases']
    pref     = user['preference']
    pref = 0 if pref == "Vegetarian" else 1

    filtered_df = dataset[
        (dataset['veg/nonveg'] == pref) &
        dataset[diseases].all(axis=1)
    ]

    meals = {'breakfast': 0.2, 'lunch': 0.35, 'snacks': 0.15, 'dinner': 0.3}
    meal_data = {}
    for meal, fraction in meals.items():
        meal_df = filtered_df[filtered_df[meal] == 1]
        meal_df = meal_df[['serial_no', 'name', 'calories', 'carbohydrate', 'total_fat', 'protein']]
        meal_data[meal] = linear_optimisation(wt, cal, fraction, meal_df)
    return render(request, 'website/display1.html', meal_data)

def knn(request):
    if "user_data" not in request.session:
        return redirect('website:home')
    dataset = pd.read_csv("final_data.csv")
    user = request.session.get("user_data")
    wt       = user['weight']
    cal      = user['calories']
    diseases = user['diseases']
    pref     = user['preference']
    pref = 0 if pref == "Vegetarian" else 1

    filtered_df = dataset[
        (dataset['veg/nonveg'] == pref) &
        dataset[diseases].all(axis=1)
    ]

    meals = {'breakfast': 0.2, 'lunch': 0.35, 'snacks': 0.15, 'dinner': 0.3}
    meal_data = {}
    for meal, fraction in meals.items():
        meal_df = filtered_df[filtered_df[meal] == 1]
        meal_df = meal_df[['serial_no', 'name', 'calories', 'carbohydrate', 'total_fat', 'protein']]
        meal_data[meal] = knn_model(wt, cal, fraction, meal_df)
    return render(request, 'website/display2.html', meal_data)


def clear_session(request):
    if "user_data" in request.session:
        del request.session['user_data']
    return redirect('website:home')