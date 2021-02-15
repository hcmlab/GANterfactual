"""
    This module is used to calculate the participants scores for their textual answers, based on the scores assigned by
    three different coders.
"""

import pandas as pd
import os
import numpy as np


def read_categories(labels):
    labels = str(labels)
    labels = labels.split('#')
    labels = [int(i) for i in labels]
    return labels


def get_point_dict(df_annotator):
    pneumonia_points = {}
    health_points = {}
    categories = df_annotator["Legend"].array
    for index in range(len(categories)):
        pneumonia_points[categories[index]] = df_annotator["PunktePneumonia"].array[index]
        health_points[categories[index]] = df_annotator["PunkteHealth"].array[index]
    return pneumonia_points, health_points


if __name__ == '__main__':
    directory = "data"

    df_annotator1 = pd.read_excel(os.path.join(directory, "categories_rater1.xlsx"))
    df_annotator2 = pd.read_excel(os.path.join(directory, "categories_rater2.xlsx"))
    df_annotator3 = pd.read_excel(os.path.join(directory, "categories_rater3.xlsx"))

    pneumonia_answers1 = df_annotator1["categoriesPneumonia"]
    pneumonia_answers2 = df_annotator2["categoriesPneumonia"]

    health_answers1 = df_annotator1["categoriesHealthy"]
    health_answers2 = df_annotator2["categoriesHealthy"]

    # build the point dictionaries:
    pneumonia_points1, health_points1 = get_point_dict(df_annotator1)
    pneumonia_points2, health_points2 = get_point_dict(df_annotator2)

    # the third annotator only assigned points to the specific answers, where the other two annotators disagreed,
    # so no further calculation is necessary here
    pneumonia_points3 = df_annotator3["PunktePneumonia"]
    health_points3 = df_annotator3["PunkteHealth"]

    final_points_pneumonia = []
    final_points_health = []

    # look for the indices where the first two annotators disagree
    # here we determine the score by median between all 3 annotators
    # for "pneumonia"
    problematic_indices = []
    for index in range(len(pneumonia_answers1)):
        labels1 = read_categories(pneumonia_answers1[index])
        points1 = [pneumonia_points1.get(item, item) for item in labels1]
        points1 = max(points1)

        labels2 = read_categories(pneumonia_answers2[index])
        points2 = [pneumonia_points2.get(item, item) for item in labels2]
        points2 = max(points2)
        if points1 != points2:
            points3 = float(pneumonia_points3[index])
            points = [points1, points2, points3]
            final_points = np.median(points)
            problematic_indices.append(index + 2)
            print("we have a problem here")
        else:
            final_points = points1
        final_points_pneumonia.append(final_points)

    print("Pneumonia")
    print(problematic_indices)
    print("number:")
    print(len(problematic_indices))

    # for "healthy"
    problematic_indices = []
    for index in range(len(health_answers1)):
        labels1 = read_categories(health_answers1[index])
        points1 = [health_points1.get(item, item) for item in labels1]
        points1 = max(points1)

        labels2 = read_categories(health_answers2[index])
        points2 = [health_points2.get(item, item) for item in labels2]
        points2 = max(points2)
        if points1 != points2:
            points3 = float(health_points3[index])
            points = [points1, points2, points3]
            final_points = np.median(points)
            problematic_indices.append(index + 2)
            print("we have a problem here")
        else:
            final_points = points1
        final_points_health.append(final_points)

    print("health")
    print(problematic_indices)
    print("number:")
    print(len(problematic_indices))

    # add the points to the important variables dataframe
    seeds = df_annotator2["seed"]

    important_values_name = os.path.join(directory, "important_variables.csv")
    important_values = pd.read_csv(important_values_name, sep=';')

    conditions = []
    for seed in seeds:
        if seed > 0:
            index = np.where(important_values["seed"] == int(seed))[0][0]
            condition = important_values["condition"][index]
        else:
            condition = -1
        conditions.append(condition)

    total_points = np.asarray(final_points_pneumonia) + np.asarray(final_points_health)

    result_df = pd.DataFrame()
    result_df["seed"] = seeds
    result_df["condition"] = conditions
    result_df["points_pneumonia"] = final_points_pneumonia
    result_df["points_health"] = final_points_health
    result_df["points_total"] = total_points

    result_name = os.path.join(directory, 'content_analysis_results.csv')
    result_df.to_csv(result_name, sep=';')
