"""
    This module creates the plots that visualize the results.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def show_and_save_plt(ax, file_name, y_label=None, ylim =None, label_size = 18, tick_size = 14, dir = "figures"):
    """
    Shows and saves the given plot and defines the appearance of the final plot.
    :param ax: the plot to be saved.
    :param file_name: save file name where the file is saved.
    :param y_label: the y axis label displayed
    :param ylim: limits of the y axis.
    :param label_size: font size of the label text
    :param tick_size: font size of the tick numbers
    :param dir: the directory where all the plots should be saved
    """
    # this only works the second time the function is used, since it sets the style for future plots.
    # It was still more convenient this way. #TODO fix this
    sns.set_style("whitegrid")
    sns.set(palette='colorblind')

    if y_label != None:
        plt.ylabel(y_label)
    plt.xlabel(None)
    if ylim != None:
        ax.set(ylim=ylim)

    try:
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.label.set_size(label_size)
    except:
        try:
            plt.ylabel(y_label, fontsize=label_size)
            plt.xlabel(fontsize=label_size)
        except Exception as e:
            print(e)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    file_name = os.path.join(dir, file_name + ".jpeg")
    if not (os.path.isdir(dir)):
        os.makedirs(dir)
    plt.tight_layout()
    plt.savefig(file_name)

    plt.show()


def analyze_distribution(data, columns):
    """
    Analyzes the distribution of the values in the given columns
    :param data: The dataframe to be analyzed
    :param columns: The name of the columns to be compared
    :return ax: a plot showing the distribution
    """
    df = data[columns]
    df['condition'] = data['condition']
    df = pd.melt(df, id_vars='condition')
    ax = sns.catplot(x='condition', y='value', hue='variable', data=df, kind='bar')

    return ax


if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set(palette='colorblind')
    order = ['LRP', 'LIME', "Counterfactual"]
    ci = 95 # confidence intervall
    directory = "data"

    ### plotting the content analysis
    data_name = os.path.join(directory, 'content_analysis_results.csv')
    data = pd.read_csv(data_name, sep=';')
    data["condition"] = data["condition"].apply(lambda x: "LRP" if x == 0 else "LIME" if x == 1 else "Counterfactual")

    column_name = "points_pneumonia"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Mean Score', ylim=(0, 1))

    column_name = "points_health"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Mean Score', ylim=(0, 1))

    column_name = "points_total"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Mean Score', ylim=(0, 2))

    ### plotting the other variables
    data_name = os.path.join(directory, 'important_variables.csv')
    data = pd.read_csv(data_name, sep=';')
    data["condition"] = data["condition"].apply(lambda x: "LRP" if x == 0 else "LIME" if x == 1 else "Counterfactual")

    # main Variables
    column_name = "correct_Ai_Predictions_percent"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Correct Predictions Percent', ylim=(0, 1))

    column_name = "explSatisfaction_mean"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Explanation Satisfaction', ylim=(1, 5))

    column_name = "Trust_mean"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Trust', ylim=(1, 5))

    # Emotion (1-7) and Competence (1-10)
    column_name = "CompetenceRating[1]"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Competence Rating', ylim=(1, 10))

    column_name = "Anger_mean"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Anger', ylim=(1, 7))

    column_name = "Happy_mean"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Happiness', ylim=(1, 7))

    column_name = "Relaxation_mean"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Relaxation', ylim=(1, 7))

    # Confidence
    column_name = "meanConfidence"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Mean Confidence', ylim=(1, 7))

    column_name = "correctMeanConfidence"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Mean Confidence', ylim=(1, 7))

    column_name = "falseMeanConfidence"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Mean Confidence', ylim=(1, 7))

    # other
    folder_name = "figures\\additonal_figures"

    column_name = "ownPred=AiPred"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Same Predictions Percent', ylim=(0, 1), dir=folder_name)

    column_name = "correct_Predictions_ForWrongAi_percent"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Correct Predicitons(Wrong AI) Percent', ylim=(0, 1), dir=folder_name)

    column_name = "correct_Predictions_ForCorrectAi_percent"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Correct Predicitons(Correct AI) Percent', ylim=(0, 1), dir=folder_name)

    column_name = "correct_own_Predictions_percent"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Correct Own Diagnosis Percent', ylim=(0, 1), dir=folder_name)

    column_name = "TimePrediction_mean"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='Prediction Time', dir=folder_name)

    column_name = "age"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=order, ci=ci)
    show_and_save_plt(ax, column_name, y_label='age', dir=folder_name)

    column_name = "gender"
    ax = sns.boxplot(x='condition', y=column_name, data=data, order=order)
    show_and_save_plt(ax, column_name, y_label='gender', dir=folder_name)

    health_exp = ["experienceHealth_none", "experienceHealth_interested", "experienceHealth_work",
                  "experienceHealth_doctor", "experienceHealth_pulmonologist", "experienceHealth_research",
                  "experienceHealth_xRays", "experienceHealth_other"]

    for column in health_exp:
        data[column] = data[column].apply(lambda x: 0 if pd.isna(x) else 1)

    ax = analyze_distribution(data, health_exp)
    show_and_save_plt(ax, "health_exp", dir=folder_name)

    ai_exp = ["experienceAI_none", "experienceAI_media", "experienceAI__privateLife",
              "experienceAI_work", "experienceAI_relatedCourse", "experienceAI_research",
              "experienceAI_other"]

    for column in ai_exp:
        data[column] = data[column].apply(lambda x: 0 if pd.isna(x) else 1)

    ax = analyze_distribution(data, ai_exp)
    show_and_save_plt(ax, "ai_exp", dir=folder_name)


