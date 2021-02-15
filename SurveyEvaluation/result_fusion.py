"""
    Converts the original raw result csv files to one more readable and csv with less unnecessary information.
"""

import pandas as pd
import os

# Columns created by Lime-survey that don't contain answers (e.g. text display questions)
delete_colums = ["id",'startlanguage',
                 "description0", "description1", "description2", "ownPredText",
                 "image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8", "image9", "image10",
                  "image11", "image12", "flag1", "flag2", "flag3", "flag4", "flag5", "flag6", "flag7", "flag8", "flag9",
                 "flag10",  "flag11",  "flag12", "lastPrediction", "next", "iexplorermobile",
                 "healthInfo", "lrpInfo", "lrpInfo2", "counterfactualInfo", "counterfactualInfo2", "limeInfo",
                 "limeInfo2",
                 "healthQuiz", "healthQuiz2", "KIrightQuiz", "lrpQuiz", "limeQuiz", "ganQuiz1", "ganQuiz2",
                 "lastAIPrediction1", "lastAIPrediction2", "lastAIPrediction3", "lastAIPrediction4", "lastAIPrediction5",
                 "lastAIPrediction6", "lastAIPrediction7", "lastAIPrediction8", "lastAIPrediction9", "lastAIPrediction10",
                 "lastAIPrediction11", "lastAIPrediction12",
                  "description10", "description20", "description30", "description40", "description50", "description60",
                 "description70", "description80", "description90", "description100", "description110",
                 "description120",
                 "description11", "description21", "description31", "description41", "description51", "description61",
                 "description71", "description81", "description91", "description101", "description111",
                 "description121",
                 "description12", "description22", "description32", "description42", "description52", "description62",
                 "description72", "description82", "description92", "description102", "description112",
                 "description122",
                 "pictures1", "pictures2", "pictures3", "pictures4", "pictures5", "pictures6", "pictures7", "pictures8",
                 "pictures9", "pictures10", "pictures11", "pictures12",
                 "explanationReminder1", "explanationReminder2", "IntroEmoCompTrust",
                 "IntroPracticalAppl"
                 ]
# the time values that should be kept. The time taken during the quiz is ignored for example.
keep_time_colums=['totalTime', 'predictionTime1', 'predictionTime2', 'predictionTime3', 'predictionTime4',
                  'predictionTime5', 'predictionTime6', 'predictionTime7', 'predictionTime8', 'predictionTime9',
                  'predictionTime10', 'predictionTime11', 'predictionTime12', 'lastPageTime']


def rename_time(data_frame, successive_name, new_name):
    index = data_frame.columns.get_loc(successive_name)
    index -= 1
    old_column_name = data_frame.columns[index]
    data_frame = data_frame.rename(columns={old_column_name: new_name}, errors="raise")
    return data_frame


def rename_times(data_frame):
    data_frame = data_frame.rename(columns={'interviewtime': 'totalTime'}, errors="raise")

    data_frame = rename_time(data_frame, 'lastAIPrediction1Time', 'predictionTime1')
    data_frame = rename_time(data_frame, 'lastAIPrediction2Time', 'predictionTime2')
    data_frame = rename_time(data_frame, 'lastAIPrediction3Time', 'predictionTime3')
    data_frame = rename_time(data_frame, 'lastAIPrediction4Time', 'predictionTime4')
    data_frame = rename_time(data_frame, 'lastAIPrediction5Time', 'predictionTime5')
    data_frame = rename_time(data_frame, 'lastAIPrediction6Time', 'predictionTime6')
    data_frame = rename_time(data_frame, 'lastAIPrediction7Time', 'predictionTime7')
    data_frame = rename_time(data_frame, 'lastAIPrediction8Time', 'predictionTime8')
    data_frame = rename_time(data_frame, 'lastAIPrediction9Time', 'predictionTime9')
    data_frame = rename_time(data_frame, 'lastAIPrediction10Time', 'predictionTime10')
    data_frame = rename_time(data_frame, 'lastAIPrediction11Time', 'predictionTime11')
    data_frame = rename_time(data_frame, 'lastAIPrediction12Time', 'predictionTime12')

    data_frame = rename_time(data_frame, 'featurePneumoniaTime', 'lastPageTime')
    return data_frame


def delete_times(data_frame, _keep_time_columns=keep_time_colums):
    """
    Deletes all time columns form the given dataframe that are not in *keep_time_columns*
    :param data: the dataframe where the time columns should be deleted
    :param keep_time_columns: the time columns to be kept
    :return: the dataframe without unnecessary time columns
    """
    start_index = data_frame.columns.get_loc('totalTime') + 1
    to_delete = []
    for i in range(start_index,len(data_frame.columns)):
        column_name = data_frame.columns[i]
        if column_name not in _keep_time_columns:
            to_delete.append(column_name)
    data_frame = data_frame.drop(to_delete, axis=1)
    return data_frame


def eval_selection_question(data_frame, column_name, correct_answer):
    """
    simple first evaluation of the trust task that checks wheter the participants got the correct agent
    :param data_frame: the dataframe containing the survey results
    :param agent_number: the number of the comparison (1,2 or 3)
    :return: the same data_frame but with an added binary column 'trust' + *agent_number* + 'correct' storing whether
    the participants where correct.
    """
    # the correct answer for each comparison
    resulting_column_name = column_name + '_correct'
    correct_answer_arr = []
    answer_arr = data_frame[column_name]
    for entry in answer_arr:
        if entry == correct_answer:
            correct = 1;
        else:
            correct = 0;
        correct_answer_arr.append(correct)

    data_frame[resulting_column_name] = correct_answer_arr

    return data_frame


def getColumn(name):
    """
    helper function to get the array of the column with the given name
    :param name: name of the column
    :return: the array of this column
    """
    return resulting_frame[name].array


def list_mean(my_list):
    if len(my_list) != 0:
        mean = sum(my_list) / len(my_list)
        float(mean)
    else:
        mean = "NaN"
    return mean


def mean_of_dict_keys(dict):
    """
    calculates the mean values of all the columns given in the keys of the dict
    :param dict: a dictionary whose keys correspond to columns in the dataframe
    :return: an array containing the mean values of the given columns
    """
    summe = None
    columns = 0;
    for column_name in dict.keys():
        columns += 1;
        current_array = resulting_frame[column_name].array
        if summe is None:
            summe = current_array
        else:
            summe = summe + current_array
    return summe / columns


if __name__ == '__main__':
    file_names = ["ClickWorkerResults_Fragencode_AntwortCode.csv", 'ErsterClickPilot_Fragencode_AntwortCode.csv']
    # file_names = ["ClickWorkerResults_Fragencode_AntwortCode.csv"]
    directory = "data"
    # file_names = ["90Batch.csv"]

    # which image shows a Lung with Pneumonia ( "sick" or "healt"(lime survey only allows up to 5 letters))
    correct_diagnosis_dict = {1: "sick", 2: "sick", 3: "sick", 4: "healt", 5: "healt", 6: "healt", 7: "healt",
                              8: "healt", 9: "healt", 10: "sick", 11: "sick", 12: "sick"}
    # what the AI predicted for each image
    ai_predictions_dict = {1: "sick", 2: "sick", 3: "sick", 4: "sick", 5: "sick", 6: "sick", 7: "healt",
                              8: "healt", 9: "healt", 10: "healt", 11: "healt", 12: "healt"}


    # load the data frame containing the raw results of the survey
    files = []
    for file_name in file_names:
        file_name = os.path.join(directory,file_name)
        data_frame = pd.read_csv(file_name, sep=';')
        data_frame = rename_times(data_frame)
        files.append(data_frame)

    # raw fusion of the results
    resulting_frame = pd.concat(files, axis=0, ignore_index=True, sort=False)
    resulting_frame.to_csv(os.path.join(directory,"raw_fusion.csv"))

    # remove participants that did not finish the survey
    resulting_frame=resulting_frame[resulting_frame.lastpage == 18]

    resulting_frame = resulting_frame.drop(delete_colums, axis=1)
    resulting_frame = delete_times(resulting_frame,keep_time_colums)
    resulting_frame.to_csv(os.path.join(directory,'fusion_cleaned.csv'))

    # rename Ai Experience
    ai_experience_dict = {"experienceAI2[6]": "experienceAI_none", "experienceAI2[1]": "experienceAI_media",
                          "experienceAI2[2]": "experienceAI__privateLife", "experienceAI2[3]": "experienceAI_work",
                          "experienceAI2[4]": "experienceAI_relatedCourse", "experienceAI2[5]": "experienceAI_research",
                          "experienceAI2[other]": "experienceAI_other"
                          }
    resulting_frame = resulting_frame.rename(columns=ai_experience_dict, errors="raise")

    # rename Healthcare Experience
    health_experience_dict = {"experienceHealth2[1]": "experienceHealth_none", "experienceHealth2[2]": "experienceHealth_interested",
                          "experienceHealth2[3]": "experienceHealth_work", "experienceHealth2[4]": "experienceHealth_doctor",
                          "experienceHealth2[5]": "experienceHealth_pulmonologist", "experienceHealth2[6]": "experienceHealth_research",
                          "experienceHealth2[7]": "experienceHealth_xRays", "experienceHealth2[other]": "experienceHealth_other"
                          }
    resulting_frame = resulting_frame.rename(columns=health_experience_dict, errors="raise")

    ####### ADDITIONAL CALCULATIONS #####
    # evaluates where the participant answered correctly to the question "Do you think the lung is infected with pneumonia"
    for i in correct_diagnosis_dict.keys():
        column_name = "ownPrediction" + str(i)
        eval_selection_question(resulting_frame, column_name, correct_diagnosis_dict[i])

    # evaluates where the participant correctly predicted the AI
    for i in ai_predictions_dict.keys():
        column_name = "aiPrediction" + str(i)
        eval_selection_question(resulting_frame, column_name, ai_predictions_dict[i])

    # sum correct predictions
    summe = None
    for i in correct_diagnosis_dict.keys():
        column_name = "ownPrediction" + str(i) + "_correct"
        current_array = resulting_frame[column_name].array
        if summe is None:
            summe = current_array
        else:
            summe = summe + current_array
    resulting_frame["correctOwnPredictionsTotal"] = summe

    summe = None
    for i in correct_diagnosis_dict.keys():
        column_name = "aiPrediction" + str(i) + "_correct"
        current_array = resulting_frame[column_name].array
        if summe is None:
            summe = current_array
        else:
            summe = summe + current_array
    resulting_frame["correctAiPredictionsTotal"] = summe

    # calculate mean values for predictions where the participant used the slider
    ignored_slider_array = []

    percent_correct_AI_predictions = []
    percent_correct_own_predictions = []
    percent_ownPred_is_AiPred = []
    percent_correct_PredictionsForWrongAi = []
    percent_correct_PredictionsForCorrectAi = []

    mean_confidence = []
    correct_mean_confidence = []
    false_mean_confidence = []

    mean_time = []

    # get the relevant columns
    confidence_arrays = []
    ownPred_arrays = []
    aiPred_arrays = []
    slider_arrays = []
    time_arrays = []
    for i in correct_diagnosis_dict.keys():
        confidence_arrays.append(getColumn("confidence" + str(i) + "[howConfident]"))
        ownPred_arrays.append(getColumn("ownPrediction" + str(i)))
        aiPred_arrays.append(getColumn("aiPrediction" + str(i)))
        slider_arrays.append(getColumn("slider" + str(i)))
        time_arrays.append(getColumn("predictionTime" + str(i)))

    for row_index in range(confidence_arrays[0].size):
        ai_is_own = []
        correct_Ai_predictions =[]
        correct_own_predictions =[]
        correct_PredictionsForWrongAi = []
        correct_PredictionsForCorrectAi = []

        confidences = []
        correct_confidences = []
        false_confidences = []

        times = []

        ignored_sliders = 0
        for column_index in range(len(confidence_arrays)):
            # check if slider was moved
            if slider_arrays[column_index][row_index] != -1:

                own_pred = ownPred_arrays[column_index][row_index]
                ai_pred = aiPred_arrays[column_index][row_index]
                conf = confidence_arrays[column_index][row_index]

                # ai = own
                if own_pred == ai_pred:
                    ai_is_own.append(1)
                else:
                    ai_is_own.append(0)
                # own Prediction correct?
                if own_pred == correct_diagnosis_dict[column_index+1]:
                    correct_own_predictions.append(1)
                else:
                    correct_own_predictions.append(0)

                # ai Prediction correct?
                if ai_pred == ai_predictions_dict[column_index+1]:
                    correct_Ai_predictions.append(1)
                    # correct confidence
                    correct_confidences.append(conf)
                    # if ai is incorrect also log this speratly
                    if (column_index+1) in [4, 5, 6, 10, 11, 12]:
                        correct_PredictionsForWrongAi.append(1)
                    else:
                        correct_PredictionsForCorrectAi.append(1)
                else:
                    correct_Ai_predictions.append(0)
                    # false confidence
                    false_confidences.append(conf)
                    # if ai is incorrect also log this
                    if (column_index+1) in [4, 5, 6, 10, 11, 12]:
                        correct_PredictionsForWrongAi.append(0)
                    else:
                        correct_PredictionsForCorrectAi.append(0)

                #general confidence
                confidences.append(conf)
                #time
                times.append(time_arrays[column_index][row_index])

            else:
                # ignore answer if slider was not used
                ignored_sliders += 1

        # add the values for this user
        ignored_slider_array.append(ignored_sliders)

        percent_correct_own_predictions.append(list_mean(correct_own_predictions))
        percent_correct_AI_predictions.append(list_mean(correct_Ai_predictions))
        percent_ownPred_is_AiPred.append(list_mean(ai_is_own))
        percent_correct_PredictionsForWrongAi.append(list_mean(correct_PredictionsForWrongAi))
        percent_correct_PredictionsForCorrectAi.append(list_mean(correct_PredictionsForCorrectAi))

        mean_confidence.append(list_mean(confidences))
        correct_mean_confidence.append(list_mean(correct_confidences))
        false_mean_confidence.append(list_mean(false_confidences))

        mean_time.append(list_mean(times))
    # write results to csv
    resulting_frame["ignored_sliders"] = ignored_slider_array
    resulting_frame["ownPred=AiPred"] = percent_ownPred_is_AiPred
    resulting_frame["correct_own_Predictions_percent"] = percent_correct_own_predictions
    resulting_frame["correct_Ai_Predictions_percent"] = percent_correct_AI_predictions
    resulting_frame["correct_Predictions_ForWrongAi_percent"] = percent_correct_PredictionsForWrongAi
    resulting_frame["correct_Predictions_ForCorrectAi_percent"] = percent_correct_PredictionsForCorrectAi
    resulting_frame["meanConfidence"] = mean_confidence
    resulting_frame["correctMeanConfidence"] = correct_mean_confidence
    resulting_frame["falseMeanConfidence"] = false_mean_confidence
    resulting_frame["TimePrediction_mean"] = mean_time

    # rename explanation satisfaction
    ES_dict = {"explSatisfaction[1]": "explSatisfaction_makeDecision", "explSatisfaction[2]": "explSatisfaction_satisfying",
               "explSatisfaction[3]": "explSatisfaction_detail", "explSatisfaction[4]": "explSatisfaction_complete",
               "explSatisfaction[5]": "explSatisfaction_usefulToPredictAi", "explSatisfaction[6]": "explSatisfaction_trust"}
    resulting_frame["explSatisfaction_mean"] = mean_of_dict_keys(ES_dict)
    resulting_frame = resulting_frame.rename(columns=ES_dict, errors="raise")

    # rename trust
    trust_dict = {"TrustRating[1]": "TrustRating_trust", "TrustRating[2]": "TrustRating_rely"}
    resulting_frame["Trust_mean"] = mean_of_dict_keys(trust_dict)
    resulting_frame = resulting_frame.rename(columns=trust_dict, errors="raise")

    # rename emotion
    anger_dict = {"EmotionRating[1]": "EmotionRating_Anger", "EmotionRating[14]": "EmotionRating_Pissed",
                    "EmotionRating[8]": "EmotionRating_Mad"}
    happy_dict = {"EmotionRating[4]": "EmotionRating_Happy", "EmotionRating[9]": "EmotionRating_Satisfaction",
                  "EmotionRating[13]": "EmotionRating_Enjoyment", "EmotionRating[15]": "EmotionRating_Liking"}
    relaxation_dict={"EmotionRating[3]": "EmotionRating_Easygoing", "EmotionRating[6]": "EmotionRating_ChilledOut",
                  "EmotionRating[11]": "EmotionRating_Calm", "EmotionRating[12]": "EmotionRating_Relaxation"}
    resulting_frame["Anger_mean"] = mean_of_dict_keys(anger_dict)
    resulting_frame["Happy_mean"] = mean_of_dict_keys(happy_dict)
    resulting_frame["Relaxation_mean"] = mean_of_dict_keys(relaxation_dict)
    emotion_dict = {**anger_dict, **happy_dict,**relaxation_dict}
    resulting_frame = resulting_frame.rename(columns=emotion_dict, errors="raise")

    # practical
    practical_dict = {"PracticalApplAreas[1]": "PracticalApplAreas_additionToMedicalPersonel",
                      "PracticalApplAreas[2]": "PracticalApplAreas_diagnosis",
                      "PracticalApplAreas[3]": "PracticalApplAreas_supportForMedicalExplanation",
                      "PracticalApplAreas[4]": "PracticalApplAreas_routineExaminations",
                      "PracticalApplAreas[5]": "PracticalApplAreas_screeningAtHome",
                      "PracticalApplAreas[other]": "PracticalApplAreas_other"}
    resulting_frame = resulting_frame.rename(columns=practical_dict, errors="raise")

    # remove participants who ignored all sliders
    ignored_sliders = resulting_frame["ignored_sliders"].array
    for index in range(len(ignored_sliders)):
        if ignored_sliders[index] == 12:
            seed = resulting_frame["seed"].array[index]
            resulting_frame = resulting_frame.drop(resulting_frame.index[[index]])
            print("dropped seed " + str(seed) + " for 12 ignores")

    resulting_frame.to_csv(os.path.join(directory,'fusion_final.csv'), sep=';')

    interesting_columns = ["seed", "age", "gender", "condition"]

    # main variables
    interesting_columns.append("correct_Ai_Predictions_percent")
    interesting_columns.append("explSatisfaction_mean")
    interesting_columns.append("Trust_mean")

    #free text
    interesting_columns.extend(["featurePneumonia", "featureHealthy", "comment"])

    #prediction specific variables
    interesting_columns.extend(["ignored_sliders", "ownPred=AiPred", "correct_own_Predictions_percent",
                                "correct_Predictions_ForWrongAi_percent","correct_Predictions_ForCorrectAi_percent",
                                "meanConfidence", "correctMeanConfidence", "falseMeanConfidence", "TimePrediction_mean"])

    # time
    interesting_columns.append("totalTime")

    interesting_columns.extend(health_experience_dict.values())
    interesting_columns.extend(ai_experience_dict.values())
    interesting_columns.append("outcomeAI[1]")

    interesting_columns.append("CompetenceRating[1]")

    interesting_columns.extend(["Anger_mean", "Happy_mean", "Relaxation_mean"])

    interesting_columns.append("PracticalApplRating[1]")
    interesting_columns.extend(practical_dict.values())

    important_df = resulting_frame[interesting_columns]
    important_df.to_csv(os.path.join(directory, 'important_variables.csv'), sep=';')




