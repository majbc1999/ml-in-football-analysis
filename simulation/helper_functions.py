import numpy as np
import copy
from pomegranate import *


def return_profit(odds, probabilites, final):
    """
    Function, that places virtual bet, based on implied `probabilities`
    and `odds`. Then it calculated profit or loss, based on `final`
    result
    """

    a = 0

    if probabilites[0] != 0:
        bet_on_1 = odds[0] - (1 / probabilites[0])
    else:
        bet_on_1 = 0
        a += 1

    if probabilites[1] != 0:
        bet_on_2 = odds[1] - (1 / probabilites[1])
    else:
        bet_on_2 = 0
        a += 1

    if probabilites[2] != 0:
        bet_on_3 = odds[2] - (1 / probabilites[2])
    else:
        bet_on_3 = 0
        a += 1

    if a == 3:
        return 0


    if bet_on_1 == max([bet_on_1, bet_on_2, bet_on_3]):
        if bet_on_1 > 0:
            if final == "home win":
                return (odds[0] - 1)
            else:
                return -1
        else:
            return 0
    elif bet_on_2 == max([bet_on_1, bet_on_2, bet_on_3]):
        if bet_on_2 > 0:
            if final == "draw":
                return (odds[1] - 1)
            else:
                return -1
        else:
            return 0
    elif bet_on_3 == max([bet_on_1, bet_on_2, bet_on_3]):
        if bet_on_3 > 0:
            if final == "away win":
                return (odds[2] - 1)
            else:
                return -1
        else:
            return 0

    else:
        raise ValueError


def test_model(learn_data, test_data, model):
    """
    Function, that tests `test_data`, based on `model`, built
    on `learn_data`. Returns tuple of brier_score, precision, 
    recall, F1, confusion_matrix and profit
    """

    features = learn_data.drop(columns=["index", "result", "odds_home", "odds_draw", "odds_away"])

    X = np.array(features)
    y = np.array(learn_data["result"])

    model.fit(X, y)

    modified_brier_score = 0
    brier_score = 0
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    profit = 0

    test_features = test_data.drop(columns=["index", "result", "odds_home", "odds_draw", "odds_away"])
    test_X = np.array(test_features)
    test_y = np.array(test_data["result"])
    test_odds = np.array(test_data[["odds_home", "odds_draw", "odds_away"]])
    
    for id, line in enumerate(test_X):
        probabilites = list(reversed(list(model.predict_proba(line.reshape(1, -1))[0])))

        if max(probabilites) == probabilites[0]:
            c_matrix_pred = 0
        elif max(probabilites) == probabilites[1]:
            c_matrix_pred = 1
        elif max(probabilites) == probabilites[2]:
            c_matrix_pred = 2
        else:
            raise ValueError

        if test_y[id] == "home win":
            brier__ = [1, 0, 0]
            c_matrix_act = 0
        elif test_y[id] == "draw":
            brier__ = [0, 1, 0]
            c_matrix_act = 1
        elif test_y[id] == "away win":
            brier__ = [0, 0, 1]
            c_matrix_act = 2
        else:
            raise ValueError

        odds = test_odds[id]
        # calculate modified brier score
        for i in range(3):
            modified_brier_score += (probabilites[i] - (1 / odds[i])) ** 2


        # calculate brier score
        for i in range(3):
            brier_score += (probabilites[i] - brier__[i]) ** 2

        # confusion matrix
        confusion_matrix[c_matrix_act][c_matrix_pred] += 1

        final_result = test_y[id]
        
        # calculate profit
        profit += (return_profit(odds, probabilites, final_result) / len(test_data))
        

    # calculate rest of the metrics
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0])
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2])

    F1 = 2 * (precision * recall) / (precision + recall)

    return brier_score / len(test_data), precision, recall, F1, confusion_matrix, profit, modified_brier_score / len(test_data)


def discretize_dataframe(data):
    """
    Filters values, that are too big for Bayesian networks
    """
    new_data = copy.deepcopy(data)

    new_data["goals_scored_home"] = data["goals_scored_home"].apply(lambda x: x if (x < 15) else 15)
    new_data["goals_scored_away"] = data["goals_scored_away"].apply(lambda x: x if (x < 15) else 15)
    new_data["goals_conceded_home"] = data["goals_conceded_home"].apply(lambda x: x if (x < 15) else 15)
    new_data["goals_conceded_away"] = data["goals_conceded_away"].apply(lambda x: x if (x < 15) else 15)

    return new_data


def test_bayesian_networks_model(learn_data, test_data):
    """
    Function, that tests `test_data`, based on Bayesian network, built
    on `learn_data`. Returns tuple of brier_score, precision, 
    recall, F1, confusion_matrix and profit
    """


    learn_data = discretize_dataframe(learn_data)
    test_data = discretize_dataframe(test_data)

    features = learn_data.drop(columns=["index", "odds_home", "odds_draw", "odds_away",
                              "goals_conceded_home", "goals_conceded_away",
                              "shots_given_home", "shots_given_away", 
                              "shots_conceded_home", "shots_conceded_away", 
                              "corners_difference_home", "corners_difference_away"])

    X_ = np.array(features)

    bayes_net_model = BayesianNetwork.from_samples(X_, algorithm='chow-liu', root=0)


    brier_score = 0
    modified_brier_score = 0
    confusion_matrix = [[0,0,0], [0,0,0], [0,0,0]]
    profit = 0

    test_features = test_data.drop(columns=["index", "odds_home", "odds_draw", "odds_away",
                              "goals_conceded_home", "goals_conceded_away",
                              "shots_given_home", "shots_given_away", 
                              "shots_conceded_home", "shots_conceded_away", 
                              "corners_difference_home", "corners_difference_away"])
    test_X = np.array(test_features)
    test_y = np.array(test_data["result"])
    
    test_odds = np.array(test_data[["odds_home", "odds_draw", "odds_away"]])
    
    for id, line_ in enumerate(test_X):

        line = list(line_)
        
        line[0] = None

        probabilites = [bayes_net_model.predict_proba([line])[0][0].parameters[0]["home win"],
                        bayes_net_model.predict_proba([line])[0][0].parameters[0]["draw"],
                        bayes_net_model.predict_proba([line])[0][0].parameters[0]["away win"]]

        if max(probabilites) == probabilites[0]:
            predicted = "home win"
            c_matrix_pred = 0
        elif max(probabilites) == probabilites[1]:
            predicted = "draw"
            c_matrix_pred = 1
        elif max(probabilites) == probabilites[2]:
            predicted = "away win"
            c_matrix_pred = 2
        else:
            raise ValueError

        if test_y[id] == "home win":
            brier__ = [1, 0, 0]
            c_matrix_act = 0
        elif test_y[id] == "draw":
            brier__ = [0, 1, 0]
            c_matrix_act = 1
        elif test_y[id] == "away win":
            brier__ = [0, 0, 1]
            c_matrix_act = 2
        else:
            raise ValueError

        odds = test_odds[id]

        # calculate modified brier score
        for i in range(3):
            modified_brier_score += (probabilites[i] - (1 / odds[i])) ** 2

        # calculate brier score
        for i in range(3):
            brier_score += (probabilites[i] - brier__[i]) ** 2

        # confusion matrix
        confusion_matrix[c_matrix_act][c_matrix_pred] += 1

        final_result = test_y[id]
        
        # calculate profit
        profit += (return_profit(odds, probabilites, final_result) / len(test_data))
        

    # calculate rest of the metrics
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0])
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2])

    F1 = 2 * (precision * recall) / (precision + recall)

    return brier_score / (3 * len(test_data)), precision, recall, F1, confusion_matrix, profit, modified_brier_score / (3 * len(test_data))
