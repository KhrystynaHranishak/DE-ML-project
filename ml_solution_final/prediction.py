import joblib

MODEL = joblib.load('ml_solution_final/model/NB_classifier.pkl')


def predict_probability(text, model=MODEL):
    """
    The function returns a probability to be toxic for input text
    :param text: string
    :param model: pkl object (trained Naive Bayes classifier)
    :return: float in range [0, 1]
    """
    return model.predict_proba([text])[:, 1][0]


# if __name__ == '__main__':
#     print('Example of usage')
#     text_1 = "I can't make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of ""types of accidents""  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know"
#     text_2 = 'You are suck!!!'
#     print('Text 1 probability to be toxic: ', predict_probability(text_1))
#     print('Text 2 probability to be toxic: ', predict_probability(text_2))