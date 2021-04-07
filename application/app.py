import sys
sys.path.append('/Users/uvik/python_programs/ucu/final_project/DE-ML-project/')
print(sys.path)

import streamlit as st

from ml_solution_final import prediction


def main():
	st.title("How toxic is your text?")
	text = st.text_input("Input you comment here")
	model_output = prediction.predict_probability(text)
	result_in_percent = round(model_output, 3) * 100
	if text:
		st.write("Toxicty of you comment is: ", result_in_percent)
	print(result_in_percent)


if __name__ == "__main__":
	main()