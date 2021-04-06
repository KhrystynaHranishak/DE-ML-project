import sys
sys.path.append('/Users/uvik/python_programs/ucu/final_project/DE-ML-project/')
print(sys.path)

import streamlit as st

from ml_solution_final import prediction


def main():
	st.title("Prediction will student's pass or fail his or her exam!")


if __name__ == "__main__":
	main()