from Utilities import run_naive_bayes_mapreduce,predict
# ---------------- EXAMPLE ----------------
if __name__ == "__main__":
    data = [
        "1\tspam\tWin money now",
        "2\tham\tHello, how are you?",
        "3\tspam\tClaim your prize",
        "4\tham\tLet's have lunch tomorrow",
        "5\tspam\tWin big cash now",
    ]

    model = run_naive_bayes_mapreduce(data, num_mapper_threads=2, num_reducer_threads=2)
    text_to_predict="Win cash prize now"
    pred_class, log_scores, probs = predict(model,text_to_predict)
    print("Predicted:", pred_class)
    print("Log-scores:", log_scores)
    print("Probabilities:", probs)