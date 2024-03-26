import sys
import numpy as np
import pandas as pd
import math

def gaussian(x, mean, std):
    exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (std * math.sqrt(2 * math.pi))) * exponent

def main():
    # Read data from CSV file
    df = pd.read_csv("fruits.csv")
    
    # Separate data by class
    apples = df[df['fruit'] == 'apple']
    oranges = df[df['fruit'] == 'orange']

    # Calculate prior probabilities
    prior_apple = len(apples) / len(df)
    prior_orange = len(oranges) / len(df)

    # Calculate mean and standard deviation for apple and orange weights
    apple_fruit_weight = apples['fruit_weight']
    mean_apple = apple_fruit_weight.mean()
    std_apple = ((apple_fruit_weight - mean_apple) ** 2).mean() ** 0.5

    orange_fruit_weight = oranges['fruit_weight']
    mean_orange = orange_fruit_weight.mean()
    std_orange = ((orange_fruit_weight - mean_orange) ** 2).mean() ** 0.5

    # Do prediction on the test set via Naive Bayes
    count_correct = 0
    for index, fruit in df.iterrows():
        x = fruit['fruit_weight']
        prob_apple = gaussian(x, mean_apple, std_apple) * prior_apple
        prob_orange = gaussian(x, mean_orange, std_orange) * prior_orange

        predicted_fruit = 'apple' if prob_apple > prob_orange else 'orange'

        if predicted_fruit == fruit['fruit']:
            count_correct += 1

    # Calculate classification accuracy
    accuracy = (count_correct / len(df)) * 100
    print('Classification accuracy =', accuracy)

if __name__ == "__main__":
    sys.exit(int(main() or 0))






