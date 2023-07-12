
  <p align="center"><a href="https://www.w3schools.com/python/python_ml_knn.asp" target="_blank"><img src="https://raw.githubusercontent.com/Mohammadnematizade/Machine-Learning/main/Assignment%2042/output/avatar-1727507864.jpg" width="300"></a></p>

# Assignment 42
## ANSUR II dataset
 Show heights for women and men on same plot.

![res](https://raw.githubusercontent.com/Mohammadnematizade/Machine-Learning/main/Assignment%2042/output/women%20%26%20men%20.png)
- ## A. Why is the data of men higher than the data of women?

  Because there are more male data than female data (because 1985 females and 4081 males).

- ## B. Why is the data of men more right than the data of women?
  Because men are taller.
  For a better comparison, the following chart with the same density is proposed.

  ![hist density](https://raw.githubusercontent.com/Mohammadnematizade/Machine-Learning/main/Assignment%2042/output/women%20and%20men%20height.png)
## Answer :
    🚹 Mean of Women height = 162.84 cm
    🚺 Mean of Men height = 175.62 cm

## Evaluate your KNN algorithm on the test dataset.
  | k      | 3      | 7      | 10      |15      |17       |
  | :---   | :----  | :----  | :----  | :----   | :----   |
  | Score  | 83.1%  | 84.7%  | 84.9%  | 85.2%   | 85.5%   |

## Evaluate the scikit-learn KNN :
  | k                                                                      | 3      | 7      | 10     | 15      |17     |
  | :---                                                                   | :----  | :----  | :----  | :----   | :----   |
  | Scores obtained by weight and stature features                         | 83.3%  | 84.5%  | 84.8%  | 85.2%   | 85.1%   |
  | Scores obtained by weight and features Gender                          | 97.9%  | 97.9%  | 98.1%  | 97.9%   | 97.9%   |
  | Scores obtained by weight, stature and Gender features                 | 97.3%  | 97.3%  | 97.7%  | 97.5%   | 97.4%   |
## Calculate confusion matrix for test dataset.

![Confusion Matrix me](https://raw.githubusercontent.com/Mohammadnematizade/Machine-Learning/main/Assignment%2042/output/Confusion%20Matrix%20in%20test%20data.png)
