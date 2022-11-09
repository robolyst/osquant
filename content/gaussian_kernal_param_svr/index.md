---
title: "Choosing the Gaussian Kernel's Parameter for Support Vector Regression"
blurb: "
Support vector regression models transform input vectors into a very high number of dimensions where the regression problem becomes linear. This new space is defined by a kernel function which has it's own parameters. A major drawback of these models is that they are slow to fit the kernel parameters. Here, we focus on quickly choosing the Gaussian kernel's bandwidth parameter. We develop an algorithm which attempts to match the distance between the input vectors to the distance between the output vectors using a linear combination of the kernel functions. We find that this algorithm is fast, robust on some datasets, and has a similar error to slower and more exhaustive methods.
"

date: "2016-01-07"
type: article
author: Adrian Letchford
katex: true
markup: "mmark"
draft: true
---

# Introduction

Consider a series of continuous values \\(y_i\\) that need to be modelled based on some corresponding vectors \\(\textbf{x}_i\\). A model that is potentially suitable for this task is called the support vector regression model. This maps the points \\(\textbf{x}_i\\) into a higher dimension denoted by \\( \varphi(\mathbf{x}_i) \\), and then models \\(y_i\\) by fitting a linear model based on a subset of \\(\textbf{x}\\) called support vectors.

Support vector regression models are part of a branch of models called support vector machines (SVM) [^Vapnik1998] [^Burges1998] [^Smola2004]. SVMs have been used for a wide variety of problems such as pattern recognition [^Wang2007], face recognition [^Phillips1998] [^Li2001], time series prediction [^Kim2003], image classification [^Chapelle1999a], system control [^Hong2015], and function approximation [^Vapnik1996].

Support vector regression is a powerful model because each input vector may be transformed into an extremely high number of dimensions. This transformation is rarely computationally efficient. However, the model's equations can be rearranged so that each transformed vector is paired with another in a dot product, like so: \\(\varphi(\mathbf{x}_i)^T\varphi(\mathbf{x}_j)\\). This dot product is usually defined with a very simple kernel function. The most common one is the Gaussian kernel:

$$
\begin{align}
\kappa(\mathbf{x}_i, \mathbf{x}_j) = \varphi(\mathbf{x}_i)^T\varphi(\mathbf{x}_j) =  e ^{ - \beta||\mathbf{x}_i - \mathbf{x}_j||^2} \label{1}
\end{align}
$$

A necessary task when fitting a support vector regression (SVR) model is tuning the kernel parameters. The most common method of selecting \\(\beta\\) in \\(\eqref{1}\\) and the parameters to a SVR is by a grid search with cross-validation [^Hong2015] [^Tang2009]. In a grid search, the parameter space is divided into a grid of points where each one is tested using cross-validation. Given a vast number of grid points, this is the most accurate method. Evaluating a SVR model's prediction error has a complexity on the order of \\(O(n^3)\\) where \\(n\\) is the number of data points. This cubic complexity makes the grid search the slowest method. Various alternatives do not necessarily work around the evaluation complexity. For example, one alternative is to test random points rather than use a grid [^Bergstra2012].

In this paper, we develop an algorithm to find a value for \\(\beta\\) from a set of training points. Our algorithm does not calculate the model's prediction error and runs in \\(O(n^2)\\) time. The resulting error is just as good as an exhaustive grid search. We show that, on some datasets, the selected value for \\(\beta\\) varies considerably between small subsets of data; which means that this algorithm can sometimes trade robustness for the improvement in speed.

# Finding the Gaussian kernel parameter

We present a method that selects a value for the Gaussian kernel's bandwidth by focusing on the distance between the output values \\(y\\). A kernel machine models the target value \\(y\\) as a function of the dot product between the input vectors \\(x\\). The dot product can be thought of as a measure of similarity between two vectors. The algorithm presented here attempts to select values for a kernel's parameters so that two \\(x\\) vectors are more similar if their corresponding \\(y\\) values are also similar.

{{<figure src="images/time_series_example.svg" title="Figure 1: An example time series." >}}
In this example, \\(y = \sin(x) + \sin(3x) + \sin(2x + 5)\\).
{{</figure>}}

Consider the example time series in Figure 1. In this example, we want to predict the next value in this series based on the last 20 values. We'll represent the last 20 values at time \\(t\\) as \\(\textbf{x}_t\\) and the target value as \\(y_t\\). The Gramian matrix, sometimes known as the feature matrix, is a matrix of dot products between each of the \\(\textbf{x}\\) vectors. We write this as \\(\textbf{K}\\):
$$
\begin{aligned}
\textbf{K} = \left[\begin{matrix}
  \kappa(\textbf{x}_0, \textbf{x}_0) & \dots & \kappa(\textbf{x}_0, \textbf{x}_n) \\\\
  \vdots & \ddots & \vdots \\\\
  \kappa(\textbf{x}_n, \textbf{x}_0) & \dots & \kappa(\textbf{x}_n, \textbf{x}_n)
\end{matrix} \right]
\end{aligned}
$$

For our example time series, the Gramian matrix with the Gaussian kernel is shown in Figure 2A. The value for \\(\beta\\) was chosen by hand and is not important in this example. Observe in Figure 2B what happens when we sort the vectors by their \\(y\\) values in an ascending order. Now, adjacent \\(\textbf{x}\\) vectors have a very close value for their corresponding target \\(y\\). Notice that cells close to the main diagonal are red (value close to 1), and the further a cell is from the diagonal the more blue (value close to 0) it is.

{{<figure src="images/time_series_gramian.svg" title="Figure 2: The Gramian matrix of the time series in Figure 1." >}}
**(A)** We set the problem of predicting the value of the example series in Figure 1 using the previous 20 values. We represent the value at time \\(t\\) with \\(y_t\\) and the previous 20 with \\(x_t\\). Here we shown the Gramian matrix where each cell is the value of the Gaussian function between two \\(x\\) vectors. We choose the bandwidth parameter to highlight the pattern, it's exactly value is not important. **(B)** If we sort the \\(x\\) vectors by their corresponding target value \\(y\\), the pattern is destroyed, however, a new pattern emerges. Now, values closer to the main diagonal are more red. That is, the kernel function of two \\(x\\) vectors tends to be closer to 1 when their corresponding \\(y\\)s are closer in value.
{{</figure>}}

Now that the vectors are sorted, the further away a cell is from the main diagonal, the more distant their target values. Ideally, just like in Fig. 2B, we want the \\(\textbf{x}\\) vectors that correspond to the cells to also be less similar. We can visualise this decrease in similarity by taking the mean value of each of the matrix's lower diagonals. We can ignore the upper diagonals because they are exactly the same--a Gramian matrix is symmetric. In Figure 3 we plot the mean of each of the diagonals in Fig. 2B. Again we can see that the kernel function's values are more similar when closer to the main diagonal. In this example, the average value decreases to zero the further from the main diagonal.

{{<figure src="images/time_series_diagonal.svg" title="Figure 3: Mean of the Gramian diagonals." >}}
**(A)** Using the sorted Gramian matrix in Figure 2B, we plot the mean of each diagonal. The first value on the left hand side is the first diagonal from the center. The last value is the cell in the lower left hand corner of the Gramian matrix. Diagonals closer to the center represent vectors that are closer together. Here we see that the mean of each diagonal decreases farther from the center. This suggests that when multiple \\(y\\)s are close together, their corresponding \\(x\\) vectors are also close. **(B)** We repeat the analysis in (A) with the exception of adding some Gaussian noise to the example time series. The least amount of noise is shown in blue and the most amount of noise is shown in red. This suggests that as noise is added to the data, \\(x\\) vectors are no longer close when their corresponding \\(y\\) values are close. We propose that the slope of this line may be a good indicator that the \\(x\\) vectors describe the \\(y\\) values. 
{{</figure>}}

If we add noise to the time series in Figure 1 and calculate the mean of each of the Gramian's diagonals, then this line becomes flatter and more erratic as more noise is added (Figure 3B). The slope of the mean diagonal line seems to be a crude measure of how the similarity between the \\(\textbf{x}\\) vectors match the similarity between the \\(y\\) values.

The algorithm in this paper chooses \\(\beta\\) such that the slope of this line is as negative as possible. For lack of a better name, we call this the ***diagonal slope algorithm***.

## Diagonal Slope Algorithm

We calculate the slope by taking the difference between each successive diagonal mean. These are then combined together in a weighted average to find the average slope. The weights are chosen to be the total number of cells in the corresponding diagonals.

Calculating the mean of each diagonal does not require us to compute the whole Gramian matrix. The main diagonal contains only 1s, and the matrix is symmetric. This means we only need the upper or lower triangle of this matrix. Here, our notation will use the lower triangle:

$$
\begin{aligned}
K = \left[\begin{matrix}
  1 \\\
  \kappa(\textbf{x}_1, \textbf{x}_0) & 1 \\\
  \vdots & \vdots & \ddots & 1 \\\
  \vdots & \vdots & \vdots & \ddots & \ddots \\\
  \kappa(\textbf{x}_n, \textbf{x}_0) & \dots  & \dots  & \dots & \kappa(\textbf{x}_n, \textbf{x}\_{n-1}) & 1
\end{matrix}\right]
\end{aligned}
$$

For simplicity's sake, we'll denote each cell with a single index instead of two, and refer to them with \\(p\\):
$$
\begin{aligned}
K &= \left[\begin{matrix}
    1 \\\
    p_0 & 1 \\\
    p_1 & p_{n-1} & 1 \\\
    p_2 & p_{n} & \ddots & 1 \\\
    \vdots & \vdots & \vdots & \ddots & \ddots \\\
    p\_{n-2} & \dots  & \dots  & \dots & p\_{n(n-1)/2} & 1
\end{matrix}\right] \\\
p_i &= \kappa(\textbf{x}\_{\text{row}(i)}, \textbf{x}_{\text{col}(i)}) \\\
\text{row}(i) &= \text{col}(i) + \text{diag}(i) + 1 \\\
\text{col}(i) &= \Bigg\lfloor{\frac{(2n - 1) - \sqrt{(2n - 1)^2 - 4 \cdot 2i}}{2}}\Bigg\rfloor \\\
\text{diag}(i) &= i - - \frac{- (2n-1)\text{col}(i) + \text{col}(i)^2}{2}
\end{aligned}
$$
The functions \\(\text{col}(i)\\), \\(\text{row}(i)\\), \\(\text{diag}(i)\\) give the column, row, and diagonal index of a cell \\(i\\) respectively. The functions are derived in the appendix.

Ignoring the main diagonal of 1s, there are \\(n-2\\) diagonals which we denote with \\(d\\). The zeroth mean diagonal is:
$$
d_0 = \frac{p_0 + p_{n-1} + \dots + p_{n(n-1)/2}}{n-1}
$$

and the \\(j^{\text{th}}\\) diagonal is:
$$
d_j = \frac{1}{n-2-j+1} \sum_{i=0}^{n-2-j} p_{in-i(i+1)/2+j}
$$

We calculate the slope by taking the weighted average of the differenced mean diagonal line:

$$
\begin{aligned}
S &= \frac{
    \frac{d\_1 -d\_0}{l\_1+l\_0} +
    \frac{d\_2 -d\_1}{l\_2+l\_1} + 
    \dots
}{
    (l\_1 + l\_0) + 
    (l\_2 + l\_1) + 
    \dots
} \\\
&= \frac{
		-d_0(l_0 + l_1) +
		\sum_{i=1}^{n-3} d_i(l_{i-1} - l_{i+1}) +
		d_{n-2}(l_{n-2} + l_{n-3})
	}{
		l_0+2\sum_{i=1}^{n-3}l_i + l_{n-2}
	}
\end{aligned}
$$

This is a linear combination of \\(p\\)s and we can write it as such:
$$
\begin{align}
S = \sum_{i=0}^{n(n-1)/2} w_i p_i \label{2}
\end{align}
$$


Each weight is calculated as:

$$
w_i^* = \frac{1}{n-2-\text{diag}(i)+1} \times \frac{1}{l_0+2\sum_{j=1}^{n-3}l_j + l_{n-2}}
$$
$$
\begin{align}
    w_i &= \begin{cases}
        -w_i^*(l_0 + l_1), & \text{if}\ \text{diag}(i)= 0 \\\
        -w_i^\*(l_{\text{diag}(i)-1} - l_{\text{diag}(i)+1}), & \text{if}\ 1 \geq \text{diag}(i) \geq n-3 \\\
        w_i^\*(l_{n-2} + l_{n-3}), & \text{if}\ \text{diag}(i)= n-2
    \end{cases} \label{3}
\end{align}
$$

The objective function \\(\eqref{2}\\) is shown in Figure 4A using the time series example in Figure 1. 

{{<figure src="images/objective_function_example.svg" title="Figure 4: Examples of the objective function." >}}
**(A)** The objective function on the example time series in Figure 1. **(B)** The objective function on a randomly generated time series shows that there is no guarantee that it will be convex.
{{</figure>}}

All weights in the objective function \\(\eqref{2}\\) are positive except for when \\(\text{diag}(i)= 0\\) \\(\eqref{3}\\). As a result, there is no guarantee that the objective function \\(\eqref{2}\\) has a single minima. Figure 4B shows an example of the objective function when the time series values are drawn from a Gaussian distribution. Because the objective function is not guaranteed to have a single minima, we use an exhaustive search to find the global minima in this paper.

# Results

We use a dataset called the *Mashable News Popularity* dataset [^Mashable] to evaluate the diagonal slope algorithm described in the previous section. This dataset is a collection of features on 2,000 articles from the online news site [Mashable](https://mashable.com/). The task is to predict the number of times each article is shared. We provide a full description of this dataset in Appendix 2.

We randomly split this dataset into two even subsets, one for training and one for testing. We convert any non-numeric dimensions into an orthogonal representation. We normalise the training dataset by subtracting the mean from each dimension and dividing by their standard deviations. We normalise the testing dataset using the same mean and standard deviation.

We evaluate the diagonal slope algorithm's objective function for each \\(\beta\\) from \\(10^{-3}\\) to \\(10^{3}\\) at \\(80\\) logarithmic evenly spaced values. After finding the \\(\beta\\) which minimises the diagonal slope, we use a grid search to fit the \\(\epsilon\\) and \\(C\\) parameters of a support vector regressor. We evaluate \\(\epsilon\\) at 5 logarithmic evenly spaced values from \\(10^{-3}\\) to \\(10^1\\), and \\(C\\) at 7 logarithmic evenly spaced values from \\(10^{-3}\\) to \\(10^{3}\\).

We find that the support vector regressor fitted with the diagonal slope algorithm has a mean absolute error (MAE) of 0.30 on the testing data while the plain grid search's MAE is also 0.30. These results show that our algorithm achieves a comparable forecasting error with an exhaustive grid search but at a fraction of the time.

We verify this result by performing the same analysis on six other datasets. There is a dataset for predicting Portuguese student's mathematics grades [^Mathematics], predicting Portuguese student's Portuguese grades [^Portuguese], classifying the age of a song [^Music], predicting housing prices in Boston [^Housing], predicting the number of comments on blog posts [^Blog], and predicting the number of bicycles rented in a bike-sharing system [^Bike]. Again, we provide a full description of each of the datasets in Appendix 2.

{{<figure src="images/accuracy.svg" title="Figure 5: The error is just as good as the standard slow and exhaustive method." >}}
The dataset *Mashable News Popularity* is a collection of articles from the news website [Mashable](https://mashable.com/) and the aim is to predict the the number of times each article is shared based on a set of features. We randomly split this dataset evenly into a training and testing set. We train the proposed diagonal slope algorithm and a basic grid search on the training sample and evaluate the mean absolute percentage error (MAPE) on the testing set. We repeat this process on the remaining datasets. We find that the error from the diagonal slope algorithm is never more than 5\% greater than the grid search.
{{</figure>}}

The results are shown in Figure 5. While the diagonal slope algorithm's error is slightly higher or lower on some datasets, it is never greater by more than 5\%. These results suggest that our algorithm does not trade accuracy for speed.

Because kernel algorithms can be very slow when using large datasets, a common practice is to fit a model on a small subset. However, there is a possibility that an algorithm for choosing the kernel parameters is very sensitive to the subset. If the subset changes, the selected parameters might also change quite significantly. An algorithm that is able to choose consistent kernel parameters is called robust.

To check whether or not our algorithm is more or less robust than the grid search, we draw with replacement 30 random subsets of 100 points from each datasets. We use the diagonal slope algorithm and grid search to find a suitable value for \\(\beta\\) on each subset. We then compare the variance of the \\(\beta\\)s as chosen by the two algorithms.

{{<figure src="images/robustness.svg" title="Figure 6: The proposed algorithm is not always robust." >}}
We take 30 random samples of 100 articles from the *Mashable News Popularity* dataset. We fit both the diagonal slope algorithm and a basic grid search to each of the samples and record the fitted \\(\beta\\). We then calculate the variance between the selected values. Here we show that the value for \\(\beta\\) selected by the diagonal slope algorithm has a greater variance than if selected by the grid search. We repeat this analysis on all the data sets and find that the diagonal slope algorithm's variance is greater for three of the datasets. This demonstrates that the proposed algorithm is not always robust.
{{</figure>}}

We show the variance of the \\(\beta\\)s in Figure 6. On three data sets, the diagonal slope algorithm has a greater variance than the grid search. We conclude that the diagonal slope algorithm is not always as robust as a grid search.

# Summary

In this paper, we propose an algorithm for choosing kernel parameters in regression tasks which we call the *diagonal slope algorithm*. 

We test this algorithm on a variety of datasets using the Gaussian kernel and support vector regression. We find that the algorithm's accuracy is comparable to a grid search which represents the best possible result. However, we find that the algorithm's choice of the Gaussian kernel's bandwidth parameter is sometimes sensitive to changes in the training data.

Our results suggest that the speed gained from using the *diagonal slope algorithm* does not reduce performance, but it does sometimes reduce robustness. This algorithm can potentially be applied to kernels other than the Gaussian kernel and to kernels with more than one parameter. Such kernels will need to be a measure of similarity or be normalised. A normalised kernel represents the correlation of two points in feature space.

# Appendices 
## Appendix 1 - Calculating the row, column and diagonal indices of a Gramian matrix cell

A Gramian matrix is a matrix of the dot products between a set of vectors. The matrix is symmetric and the values along the main diagonal are identical because they correspond to the dot product between two identical points. The only unique values in a Gramian matrix are in the upper or lower triangle. Here, we will use the lower triangle.

{{<figure src="images/grid.svg" title="Figure 7: Matrix indexing." >}}
Because a Gramian marix is symmetric and the diagonals are all the same value we only need to index the lower triangle. Here, we depict how each cell is indexed with a top to bottom approach. The indexes for the columns, rows and diagonals are also shown. 
{{</figure>}}

We can index each cell in the lower triangle of a Gramian matrix as shown in Figure 7 which represents the matrix made by $6$ vectors. In this appendix, we derive the column, row and diagonal index from the cell index.


We represent the cell index with \\(i\\) and the total number of vectors which produce the Gramian matrix with \\(n\\). First we figure out the function \\(\text{col}(i)\\) which calculates the column index. Given a column index \\(c\\), the \\(i\\) that sits at the top of that column is:
$$
i = \frac{n(n-1)}{2} - \frac{(n - c)(n - c-1)}{2}
$$

We can factor this down to:
$$
%i &= n^{(T)} - (n - c)^{(T)} \\
%i &= n^{(T)} - \frac{(n-c)(n-c-1)}{2} \\
%i &= n^{(T)} - \frac{n^2 - n - (2n-1)c + c^2}{2} \\
%i - n^{(T)} &= - \frac{n^2 - n - (2n-1)c + c^2}{2} \\
%i - n^{(T)} + n^{(T)} &=  -\frac{- (2n-1)c + c^2}{2} \\
i =  -\frac{- (2n-1)c + c^2}{2}
$$

and solve for \\(c\\):
$$
\begin{aligned}
%2i &=  - (2n-1)c + c^2 \\
%0 &=  - (2n-1)c + c^2 - 2i \\
c &= \frac{(2n - 1) \pm \sqrt{(2n - 1)^2 - 4 \cdot 2i}}{2} \\\
c &= \frac{(2n - 1) - \sqrt{(2n - 1)^2 - 4 \cdot 2i}}{2}
\end{aligned}
$$

For any \\(i\\) that does not sit at the top of a column, \\(c\\) will be between the column index and the next column's index. We have to floor this value to get the column index of \\(i\\):
$$
\text{col}(i) = \Bigg\lfloor\frac{(2n - 1) - \sqrt{(2n - 1)^2 - 4 \cdot 2i}}{2}\Bigg\rfloor
$$

The diagonal is the number of spaces the \\(i\\) sits from the top of it's column:
$$
\text{diag}(i) = i - - \frac{- (2n-1)\text{col}(i) + \text{col}(i)^2}{2}
$$

The row is the column plus the diagonal:
$$
\text{row}(i) = \text{col}(i) + \text{diag}(i) + 1
$$

## Appendix 2 - Datasets

We use a variety of real-world datasets that cover a wide range of topics such as micro-blogging and education. 

**Mashable News Popularity dataset**

\textit{Mashable} is an online news site where readers can share news articles. In one study, researchers collected almost 40,000 articles from \textit{Mashable} and extracted a set of features from each one including number of positive words, LDA topic distribution and publication time \cite{Fernandes2015}. The task is to predict the number of times each article was shared.

The Mashable News Popularity dataset contains 39,797 news articles. In this study we use a random sample of 2,000 articles.

**Portuguese Students, mathematics class dataset**

To improve understanding of why Portugal’s student failure rate is high, one study collected data from two Portuguese schools \cite{Cortez2008}. The researchers collected data on each student by conducting a survey which asked questions ranging from their romantic relationship to their parent's alcohol consumption. They also collected the students final grades. 

This dataset contains data on 395 students in both schools and their final mathematics grade. The task is to correctly predict their final grade.

**Portuguese Students, Portuguese class dataset**

As well as the students' mathematics grades, the study also reports their Portuguese grades. This dataset contains data on 649 students and their Portuguese grades. Again, the task is to correctly predict their final grade.

**Million Song**

This dataset is a collection of 515,245 songs selected from the Million Song Dataset \cite{Bertin-Mahieux2011}. It contains 90 attributes for each song which are the means and covariances of the timbre across segments within a song. The task is to predict the year each song was produced. In this study we use a random sample of 2,000 songs.

**Housing**

This is a dataset of 506 houses in Boston and their prices as used by \cite{Quinlan1993}. The task is to predict the median housing price from a set of features which includes crime rate, average number of rooms, tax rates and socio-economic states.

**Blog Feedback**

This is a dataset of 60,000 blog posts from around 1,200 Hungarian blogs. There are 280 recorded features for each post including number of links, number of comments received thus far and the most discriminative features from a bag of words analysis. The goal is to predict the number of comments each post will recieve in the next 24 hours. This dataset was used by \cite{Buza2014}. We use a random subset of 2,000 blog posts.

**Bike Sharing**

Bike sharing systems completely automate the rental and return process of renting bikes. Users are able to rent a bike from one location, and return the bike to another. This dataset contains daily records of a bike-sharing system called Captial Bike Sharing in Washington, D.C., USA. There are two years of records from the 1$^\text{st}$ of January 2011 to the 31$^\text{st}$ of December 2012 for a total of 731 days. This dataset was used by \cite{Fanaee-T2014} to test an event detection algorithm. In this paper, the task is to predict the number of rented bikes from the day's weather records.





{{% refbook
    id="Vapnik1998"
    author="Vapnik, Vladimir"
    title="Statistical learn theory"
    year="1998"
    publisher="Wiley & Sons"
    address="New York"
%}}

{{% refjournal 
    id="Burges1998"
    author="Burges, Christopher J C"
    title="A tutorial on support vector machines for pattern recognition"
    journal="Data Mining and Knowledge Discovery"
    year="1998"
    volume="2"
    number="2"
    pages="121--167"
    publisher="Springer"
%}}

{{% refjournal 
    id="Smola2004"
    author="Smola, AlexJ. and Schölkopf, Bernhard"
    title="A tutorial on support vector regression"
    journal="Statistics and Computing"
    year="2004"
    volume="14"
    number="3"
    pages="199--222"
    publisher="Kluwer Academic Publishers"
%}}

{{% refproceedings 
    id="Phillips1998"
    author="Phillips, P Jonathon"
    title="Support Vector Machines Applied to Face Recognition"
    booktitle="Proceedings of the 1998 Conference on Advances in Neural Information Processing Systems II"
    year="1999"
    pages="803--809"
    publisher="MIT Press"
    address="Cambridge, MA, USA"
%}}

{{% refjournal 
    id="Wang2007"
    author="Wang, Tai-Yue and Chiang, Huei-Min"
    title="Fuzzy support vector machine for multi-class text categorization"
    journal="Information Processing & Management"
    year="2007"
    volume="43"
    number="4"
    pages="914--929"
    publisher="Kluwer Academic Publishers"
%}}

{{% refproceedings 
    id="Li2001"
    author="Li, S Z and Fu, QingDong and Gu, Lie and Scholkopf, Bernhard and Cheng, Yimin and Zhang, Hongjiag"
    title="Kernel machine based learning for multi-view face detection and pose estimation"
    booktitle="Computer Vision, 2001. ICCV 2001. Proceedings. Eighth IEEE International Conference on"
    year="2001"
    pages="674--679"
    volume="2"
    publisher="MIT Press"
    address="Cambridge, MA, USA"
%}}

{{% refjournal 
    id="Kim2003"
    author="Kim, Kyoung-jae"
    title="Financial time series forecasting using support vector machines"
    journal="Neurocomputing"
    year="2003"
    volume="55"
    number="1-2"
    pages="307--319"
%}}

{{% refjournal 
    id="Chapelle1999a"
    author="Chapelle, O and Haffner, P and Vapnik, V N"
    title="Support vector machines for histogram-based image classification"
    journal="Neural Networks, IEEE Transactions on"
    year="1999"
    volume="10"
    number="5"
    pages="1055--1064"
%}}

{{% refjournal 
    id="Hong2015"
    author="Hong, X and Chen, S and Gao, J and Harris, C J"
    title="Nonlinear Identification Using Orthogonal Forward Regression With Nested Optimal Regularizationn"
    journal="Neural Networks, IEEE Transactions on"
    year="2015"
    volume="PP"
    number="99"
    pages="1"
%}}


{{% refproceedings 
    id="Vapnik1996"
    author="Vapnik, Vladimir and Golowich, Steven E and Smola, Alex"
    title="Support Vector Method for Function Approximation, Regression Estimation, and Signal Processing"
    booktitle="Advances in Neural Information Processing Systems"
    year="1996"
    pages="281--287"
    volume="9"
    publisher="MIT Press"
    address="Cambridge, MA, USA"
%}}

{{% refjournal 
    id="Tang2009"
    author="Tang, Yaohua Tang Yaohua and Guo, Weimin Guo Weimin and Gao, Jinghuai Gao Jinghuai"
    title="Efficient model selection for Support Vector Machine with Gaussian kernel function"
    journal="IEEE Symposium on Computational Intelligence and Data Mining"
    year="2009"
    pages="40-45"
%}}

{{% refjournal 
    id="Bergstra2012"
    author="Bergstra, James and Bengio, Yoshua"
    title="Random Search for Hyper-Parameter Optimization"
    journal="Journal of Machine Learning Research"
    year="2012"
    volume="13"
    number=""
    pages="281--305"
%}}

[^Mashable]: [Online news popularity dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
[^Mathematics]: [Student performance, mathematics dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
[^Portuguese]: [Student performance, Portuguese dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
[^Music]: [Age of song dataset](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD)
[^Housing]: [House prices dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
[^Blog]: [Blog feedback dataset](https://archive.ics.uci.edu/ml/datasets/BlogFeedback)
[^Bike]: [Bike sharing dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)



<!-- 
@article{Adankon2006,
abstract = {Model selection for support vector machines concerns the tuning of SVM hyperparameters as C controlling the amount of overlap and the kernel parameters. Several criteria developed for tuning the SVM hyperparameters, may not be differentiable w.r.t. C, consequently, gradient-based optimization methods are not applicable. In this paper, we propose a new formulation for SVM that makes possible to include the hyperparameter C in the definition of the kernel parameters. Then, tuning hyperparameters for SVM is equivalent to choosing the best values of kernel parameters. We tested this new formulation for model selection by using the criterion of empirical error, technique based on generalization error minimization through a validation set. The experiments on different benchmarks show promising results confirming our approach.},
author = {Adankon, M.M. and Cheriet, M.},
doi = {10.1109/IJCNN.2006.246912},
file = {:home/adrian/Documents/Mendeley Desktop/Adankon, Cheriet - 2006 - New Formulation of SVM for Model Selection.pdf:pdf},
isbn = {0-7803-9490-9},
issn = {10987576},
journal = {The 2006 IEEE International Joint Conference on Neural Network Proceedings},
pages = {1900--1907},
title = {{New Formulation of SVM for Model Selection}},
year = {2006}
}
@article{Ahn2010,
author = {Ahn, Jeongyoun},
file = {:home/adrian/Documents/Mendeley Desktop/Ahn - 2010 - A stable hyperparameter selection for the Gaussian RBF kernel for discrimination.pdf:pdf},
journal = {Statistical Analysis and Data Mining},
keywords = {classification,data embedding,hyperparameter selection,kernel method,stability,support vector machines},
number = {3},
pages = {142--148},
publisher = {Wiley Subscription Services, Inc., A Wiley Company},
title = {{A stable hyperparameter selection for the Gaussian RBF kernel for discrimination}},
volume = {3},
year = {2010}
}
@article{Baldi2014,
abstract = {Collisions at high-energy particle colliders are a traditionally fruitful source of exotic particle discoveries. Finding these rare particles requires solving difficult signal-versus-background classification problems, hence machine-learning approaches are often used. Standard approaches have relied on ‘shallow’ machine-learning models that have a limited capacity to learn complex nonlinear functions of the inputs, and rely on a painstaking search through manually constructed nonlinear features. Progress on this problem has slowed, as a variety of techniques have shown equivalent performance. Recent advances in the field of deep learning make it possible to learn more complex functions and better discriminate between signal and background classes. Here, using benchmark data sets, we show that deep-learning methods need no manually constructed inputs and yet improve the classification metric by as much as 8{\%} over the best current approaches. This demonstrates that deep-learning approaches can improve the power of collider searches for exotic particles.},
annote = {Supplementary information available for this article at http://www.nature.com/ncomms/2014/140702/ncomms5308/suppinfo/ncomms5308{\_}S1.html},
author = {Baldi, P and Sadowski, P and Whiteson, D},
file = {:home/adrian/Documents/Mendeley Desktop/Baldi, Sadowski, Whiteson - 2014 - Searching for exotic particles in high-energy physics with deep learning.pdf:pdf},
journal = {Nat Commun},
number = {4308},
publisher = {Nature Publishing Group, a division of Macmillan Publishers Limited. All Rights Reserved.},
title = {{Searching for exotic particles in high-energy physics with deep learning}},
volume = {5},
year = {2014}
}
@article{Beltrami2015,
author = {Beltrami, Monica and da Silva, Arinei Carlos Lindbeck},
file = {:home/adrian/Documents/Mendeley Desktop/Beltrami, Silva - 2015 - Grid - Quadtree Algorithm for Support Vector Classification Parameters Selection.pdf:pdf},
journal = {Applied Mathematical Sciences},
keywords = {grid search,parameters selection,quadtree,support vector machine},
number = {2},
pages = {75--82},
title = {{Grid - Quadtree Algorithm for Support Vector Classification Parameters Selection}},
volume = {9},
year = {2015}
}
@inproceedings{Bertin-Mahieux2011,
author = {Bertin-Mahieux, Thierry and Ellis, Daniel P W and Whitman, Brian and Lamere, Paul},
booktitle = {Proceedings of the 12th International Conference on Music Information Retrieval ({\{}ISMIR{\}} 2011)},
title = {{The Million Song Dataset}},
year = {2011}
}
@article{Blundell1998,
author = {Blundell, Richard and Duncan, Alan},
journal = {The Journal of Human Resources},
number = {1},
pages = {62--87},
title = {{Kernel regression in empirical microeconomics}},
volume = {33},
year = {1998}
}
@incollection{Buza2014,
author = {Buza, Krisztian},
booktitle = {Data Analysis, Machine Learning and Knowledge Discovery SE - 16},
editor = {Spiliopoulou, Myra and Schmidt-Thieme, Lars and Janning, Ruth},
pages = {145--152},
publisher = {Springer International Publishing},
series = {Studies in Classification, Data Analysis, and Knowledge Organization},
title = {{Feedback Prediction for Blogs}},
year = {2014}
}

@article{Chapelle1999,
author = {Chapelle, Olivier and Vapnik, Vladimir},
file = {:home/adrian/Documents/Mendeley Desktop/Chapelle, Vapnik - 1999 - Model selection for support vector machines.pdf:pdf},
journal = {Advances in neural information processing systems},
number = {1},
pages = {230--236},
title = {{Model selection for support vector machines}},
url = {http://olivier.chapelle.cc/pub/nips99.pdf},
volume = {12},
year = {1999}
}
@article{Chapelle2002,
abstract = {The problem of automatically tuning multiple parameters for pattern recognition Support Vector$\backslash$nMachines (SVMs) is considered. This is done by minimizing some estimates of the generalization error of SVMs$\backslash$nusing a gradient descent algorithm over the set of parameters. Usual methods for choosing parameters, based$\backslash$non exhaustive search become intractable as soon as the number of parameters exceeds two. Some experimental$\backslash$nresults assess the feasibility of our approach for a large number of parameters (more than 100) and demonstrate$\backslash$nan improvement of generalization performance.$\backslash$n},
author = {Chapelle, Olivier and Vapnik, Vladimir and Bousquet, Olivier and Mukherjee, Sayan},
file = {:home/adrian/Documents/Mendeley Desktop/Chapelle et al. - 2002 - Choosing multiple parameters for support vector machines.pdf:pdf},
journal = {Machine Learning},
keywords = {Feature selection,Gradient descent,Kernel selection,Leave-one-out procedure,Support vector machines},
number = {1-3},
pages = {131--159},
title = {{Choosing multiple parameters for support vector machines}},
volume = {46},
year = {2002}
}
@inproceedings{Cortez2008,
abstract = {Although the educational level of the Portuguese population has improved in the last decades, the statistis keep Portugal at Europe’s tail end due to its high student failure rates. In particular, lack of success in The core classes of Mathematics and the Portuguese language is extremely serious. On the other hand, The ﬁelds of Business Intelligence (BI)/Data Mining (DM), which aim at extracting high-level knowledge from Faw data, oﬀer interesting automated tools that can aid The education domain. The present work intends to approach student achievement in secondary education using BI/DM techniques. Recent real-world data (e.g. student grades, demographic, social and school related features) was collected by using school reports and questionnaires. The two core classes (i.e. Mathematics Ana Portuguese) were modeled under binary/ﬁve-level classiﬁcation and regression tasks. Also, four DM models (i.e. Decision Trees, Random Forest, Neural Networks and Support Vector Machines) and three input selections (e.g. with and without previous grades) Ade tested. The results show that a good predictive accuracy can be achieved, provided that the ﬁrst and/or Second school period grades are available. Although student achievement is highly inﬂuenced by past evaluations, an explanatory analysis has shown that there are also other relevant features (e.g. number of absences, parent’s job and education, alcohol consumption). As a direct outcome of this research, more eﬃcient student prediction tools can be be developed, improving the quality of education and enhancing school resource management.},
address = {A. Brito and J. Teixeira},
author = {Cortez, Paulo and Silva, Alice},
booktitle = {In the Proceedings of 5 th Annual Future Business Technology Conference},
file = {:home/adrian/Documents/Mendeley Desktop/Cortez, Silva - 2008 - Using Data Mining To Predict Secondary School Student Performance.pdf:pdf},
keywords = {business intelligence in education,classification and,decision trees,random forest,regression},
number = {2000},
pages = {pp. 5--12},
title = {{Using Data Mining To Predict Secondary School Student Performance}},
volume = {2003},
year = {2008}
}
@article{Duan2003,
author = {Duan, Kaibo and Keerthi, S.Sathiya and Poo, Aun Neow},
doi = {10.1016/S0925-2312(02)00601-X},
issn = {09252312},
journal = {Neurocomputing},
keywords = {generalization error bound,model selection,svm},
month = {apr},
pages = {41--59},
title = {{Evaluation of simple performance measures for tuning SVM hyperparameters}},
url = {http://linkinghub.elsevier.com/retrieve/pii/S092523120200601X},
volume = {51},
year = {2003}
}
@phdthesis{Eigensatz2006,
author = {Eigensatz, Michael},
title = {{Insights into the Geometry of the Gaussian Kernel and an Application in Geometric Modeling}},
year = {2006}
}
@article{Fanaee-T2014,
author = {Fanaee-T, Hadi and Gama, Joao},
journal = {Progress in Artificial Intelligence},
keywords = {Background knowledge,Ensemble learning,Event detection,Event labeling},
number = {2-3},
pages = {113--127},
publisher = {Springer Berlin Heidelberg},
title = {{Event labeling combining ensemble detectors and background knowledge}},
volume = {2},
year = {2014}
}
@inproceedings{Fernandes2015,
address = {Coimbra, Portugal},
author = {Fernandes, K. and Vinagre, P. and Cortez, P.},
booktitle = {Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence},
title = {{A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News}},
year = {2015}
}
@inproceedings{Giesen2014,
address = {Beijing, China},
author = {Giesen, Joachim and Wieschollek, Patrick and Jena, U N I and Info, Patrick Wieschollek},
booktitle = {Proceedings of the 31st International Conference on Machine Learni},
file = {:home/adrian/Documents/Mendeley Desktop/Giesen et al. - 2014 - Robust and Efficient Kernel Hyperparameter Paths with Guarantees.pdf:pdf},
title = {{Robust and Efficient Kernel Hyperparameter Paths with Guarantees}},
volume = {32},
year = {2014}
}
@article{Gold2003,
abstract = {We address the problem of model selection for Support Vector Machine (SVM) classification. For fixed functional form of the kernel, model selection amounts to tuning kernel parameters and the slack penalty coefficient C. We begin by reviewing a recently developed probabilistic frame-work for SVM classification. An extension to the case of SVMs with quadratic slack penalties is given and a simple approximation for the evidence is derived, which can be used as a criterion for model selection. We also derive the exact gradients of the evidence in terms of posterior averages and describe how they can be estimated numerically using Hybrid Monte-Carlo techniques. Though computationally demanding, the resulting gradient ascent algorithm is a useful baseline tool for probabilistic SVM model selection, since it can locate maxima of the exact (unapproximated) evidence. We then perform extensive experiments on several benchmark data sets. The aim of these experiments is to compare the performance of probabilistic model selection criteria with alternatives based on estimates of the test error, namely the so-called "span estimate" and Wahba's Generalized Approximate Cross-Validation (GACV) error. We find that all the "simple" model criteria (Laplace evidence approximations, and the span and GACV error estimates) exhibit multiple local optima with respect to the hyperparameters. While some of these give performance that is competitive with results from other approaches in the literature, a significant fraction lead to rather higher test errors. The results for the evidence gradient ascent method show that also the exact evidence exhibits local optima, but these give test errors which are much less variable and also consistently lower than for the simpler model selection criteria. ?? 2003 Elsevier B.V. All rights reserved.},
archivePrefix = {arXiv},
arxivId = {cond-mat/0203334},
author = {Gold, Carl and Sollich, Peter},
doi = {10.1016/S0925-2312(03)00375-8},
eprint = {0203334},
file = {:home/adrian/Documents/Mendeley Desktop/Gold, Sollich - 2003 - Model selection for support vector machine classification.pdf:pdf},
isbn = {09252312},
issn = {09252312},
journal = {Neurocomputing},
keywords = {Bayesian evidence,Classification,Model selection,Probabilistic methods,Support vector machines},
number = {1-2},
pages = {221--249},
primaryClass = {cond-mat},
title = {{Model selection for support vector machine classification}},
volume = {55},
year = {2003}
}
@misc{Gretton2013,
author = {Gretton, Arthur},
pages = {1--29},
title = {{Introduction to RKHS , and some simple kernel algorithms}},
year = {2013}
}
@misc{Gurram2010,
author = {Gurram, P and Kwon, Heesung},
booktitle = {Geoscience and Remote Sensing Symposium (IGARSS), 2010 IEEE International},
doi = {10.1109/IGARSS.2010.5649859},
isbn = {2153-6996 VO -},
keywords = {Bandwidth,Chemical Plume Detection,Chemicals,Ensemble Learning,Gaussian kernel based SVM ensemble learning,Hyperspectral imaging,Kernel,Kernel Parameter Optimization,Optimization,SVM,SVM classifier,Sparse Kernel Learning,atmospheric techniques,ensemble decision,full diagonal bandwidth SVM ensemble learning,full diagonal bandwidth parameter matrix,geophysical signal processing,gradient descent algorithm,gradient methods,hyperspectral chemical plume detection,learning (artificial intelligence),majority voting,remote sensing,signal classification,sparse kernel based SVM ensemble learning,spectral feature subspaces,support vector machines,weak classifier},
pages = {2804--2807},
title = {{A full diagonal bandwidth gaussian kernel SVM based ensemble learning for hyperspectral chemical plume detection}},
year = {2010}
}
@incollection{Hinton2002,
address = {Cambridge, MA, USA},
author = {Hinton, Geoffrey and Roweis, Sam},
booktitle = {Advances in Neural Information Processing Systems},
keywords = {Visualization},
mendeley-tags = {Visualization},
pages = {833--840},
publisher = {The MIT Press},
title = {{Stochastic Neighbor Embedding}},
volume = {15},
year = {2002}
}

@inproceedings{Jaakkola1999,
author = {Jaakkola, Tommi S and Haussler, David},
booktitle = {Proceedings of the 1999 Conference on AI and Statistics},
title = {{Probabilistic Kernel Regression Models}},
year = {1999}
}
@article{Jeng2006,
author = {Jeng, Jt},
file = {:home/adrian/Documents/Mendeley Desktop/Jeng - 2006 - Hybrid approach of selecting hyperparameters of support vector machine for regression.pdf:pdf},
journal = {IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics},
number = {3},
pages = {699--709},
title = {{Hybrid approach of selecting hyperparameters of support vector machine for regression}},
url = {http://ieeexplore.ieee.org/xpls/abs{\_}all.jsp?arnumber=1634661},
volume = {36},
year = {2006}
}
@inproceedings{Joachims2000,
address = {San Franciso, CA},
author = {Joachims, T.},
booktitle = {Conference on Machine Learning},
publisher = {Morgan Kaufman},
title = {{Estimating the generalization performance of a SVM efficiently}},
year = {2000}
}
@article{Kaneko2015,
author = {Kaneko, Hiromasa and Funatsu, Kimito},
doi = {10.1016/j.chemolab.2015.01.001},
file = {:home/adrian/Documents/Mendeley Desktop/Kaneko, Funatsu - 2015 - Fast optimization of hyperparameters for support vector regression models with highly predictive ability.pdf:pdf},
issn = {01697439},
journal = {Chemometrics and Intelligent Laboratory Systems},
keywords = {Computational time,Hyperparameter,Optimization,Predictive ability,Support vector regression,support vector regression},
pages = {64--69},
publisher = {Elsevier B.V.},
title = {{Fast optimization of hyperparameters for support vector regression models with highly predictive ability}},
url = {http://linkinghub.elsevier.com/retrieve/pii/S0169743915000039},
volume = {142},
year = {2015}
}
@article{Laanaya2011,
abstract = {We propose a new method for general Gaussian kernel hyperparameter optimization for support vector machines classification. The hyperparameters are constrained to lie on a differentiable manifold. The proposed optimization technique is based on a gradient-like descent algorithm adapted to the geometrical structure of the manifold of symmetric positive-definite matrices. We compare the performance of our approach with the classical support vector machine for classification and with other methods of the state of the art on toy data and on real world data sets. © 2011 Elsevier B.V. All rights reserved.},
author = {Laanaya, Hicham and Abdallah, Fahed and Snoussi, Hichem and Richard, C{\'{e}}dric},
doi = {10.1016/j.patrec.2011.05.009},
file = {:home/adrian/Documents/Mendeley Desktop/Laanaya et al. - 2011 - Learning general Gaussian kernel hyperparameters of SVMs using optimization on symmetric positive-definite matri.pdf:pdf},
issn = {01678655},
journal = {Pattern Recognition Letters},
keywords = {General Gaussian kernel,Kernel optimization,Support vector machines,Symmetric positive-definite matrices manifold},
number = {13},
pages = {1511--1515},
publisher = {Elsevier B.V.},
title = {{Learning general Gaussian kernel hyperparameters of SVMs using optimization on symmetric positive-definite matrices manifold}},
url = {http://dx.doi.org/10.1016/j.patrec.2011.05.009},
volume = {32},
year = {2011}
}
@article{Lee,
author = {Lee, Martin M S and Keerthi, Sathiya and Ong, Chong Jin and Decoste, Dennis},
file = {:home/adrian/Documents/Mendeley Desktop/Lee et al. - Unknown - An Efficient Method for Computing Leave-One-Out Error in Support Vector Machines with Gaussian Kernels.pdf:pdf},
title = {{An Efficient Method for Computing Leave-One-Out Error in Support Vector Machines with Gaussian Kernels}}
}
@article{Lendasse2005,
abstract = {This paper presents a new method for the selection of the two hyperparameters of Least Squares Support Vector Machine (LS-SVM) approximators with Gaussian Kernels. The two hyperparameters are the width $\sigma$ of the Gaussian kernels and the regularization parameter $\lambda$. For different values of $\sigma$, a Nonparametric Noise Estimator (NNE) is introduced to estimate the variance of the noise on the outputs. The NNE allows the determination of the best $\lambda$ for each given $\sigma$. A Leave-one-out methodology is then applied to select the best $\sigma$. Therefore, this method transforms the double optimization problem into a single optimization one. The method is tested on 2 problems: a toy example and the Pumadyn regression Benchmark.},
author = {Lendasse, Amaury and Ji, Yongnan and Reyhani, Nima and Verleysen, Michel},
doi = {10.1007/11550907{\_}99},
file = {:home/adrian/Documents/Mendeley Desktop/Lendasse et al. - 2005 - LS-SVM hyperparameter selection with a nonparametric noise estimator.pdf:pdf},
isbn = {3540287558},
issn = {03029743},
keywords = {Computational, Information-Theoretic Learning with,Learning/Statistics {\&} Optimisation,Theory {\&} Algorithms},
pages = {625--630},
title = {{LS-SVM hyperparameter selection with a nonparametric noise estimator}},
url = {http://eprints.pascal-network.org/archive/00001666/},
year = {2005}
}
@book{Liu2010a,
address = {Hoboken, New Jersey},
author = {Liu, Weifeng and Principe, Jos{\'{e}} C and Haykin, Simon},
publisher = {Wiley {\&} Sons Inc.},
title = {{Kernel Adaptive Filtering: A Comprehensive Introduction}},
year = {2010}
}
@article{Loader1999,
author = {Loader, Clive R.},
journal = {The Annals of Statistics},
number = {2},
pages = {415--438},
title = {{Bandwidth Selection : Classical or Plug-In?}},
volume = {27},
year = {1999}
}
@article{Maaten2008,
author = {Maaten, Laurens Van Der and Hinton, Geoffrey},
journal = {Journal of Machine Learning},
keywords = {Visualization,dimensionality reduction,embedding algorithms,manifold learning,multidimensional scaling,visualization},
mendeley-tags = {Visualization},
pages = {2579--2605},
title = {{Visualizing Data using t-SNE}},
volume = {9},
year = {2008}
}
@incollection{Opper2000,
address = {Cambridge, MA},
author = {Opper, M. and Winther, O.},
booktitle = {Advances in large margin classifiers},
editor = {Smola, A. J. and Bartlett, P. L. and Scholkopf, B. and Schuurmans, D.},
pages = {311--326},
publisher = {MIT Press},
title = {{Gaussian processes and svm: Mean field and leave-one-out}},
year = {2000}
}
@article{Pang2011,
author = {Pang, Hong-xia and Dong, Wen-de and Xu, Zhi-hai and Feng, Hua-jun and Li, Qi and Chen, Yue-ting},
file = {:home/adrian/Documents/Mendeley Desktop/Pang et al. - 2011 - Novel linear search for support vector machine parameter selection.pdf:pdf},
journal = {Journal of Zhejiang University SCIENCE C},
keywords = {10,1631,c1100006,doi,jzus,linear search,motion prediction,parameter selection,rough line rule,support vector machine,svm},
number = {11},
pages = {885--896},
title = {{Novel linear search for support vector machine parameter selection}},
volume = {12},
year = {2011}
}
@inproceedings{Quinlan1993,
author = {Quinlan, J R},
booktitle = {Proceedings of the Tenth International Conference of Machine Learning},
pages = {236--243},
publisher = {Morgan Kaufmann},
title = {{Combining Instance-Based and Model-Based Learning}},
year = {1993}
}
@article{Rognvaldsson2014,
author = {Rognvaldsson, T. and You, L. and Garwicz, D.},
file = {:home/adrian/Documents/Mendeley Desktop/Rognvaldsson, You, Garwicz - 2014 - State of the art prediction of HIV-1 protease cleavage sites.pdf:pdf},
journal = {Bioinformatics},
number = {8},
pages = {1204--1210},
title = {{State of the art prediction of HIV-1 protease cleavage sites}},
volume = {31},
year = {2014}
}
@article{Rubio2011,
abstract = {Least Squares Support Vector Machines (LS-SVM) are the state of the art in kernel methods for regression. These models have been successfully applied for time series modelling and prediction. A critical issue for the performance of these models is the choice of the kernel parameters and the hyperparameters which define the function to be minimized. In this paper a heuristic method for setting both the $\sigma$ parameter of the Gaussian kernel and the regularization hyperparameter based on information extracted from the time series to be modelled is presented and evaluated. © 2010 International Institute of Forecasters.},
author = {Rubio, Gin{\'{e}}s and Pomares, H{\'{e}}ctor and Rojas, Ignacio and Herrera, Luis Javier},
doi = {10.1016/j.ijforecast.2010.02.007},
file = {:home/adrian/Documents/Mendeley Desktop/Rubio et al. - 2011 - A heuristic method for parameter selection in LS-SVM Application to time series prediction.pdf:pdf},
isbn = {01692070},
issn = {01692070},
journal = {International Journal of Forecasting},
keywords = {Gaussian kernel parameters,Hyperparameters optimization,Least squares support vector machines,Time series prediction},
number = {3},
pages = {725--739},
title = {{A heuristic method for parameter selection in LS-SVM: Application to time series prediction}},
volume = {27},
year = {2011}
}
@article{Ruppert1995,
author = {Ruppert, D. and Sheather, S. J. and Wand, M. P.},
journal = {Journal of the American Statistical Association},
keywords = {boundary effects,kernel estimator,local polynomial fitting,nonparametric regression,pilot estimation},
number = {432},
pages = {1257--1270},
title = {{An Effective Bandwidth Selector for Local Least Squares Regression}},
volume = {90},
year = {1995}
}
@misc{Srebro,
author = {Srebro, Nathan and Roweis, Sam},
number = {c},
title = {{Adaptive Gaussian Kernel SVMs}}
}
@book{Vapnik1995,
address = {Berlin},
author = {Vapnik, V.},
publisher = {Springer},
title = {{The nature of statistical learning theory}},
year = {1995}
}
@article{Wang2010,
abstract = {Feature selection aims at determining a subset of available features which is most discriminative and informative for data analysis. This paper presents an effective feature selection method for support vector machine (SVM). Unlike the traditional combinatorial searching method, feature selection is translated into the model selection of SVM which has been well studied. In more detail, the basic idea of this method is to tune the hyperparameters of the Gaussian Automatic Relevance Determination (ARD) kernels via optimization of kernel polarization, and then to rank all features in decreasing order of importance so that more relevant features can be identified. We test the proposed method with some UCI machine learning benchmark examples and show that it can dramatically reduce the number of features and outperforms SVM trained using the features selected according to correlation coefficient and using all features. © 2010 Elsevier Ltd. All rights reserved.},
author = {Wang, Tinghua and Huang, Houkuan and Tian, Shengfeng and Xu, Jianfeng},
doi = {10.1016/j.eswa.2010.03.054},
file = {:home/adrian/Documents/Mendeley Desktop/Wang et al. - 2010 - Feature selection for SVM via optimization of kernel polarization with Gaussian ARD kernels.pdf:pdf},
issn = {09574174},
journal = {Expert Systems with Applications},
keywords = {Classification,Feature selection,Kernel polarization,Model selection,Support vector machine (SVM)},
number = {9},
pages = {6663--6668},
publisher = {Elsevier Ltd},
title = {{Feature selection for SVM via optimization of kernel polarization with Gaussian ARD kernels}},
url = {http://dx.doi.org/10.1016/j.eswa.2010.03.054},
volume = {37},
year = {2010}
}
@article{Xu2009,
author = {Xu, Zongben and Dai, Mingwei and Meng, Deyu},
file = {:home/adrian/Documents/Mendeley Desktop/Xu, Dai, Meng - 2009 - Fast and Efficient Strategies for Model Selection of Gaussian Support Vector Machine.pdf:pdf},
journal = {International Journal of Forecasting},
number = {5},
pages = {1292--1307},
title = {{Fast and Efficient Strategies for Model Selection of Gaussian Support Vector Machine}},
volume = {39},
year = {2009}
}
@article{You2005,
abstract = {Rapidly developing viral resistance to licensed human immunodeficiency virus type 1 (HIV-1) protease inhibitors is an increasing problem in the treatment of HIV-infected individuals and AIDS patients. A rational design of more effective protease inhibitors and discovery of potential biological substrates for the HIV-1 protease require accurate models for protease cleavage specificity. In this study, several popular bioinformatic machine learning methods, including support vector machines and artificial neural networks, were used to analyze the specificity of the HIV-1 protease. A new, extensive data set (746 peptides that have been experi- mentally tested for cleavage by the HIV-1 protease) was compiled, and the data were used to construct different classifiers that predicted whether the protease would cleave a given peptide substrate or not. The best predictor was a nonlinear predictor using two physicochemical parameters (hydrophobicity, or alternatively polarity, and size) for the amino acids, indicating that these properties are the key features recognized by the HIV-1 protease. The present in silico study provides new and important insights into the workings of the HIV-1 protease at the molecular level, supporting the recent hypothesis that the protease primarily recognizes a conformation rather than a specific amino acid sequence. Furthermore, we demonstrate that the presence of 1 to 2 lysine residues near the cleavage site of octameric peptide substrates seems to prevent cleavage efficiently, suggesting that this positively charged amino acid plays an important role in hindering the activity of the HIV-1 protease.},
author = {You, Liwen and Garwicz, Daniel and R{\"{o}}gnvaldson, Thorsteinn},
file = {:home/adrian/Documents/Mendeley Desktop/You, Garwicz, R{\"{o}}gnvaldson - 2005 - Comprehensive Bioinformatic Analysis of the Specificity of Human Immunodeficiency Virus Type 1 Prote.pdf:pdf},
journal = {Journal of Virology},
number = {19},
pages = {12477--12486},
title = {{Comprehensive Bioinformatic Analysis of the Specificity of Human Immunodeficiency Virus Type 1 Protease}},
volume = {79},
year = {2005}
}
@article{Zhu2001,
author = {Zhu, Ji and Hastie, Trevor},
doi = {10.1198/106186005X25619},
issn = {1061-8600},
journal = {Journal of Computational and Graphical Statistics},
month = {mar},
pages = {1081--1088},
title = {{Kernel Logistic Regression and the Import Vector Machine}},
url = {http://www.tandfonline.com/doi/abs/10.1198/106186005X25619},
year = {2001}
}
@book{Scholkopf2002,
author = {Sch{\"{o}}lkopf, Bernhard and Smola, Alexander J. Smola},
publisher = {MIT Press},
title = {{Learning with Kernels}},
year = {2002}
} -->