# Statistical Tests Reference for Data Science and Machine Learning

## Normality Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **Shapiro-Wilk Test** | Tests if data follows normal distribution<br>H₀: Data is normally distributed<br>H₁: Data is not normally distributed | • Sample size: 3 ≤ n ≤ 5000<br>• Continuous data<br>• Independent observations | • Checking normality assumption before parametric tests<br>• Best for small samples | Testing if model residuals are normally distributed before linear regression |
| **Kolmogorov-Smirnov Test** | Tests if data follows a specified distribution<br>H₀: Data follows specified distribution<br>H₁: Data does not follow specified distribution | • Continuous data<br>• Large sample size (n > 50)<br>• Known distribution parameters | • Comparing sample to any continuous distribution<br>• Large datasets | Verifying if feature distributions match between train and test sets |
| **Anderson-Darling Test** | Tests if data comes from specified distribution<br>H₀: Data comes from specified distribution<br>H₁: Data does not come from specified distribution | • Continuous data<br>• More weight on tails than K-S<br>• Sample size > 5 | • More sensitive to deviations in tails<br>• Quality control applications | Checking if log-transformed features follow normal distribution |
| **D'Agostino-Pearson Test** | Omnibus test for normality using skewness and kurtosis<br>H₀: Data is normally distributed<br>H₁: Data is not normally distributed | • Sample size n ≥ 20<br>• Tests both skewness and kurtosis<br>• Continuous data | • When you need to understand why normality fails<br>• Medium to large samples | Assessing normality of neural network weight distributions |
| **Q-Q Plot** | Visual assessment of normality<br>Not a formal test but diagnostic tool | • Any sample size<br>• Visual interpretation<br>• Shows where deviations occur | • Initial exploration<br>• Understanding nature of non-normality | Diagnosing distribution of prediction errors in regression models |

## Two-Sample Comparison Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **Independent t-test** | Compares means of two independent groups<br>H₀: μ₁ = μ₂<br>H₁: μ₁ ≠ μ₂ | • Normal distribution<br>• Independent samples<br>• Equal variances (classic)<br>• Continuous data | • A/B testing<br>• Comparing two algorithms<br>• Treatment vs control | Comparing accuracy of two different models on same dataset |
| **Welch's t-test** | Compares means when variances unequal<br>H₀: μ₁ = μ₂<br>H₁: μ₁ ≠ μ₂ | • Normal distribution<br>• Independent samples<br>• Unequal variances allowed<br>• Continuous data | • Default choice over Student's t-test<br>• Robust to unequal variances | Comparing performance metrics between models trained on different data sizes |
| **Paired t-test** | Compares means of paired/matched samples<br>H₀: μd = 0 (mean difference)<br>H₁: μd ≠ 0 | • Paired observations<br>• Differences normally distributed<br>• Continuous data | • Before/after measurements<br>• Matched pairs<br>• Same subjects tested twice | Comparing model performance before and after feature engineering on same test set |
| **Mann-Whitney U Test** | Non-parametric alternative to independent t-test<br>H₀: Distributions are identical<br>H₁: Distributions differ | • Independent samples<br>• Ordinal or continuous data<br>• No normality assumption | • Non-normal data<br>• Ordinal data<br>• Robust to outliers | Comparing median inference times between CPU and GPU deployments |
| **Wilcoxon Signed-Rank Test** | Non-parametric alternative to paired t-test<br>H₀: Median difference = 0<br>H₁: Median difference ≠ 0 | • Paired observations<br>• Continuous or ordinal data<br>• Symmetric distribution of differences | • Non-normal paired data<br>• Small sample sizes | Comparing ranking algorithms on same set of queries |

## Multiple Group Comparison Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **One-way ANOVA** | Compares means across 3+ groups<br>H₀: μ₁ = μ₂ = ... = μₖ<br>H₁: At least one mean differs | • Normal distribution within groups<br>• Independent observations<br>• Equal variances<br>• Continuous dependent variable | • Comparing multiple algorithms<br>• Testing multiple treatments<br>• Factor has 3+ levels | Comparing accuracy across multiple classification algorithms |
| **Two-way ANOVA** | Tests main effects and interaction of two factors<br>H₀: No main effects or interaction<br>H₁: Significant effects exist | • Normal distribution<br>• Independent observations<br>• Equal variances<br>• Balanced design preferred | • Two categorical factors<br>• Testing interactions<br>• Factorial experiments | Testing effects of algorithm type and dataset size on performance |
| **Repeated Measures ANOVA** | Compares means when same subjects measured multiple times<br>H₀: μ₁ = μ₂ = ... = μₖ<br>H₁: At least one mean differs | • Sphericity assumption<br>• Normal distribution<br>• Same subjects across conditions | • Time series measurements<br>• Within-subject designs<br>• Longitudinal studies | Comparing model performance across multiple epochs during training |
| **Kruskal-Wallis Test** | Non-parametric alternative to one-way ANOVA<br>H₀: All groups have same distribution<br>H₁: At least one distribution differs | • Independent samples<br>• Ordinal or continuous data<br>• 3+ groups | • Non-normal data<br>• Ordinal responses<br>• Robust analysis | Comparing median latencies across multiple model architectures |
| **Friedman Test** | Non-parametric alternative to repeated measures ANOVA<br>H₀: All treatments have same effect<br>H₁: At least one treatment differs | • Related samples<br>• 3+ measurements<br>• Ordinal or continuous data | • Non-normal repeated measures<br>• Ranking data<br>• Blocked designs | Comparing multiple algorithms across different datasets |

## Correlation and Association Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **Pearson Correlation** | Measures linear correlation<br>H₀: ρ = 0 (no linear correlation)<br>H₁: ρ ≠ 0 | • Continuous variables<br>• Linear relationship<br>• Bivariate normal distribution<br>• No outliers | • Feature correlation analysis<br>• Linear relationships<br>• Continuous variables | Identifying multicollinearity between features before regression |
| **Spearman Correlation** | Measures monotonic correlation<br>H₀: ρₛ = 0 (no monotonic correlation)<br>H₁: ρₛ ≠ 0 | • Ordinal or continuous data<br>• Monotonic relationship<br>• No normality required | • Non-linear monotonic relationships<br>• Ordinal data<br>• Robust to outliers | Correlating feature importance rankings from different methods |
| **Kendall's Tau** | Measures ordinal association<br>H₀: τ = 0 (no association)<br>H₁: τ ≠ 0 | • Ordinal or continuous data<br>• Smaller sample sizes<br>• Many tied values | • Ranking correlation<br>• Small samples<br>• Many ties | Comparing rankings of feature importance across models |
| **Point-Biserial Correlation** | Correlation between binary and continuous variable<br>H₀: rpb = 0<br>H₁: rpb ≠ 0 | • One binary, one continuous variable<br>• Continuous variable normally distributed | • Feature-target correlation in binary classification<br>• Validating binary features | Measuring correlation between binary feature and continuous target |
| **Chi-Square Test of Independence** | Tests association between categorical variables<br>H₀: Variables are independent<br>H₁: Variables are associated | • Categorical variables<br>• Expected frequencies ≥ 5<br>• Independent observations | • Feature selection for categorical data<br>• Testing dependencies | Testing if categorical features are independent before naive Bayes |
| **Fisher's Exact Test** | Tests association for 2×2 tables with small samples<br>H₀: Variables are independent<br>H₁: Variables are associated | • 2×2 contingency table<br>• Small sample sizes<br>• Exact test | • Small sample categorical analysis<br>• When chi-square assumptions fail | Testing association between binary model predictions and rare events |

## Variance and Distribution Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **F-test** | Compares variances of two groups<br>H₀: σ₁² = σ₂²<br>H₁: σ₁² ≠ σ₂² | • Normal distributions<br>• Independent samples<br>• Sensitive to non-normality | • Testing homogeneity of variance<br>• Before t-tests<br>• Quality control | Comparing variance in predictions between two models |
| **Levene's Test** | Tests equality of variances across groups<br>H₀: All variances equal<br>H₁: At least one variance differs | • Less sensitive to non-normality<br>• 2+ groups<br>• Continuous data | • Robust alternative to F-test<br>• Before ANOVA<br>• Multiple groups | Testing homoscedasticity assumption in regression residuals |
| **Bartlett's Test** | Tests equality of variances (multiple groups)<br>H₀: All variances equal<br>H₁: At least one variance differs | • Normal distributions<br>• Very sensitive to non-normality<br>• 2+ groups | • When normality is certain<br>• Multiple group comparison | Comparing variability in ensemble model predictions |
| **Brown-Forsythe Test** | Robust test for equality of variances<br>H₀: All variances equal<br>H₁: At least one variance differs | • Robust to non-normality<br>• Uses median instead of mean<br>• 2+ groups | • Non-normal data<br>• Robust analysis needed | Testing if different preprocessing methods affect prediction variance |
| **Fligner-Killeen Test** | Non-parametric test for homogeneity of variances<br>H₀: All variances equal<br>H₁: At least one variance differs | • Non-parametric<br>• Very robust<br>• 2+ groups | • Highly non-normal data<br>• Ordinal scale possible | Comparing spread of errors across different data segments |

## Goodness of Fit Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **Chi-Square Goodness of Fit** | Tests if data follows expected distribution<br>H₀: Data follows expected distribution<br>H₁: Data does not follow expected distribution | • Categorical data<br>• Expected frequencies ≥ 5<br>• Independent observations | • Testing distributional assumptions<br>• Categorical outcomes<br>• Model validation | Testing if predicted class proportions match expected distribution |
| **G-test** | Alternative to chi-square goodness of fit<br>H₀: Observed = Expected<br>H₁: Observed ≠ Expected | • Similar to chi-square<br>• Better for small samples<br>• Uses likelihood ratio | • Small sample sizes<br>• More accurate p-values | Comparing observed vs expected feature value distributions |
| **Hosmer-Lemeshow Test** | Tests calibration of logistic regression<br>H₀: Model is well-calibrated<br>H₁: Model is not well-calibrated | • Binary outcomes<br>• Groups by predicted probabilities<br>• Large samples | • Logistic regression diagnostics<br>• Model calibration | Testing if predicted probabilities match observed frequencies |
| **Cramér-von Mises Test** | Tests if data comes from specified distribution<br>H₀: Data follows specified distribution<br>H₁: Data does not follow | • Continuous data<br>• Known distribution parameters<br>• All parts of distribution weighted equally | • Alternative to K-S test<br>• Emphasis on whole distribution | Testing if residuals follow assumed error distribution |

## Time Series and Sequential Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **Augmented Dickey-Fuller Test** | Tests for unit root (stationarity)<br>H₀: Unit root present (non-stationary)<br>H₁: No unit root (stationary) | • Time series data<br>• No missing values<br>• Regular intervals | • Testing stationarity<br>• Before ARIMA modeling<br>• Feature engineering | Testing if time series features are stationary before modeling |
| **KPSS Test** | Tests for stationarity (opposite null of ADF)<br>H₀: Data is stationary<br>H₁: Data has unit root | • Time series data<br>• Complements ADF test<br>• Different null hypothesis | • Confirming stationarity<br>• Used with ADF test | Double-checking stationarity of detrended features |
| **Ljung-Box Test** | Tests for autocorrelation in residuals<br>H₀: No autocorrelation<br>H₁: Autocorrelation present | • Time series residuals<br>• Tests multiple lags jointly<br>• After model fitting | • Model diagnostic<br>• Residual analysis<br>• ARIMA validation | Testing if model residuals are white noise |
| **Durbin-Watson Test** | Tests for first-order autocorrelation<br>H₀: No autocorrelation<br>H₁: Positive or negative autocorrelation | • Regression residuals<br>• Time-ordered data<br>• First-order only | • Regression diagnostics<br>• Time series regression | Detecting autocorrelation in linear model residuals |
| **Run Test** | Tests for randomness in sequence<br>H₀: Sequence is random<br>H₁: Sequence is not random | • Binary or continuous data<br>• Sequential observations<br>• Non-parametric | • Testing randomness<br>• Model residuals<br>• Pattern detection | Testing if model errors show systematic patterns |

## Machine Learning Specific Tests

| Test Name | Purpose & Hypothesis | Assumptions & Requirements | When to Use | ML/DS Example |
|-----------|---------------------|---------------------------|-------------|---------------|
| **McNemar's Test** | Compares paired binary classifications<br>H₀: No difference in errors<br>H₁: Significant difference in errors | • Paired binary outcomes<br>• 2×2 contingency table<br>• Same test set | • Comparing classifiers<br>• Paired predictions<br>• Model selection | Comparing two classifiers on same test set |
| **5x2cv Paired t-test** | Compares ML algorithms with cross-validation<br>H₀: No performance difference<br>H₁: Significant difference | • 5 iterations of 2-fold CV<br>• Reduces Type I error<br>• Continuous metrics | • Robust algorithm comparison<br>• Small datasets<br>• Conservative test | Comparing deep learning vs traditional ML on small dataset |
| **Cochran's Q Test** | Compares multiple classifiers on binary outcomes<br>H₀: All classifiers perform equally<br>H₁: At least one differs | • 3+ classifiers<br>• Binary outcomes<br>• Same test set | • Multiple classifier comparison<br>• Binary problems<br>• Initial screening | Comparing multiple binary classifiers simultaneously |
| **DeLong's Test** | Compares ROC AUC scores<br>H₀: AUC₁ = AUC₂<br>H₁: AUC₁ ≠ AUC₂ | • Binary classification<br>• Same test set<br>• Correlated ROC curves | • Comparing model discrimination<br>• Medical diagnostics<br>• Ranking models | Testing if one model has significantly better AUC than another |
| **Permutation Test** | Non-parametric test for any statistic<br>H₀: No difference between groups<br>H₁: Significant difference | • Any metric<br>• No distributional assumptions<br>• Computationally intensive | • Custom metrics<br>• Small samples<br>• No assumptions | Testing if custom performance metric differs between models |