

#%%


import polars as pl
import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable("vegafusion")
from pingouin import bayesfactor_ttest
import pingouin as pg
from scipy import stats
from scipy.spatial.distance import mahalanobis
from itertools import combinations
import numpy as np
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu


#%%

####Task 1

##Load data

df = pl.read_csv("Baqend_Dummy_Data_Set.csv")


##split descriptives

#%%

#Descriptives and inference

def inference_factors(vars_to_compare):

    vars_to_compare = vars_to_compare

    for var in vars_to_compare:

        counts = (

        df.group_by(["testGroup", var])

        .len()

        .rename({"len": "count"})

        .sort("testGroup")

        )

        contingency = counts.pivot(

            values="count", 

            index=var, 

            columns="testGroup"

        ).fill_null(0)
        
        print(f"\nDescriptive Summary for {var}:")

        print(contingency)

        image = counts.plot.bar(

            x=var,

            y="count",

            color="testGroup"

        )

        image.show()

        plot_path = f"plots/{var}_bar.png"

        #image.save(plot_path)

        chi2_table = contingency.select(pl.exclude(var)).to_numpy()

        chi2, p, dof, expected = stats.chi2_contingency(chi2_table)

        print(f"\nChi-Square Test for {var}:")

        print(f"Chi2 = {chi2:.3f}, df = {dof}, p = {p:.4f}")

    """
        html_output += f"<h2>Descriptive Summary for {var}</h2>"

        html_output += contingency.to_pandas().to_html(index=False)

        html_output += f'<img src="{plot_path}" width="600"><br>'
    """


###Continious



def inference_continious(outlier : bool, vars_to_compare: list):

    es_mwu = []

    vars_to_compare = vars_to_compare

    for var in vars_to_compare:


        #raw data

        values = df[var].to_numpy()

        plt.figure(figsize=(8, 4))

        plt.hist(values, bins=100, alpha=0.5)

        plt.xlabel(var)

        plt.ylabel("Frequency")

        plt.title(f"Distribution of {var}")

        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        plt.show()


        if outlier == True:

            q1 = df.select(pl.col(var).quantile(0.25, "nearest")).item()

            q3 = df.select(pl.col(var).quantile(0.75, "nearest")).item()

            iqr = q3 - q1

            lower = q1 - 3 * iqr

            upper = q3 + 3 * iqr

            df_filtered = df.filter(pl.col(var).is_between(lower, upper))

        else:
             
             df_filtered = df


        summary = (

            df_filtered.group_by("testGroup")

            .agg([

                pl.col(var).mean().alias("mean"),

                pl.col(var).std().alias("std"),

                pl.col(var).min().alias("min"),

                pl.col(var).max().alias("max"),

                pl.col(var).median().alias("median"),

                pl.col(var).count().alias("n")

            ])

            .sort("testGroup")

        )

        print(f"\nDescriptive Summary for {var}:")

        print(summary)

        test_groups = df["testGroup"].unique().to_list()

        data = [

            df_filtered.filter(pl.col("testGroup") == group)[var].to_numpy()

            for group in test_groups

        ]

        plt.figure(figsize=(8, 4))

        plt.boxplot(

            data,

            labels=test_groups,

            notch=True,

            patch_artist=True,

            widths=0.5,

            vert=True

        )


        plt.title(f"{var} distribution by testGroup")

        plt.xlabel("Test Group")

        plt.ylabel(var)

        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        plt.show()




        values_by_group = [

            df_filtered.filter(pl.col("testGroup") == group)[var].to_numpy()

            for group in test_groups

        ]

        plt.figure(figsize=(8, 4))

        plt.hist(

            values_by_group[0], bins=20, alpha=0.5, label=test_groups[0], density=False

        )

        plt.hist(

            values_by_group[1], bins=20, alpha=0.5, label=test_groups[1], density=False
        )

        plt.title(f"{var} distribution by testGroup (Histogram)")

        plt.xlabel(var)

        plt.ylabel("Frequency")

        plt.legend()

        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        plt.show()


        t_stat, p_val = stats.ttest_ind(

            values_by_group[0],

            values_by_group[1],

            equal_var=False  
        )

        print(f"\nT-Test for {var} between {test_groups[0]} and {test_groups[1]}:")

        print(f"t = {t_stat:.3f}, p = {p_val:.4f}")

        n1 = len(values_by_group[0])

        n2 = len(values_by_group[1])

        t = float(f"{t_stat:.3f}")

        bf = bayesfactor_ttest(t, n1, n2)

        print("Bayes Factor: %.3f (two-sample independent)" % bf)



        u_stat, p_value = mannwhitneyu(values_by_group[0], values_by_group[1], alternative='two-sided')

        print("UTest: ", u_stat, "p_value: ", p_value)

        N = n1 + n2

        mean_U = n1 * n2 / 2

        std_U = np.sqrt(n1 * n2 * (N + 1) / 12)

        z = (u_stat - mean_U) / std_U

        r = abs(z) / np.sqrt(N)

        es_mwu.append(r)

    return es_mwu





#%%


inference_factors(["device", "browser", "browserVersion", "userAgent"])

results = inference_continious(False, [

    "users", "newUsers", "sessions", "pis", "acceleratedPis",

    "addToCartEvents", "creditcardPaymentEvent", "paypalPaymentEvent",

    "applepayPaymentEvent", "conversions"

])

#%%

#we now test distributins of effect sizes againt 0 baysian

data = [float(x) for x in results]

pg.ttest(data, 0).round(2)


plt.hist(data, bins=100, color="#0568FD", edgecolor="black")

# Add labels and title
plt.title("Effect size distribution KPI's")
plt.xlabel("Effect Size")
plt.ylabel("Freq")

# Show plot
plt.show()



#%%

#POI:

"""
Very odd distributions
valid data? 1000s from one ip?
Is index timestamps?




"""


#%%

##Further analysis: look into outliers


q1 = df.select(pl.col("users").quantile(0.25, "nearest")).item()

q3 = df.select(pl.col("users").quantile(0.75, "nearest")).item()

iqr = q3 - q1

lower = q1 - 3 * iqr

upper = q3 + 3 * iqr

df_filtered = df.filter(~pl.col("users").is_between(lower, upper)).sort("users", descending=True)


values = df_filtered["users"].to_numpy()

plt.figure(figsize=(8, 4))

plt.hist(values, bins=100, alpha=0.5)

plt.xlabel("users")

plt.ylabel("Frequency")

plt.title(f"Distribution of {"users"}")

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.show()



##Ratios 

df = df.with_columns([

    ((pl.col("users") / (pl.col("sessions") + pl.col("users")))*100).alias("ratio_user_per_session"),

    ((pl.col("pis") / (pl.col("newUsers") + pl.col("pis")))*100).alias("ratio_pis_per_newUser"),

    ((pl.col("addToCartEvents") / (pl.col("newUsers") + pl.col("addToCartEvents")))*100).alias("ratio_addToCart_per_newUser"),

    ((pl.col("conversions") / (pl.col("creditcardPaymentEvent") + pl.col("conversions")))*100).alias("ratio_conversion_per_creditcard"),

    ((pl.col("conversions") / (pl.col("conversions") + pl.col("newUsers")))*100).alias("ratio_newUsers_conversion")

])

df_filter = df.filter(pl.col("ratio_user_per_session").is_finite()).sort("sessions", descending=True)

df_filter.head(10)

#247	"SpeedKit"	"Desktop"	"Chrome"	"101.0.4951.64"	"Mozilla/5.0 (X11; Linux x86_64…	1	0	32160	32160	???

#%%

###further pattern inspection corr mahalanobis distance for outliers


cols = [

    "users", "newUsers", "sessions", "pis", "acceleratedPis",

    "addToCartEvents", "creditcardPaymentEvent", "paypalPaymentEvent",

    "applepayPaymentEvent", "conversions"

]

df_numeric = df.select(cols)

df_np = df_numeric.to_numpy()

inv_cov = np.linalg.inv(np.cov(df_np, rowvar=False))

mean_vec = df_np.mean(axis=0)



mahal_dists = np.array([

    mahalanobis(row, mean_vec, inv_cov)

    for row in df_np

])


threshold = np.percentile(mahal_dists, 99.9)

outliers = mahal_dists > threshold


for col_x, col_y in combinations(cols, 2):

    x = df[col_x].to_numpy()

    y = df[col_y].to_numpy()

    plt.figure(figsize=(6, 4))

    plt.scatter(x[~outliers], y[~outliers], alpha=0.5, label="Normal")

    plt.scatter(x[outliers], y[outliers], color="red", label="Outlier", marker="x")

    plt.xlabel(col_x)

    plt.ylabel(col_y)

    plt.title(f"{col_x} vs. {col_y} (Mahalanobis Outliers)")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.show()

#->10000 new users, no conversion?
#->50k new users, 0 apple payment?
#->32k sessions, no conversions(as above)



df.filter(pl.col("newUsers").is_between(10000,15000))


#5169	"SpeedKit"	"Robot"	"Chrome"	"106"	"Mozilla/5.0 (Windows NT 10.0; …	11452	11452	11452	11498	33	2	0	0	0	0	50.0	50.100218	0.017461	NaN	0.0



df.filter((pl.col("newUsers")>10 ) & (pl.col("conversions") == 0 ))

#0	"Control"	"Robot"	"Chrome"	"92.0.4512.0"	"Mozilla/5.0 (Windows NT 10.0; …	315	314	315	315	0	0	0	0	0	0	50.0	50.079491	0.0	NaN	0.0

#some hundred users but no interaction


#%%

##we test also device and browser

for factor in ["device", "browser"]:

    for col in cols:

        df_pandas = df.to_pandas().dropna()

        summary_stats = df_pandas.groupby([factor])[col].agg(["mean", "median", "std", "count"]).reset_index()
        
        print(f"\nSummary statistics for {col} by {factor}:\n", summary_stats)

        aov = pg.anova(dv=col, between= factor, data=df_pandas ,detailed=True)

        print(f"\nANOVA for {col} by {factor}:\n", aov)
        
        tukey_res = pairwise_tukeyhsd(endog=df_pandas[col], groups=df_pandas[factor], alpha=0.05)

        print(f"\nTukey HSD for {col} by {factor}:\n", tukey_res.summary())
        
        plt.figure(figsize=(10, 6))

        sns.lineplot(data=df_pandas, x=factor, y=col, estimator="mean", ci="sd", marker="o")

        plt.title(f"{col} by {factor} ")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()

#->Tablet + Robot very few data

#->Edge + Opera no apple payment despite 76 trials?


#Task 2 basic
# %%
"""

-went to devtools, strg f looked for main and went down the path, the element is the next arrow in the image gallery
-hypotheses:

Unresponsive Clicks: Users tapped the arrow, but no immediate feedback or image change occurred—leading to repeated (rage) taps.

->Small Tap Area: The clickable area of the arrow was likely too small ->click in image leeds to new area with click and back button

Overlapping Layers: A transparent or misaligned overlay (e.g., zoom, modal, or animation placeholder) could have blocked the clicks without visible indication.

Touch Event Conflicts: Conflicting touchstart or click listeners may have caused inconsistent behavior on mobile browsers.

No Visual Feedback: Lack of visual response (like a highlight or slide animation start) reinforced the feeling that “nothing happened.”

but On my devise everything went fine

"""

#Task 2 advanced

"""

LCP mobile 2.52 sec vs web 1.50

->devtools perormance record

<a href="https://www.obi.de/marken/denpanels" class="disc-link ...">
  <span class="tw-relative ...">
    <span class="disc-headline ...">Zum Denpanels Markenshop</span>
  </span>
</a>


this might be elemnt for lcp

1. Product Image (LCP candidate on desktop) is Lazy-Loaded and Below the Fold on Mobile

On desktop, the large <img> (product image) appears high in the viewport, is not lazy-loaded, and is painted early, making it the LCP candidate.

On mobile, the same <img>:

Has the attribute: loading="lazy" ✅

Is pushed below the fold due to mobile layout grid (.pdp-grid-layout)

Therefore, not painted early, and ineligible as LCP

Result: The browser falls back to a text element like the "Zum Denpanels Markenshop" link — which:

Depends on web fonts

Paints later

Is less visually prominent

2. Text LCP Fallback is Font-Blocked

Mobile LCP element uses:

<span class="disc-headline tw-font-obi-bold ...">

  Zum Denpanels Markenshop

</span>

This font (tw-font-obi-bold) is a custom web font, which:

Delays rendering until the font is downloaded and applied

Often triggers a Flash of Invisible Text (FOIT)

That adds critical milliseconds to LCP on mobile.

"""
# %%
