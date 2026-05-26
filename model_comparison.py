import matplotlib.pyplot as plt

models = [

"Speech",

"Text",

"Fusion"

]

accuracy = [

100,

99,

100

]

colors = [

"#AE2448",

"#FFCEE3",

"#E6F082"
]

plt.figure(

figsize=(10,6),

facecolor="#FFF7FB"
)

bars = plt.bar(

models,

accuracy,

color=colors,

width=0.6
)

plt.title(

"Accuracy Comparison Across Models",

fontsize=20,

fontweight="bold",

color="#7A1833"
)

plt.xlabel(

"Model Type",

fontsize=14,

fontweight="bold",

color="#7A1833"
)

plt.ylabel(

"Accuracy (%)",

fontsize=14,

fontweight="bold",

color="#7A1833"
)

plt.ylim(

95,

102
)

plt.grid(

axis="y",

linestyle="--",

alpha=0.3
)

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x()+0.26,

        height+0.15,

        f"{height}%",

        fontsize=12,

        fontweight="bold",

        color="#AE2448"
    )

ax = plt.gca()

ax.set_facecolor("#FFFDF8")

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#AE2448")

ax.spines["bottom"].set_color("#AE2448")

plt.savefig(

"Results/plots/model_comparison.png",

bbox_inches="tight",

dpi=300
)

plt.show()

print(
"\n comparison graph generated "
)