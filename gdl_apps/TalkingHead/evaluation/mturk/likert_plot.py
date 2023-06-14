# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# def diverging_likert_chart(data, ax=None):
#     """Create a diverging bar chart based on Likert scale data."""
#     # Prepare the data
#     mid = len(data.columns) // 2
#     data['positive'] = data.iloc[:, :mid].sum(axis=1)
#     data['negative'] = -data.iloc[:, mid:].sum(axis=1)
#     data.sort_values(by=['negative', 'positive'], inplace=True)

#     # Create the chart
#     data[['negative', 'positive']].plot(kind='barh', stacked=True, color=['#d65f5f', '#5fba7d'], ax=ax)

# # Example usage:
# likert_scale_items = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
# model1_responses = [1, 2, 3, 2, 1]  # replace with your model's responses
# model2_responses = [2, 3, 3, 2, 1]  # replace with your model's responses

# data = pd.DataFrame({item: [response1, response2] for item, response1, response2 in zip(likert_scale_items, model1_responses, model2_responses)}, index=['Model 1', 'Model 2'])
# diverging_likert_chart(data)
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Use a style and color palette from seaborn
# sns.set_style("whitegrid")

# # Define the Likert scale points and the responses of each model
# likert_points = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
# model1_responses = [10, 20, 30, 25, 15]  # replace with your model's responses
# model2_responses = [15, 25, 25, 20, 15]  # replace with your model's responses

# # Create a DataFrame with the responses
# data = pd.DataFrame({'Model 1': model1_responses, 'Model 2': model2_responses}, index=likert_points)

# # Create the bar chart
# data.plot(kind='bar', color=sns.color_palette("Set2", 2))

# # Set labels and title
# plt.xlabel('Likert Scale Points')
# plt.ylabel('Frequency')
# plt.title('Comparison of Two Models')

# # Set the tick parameters to remove ticks
# plt.tick_params(axis='both', which='both', length=0)

# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Use a style and color palette from seaborn
# sns.set_style("whitegrid")

# # Define the Likert scale points and the responses of each model
# likert_points = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
# model1_responses = [10, 20, 30, 25, 15]  # replace with your model's responses
# model2_responses = [15, 25, 25, 20, 15]  # replace with your model's responses

# # Create a DataFrame with the responses
# data = pd.DataFrame({'Model 1': model1_responses, 'Model 2': model2_responses}, index=likert_points)

# # Create a diverging color palette
# colors = sns.color_palette("coolwarm", 5)

# # Create the bar chart
# data.plot(kind='barh', stacked=True, color=colors)

# # Set labels and title
# plt.xlabel('Frequency')
# plt.ylabel('Likert Scale Points')
# plt.title('Comparison of Two Models')

# # Set the tick parameters to remove ticks
# plt.tick_params(axis='both', which='both', length=0)

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Assume we have 3 models and 5 bins representing a Likert scale.
# models = ["Model 1", "Model 2", "Model 3"]
# likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# # Randomly generated data for the sake of example.
# np.random.seed(0)
# data = np.random.rand(3, 5)

# # Normalization of data so that the sum of each row equals 1.
# data /= data.sum(axis=1, keepdims=True)

# # Split the neutral category into two.
# data[:, 2] /= 2
# data_neg = data[:, :3].copy()
# data_pos = data[:, 2:].copy()

# # The order of the bars does not matter, but for aesthetic reasons we might want to sort them.
# order = np.argsort(-data_neg[:, :2].sum(axis=1))

# fig, ax = plt.subplots()

# # Create bars for the negative side.
# for i in range(2, -1, -1):
#     ax.barh(models, data_neg[order, i], color=plt.get_cmap('RdBu')(i / 4), edgecolor='gray', left=-data_neg[order, :i].sum(axis=1))

# # Create bars for the positive side.
# for i in range(2, 5):
#     ax.barh(models, data_pos[order, i], color=plt.get_cmap('RdBu')(i / 4), edgecolor='gray', left=data_pos[order, :i].sum(axis=1))

# ax.set_xlim([-0.5, 0.5])  # Set x limits.
# ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# # Add legend
# ax.legend(likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Assume we have 3 models and 5 bins representing a Likert scale.
# models = ["Model 1", "Model 2", "Model 3"]
# likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# # Randomly generated data for the sake of example.
# np.random.seed(0)
# data = np.random.rand(3, 5)

# # Normalization of data so that the sum of each row equals 1.
# data /= data.sum(axis=1, keepdims=True)

# # Split the neutral category into two.
# data[:, 2] /= 2
# data_neg = data[:, :3].copy()
# data_pos = data[:, 2:].copy()

# # The order of the bars does not matter, but for aesthetic reasons we might want to sort them.
# order = np.argsort(-data_neg[:, :2].sum(axis=1))

# fig, ax = plt.subplots()

# # Create bars for the negative side.
# for i in range(2, -1, -1):
#     ax.barh(models, data_neg[order, i], color=plt.get_cmap('RdBu')(i / 4), edgecolor='gray', left=-data_neg[order, :i].sum(axis=1))

# # Create bars for the positive side.
# for i in range(0, 3):
#     ax.barh(models, data_pos[order, i], color=plt.get_cmap('RdBu')(i / 4), edgecolor='gray', left=data_pos[order, :i].sum(axis=1))

# ax.set_xlim([-0.5, 0.5])  # Set x limits.
# ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# # Add legend
# ax.legend(likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Assume we have 3 models and 5 bins representing a Likert scale.
# models = ["Model 1", "Model 2", "Model 3"]
# likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# # Randomly generated data for the sake of example.
# np.random.seed(0)
# data = np.random.rand(3, 5)

# # Normalization of data so that the sum of each row equals 1.
# data /= data.sum(axis=1, keepdims=True)

# # Split the neutral category into two.
# data[:, 2] /= 2
# data_neg = data[:, :3].copy()
# data_pos = data[:, 2:].copy()

# # The order of the bars does not matter, but for aesthetic reasons we might want to sort them.
# order = np.argsort(-data_neg[:, :2].sum(axis=1))

# fig, ax = plt.subplots()

# # Create bars for the negative side.
# for i in range(2, -1, -1):
#     ax.barh(models, data_neg[order, i], color=plt.get_cmap('RdBu')(i / 4), edgecolor='gray', left=-data_neg[order, :i].sum(axis=1))

# # Create bars for the positive side.
# for i in range(0, 3):
#     ax.barh(models, data_pos[order, i], color=plt.get_cmap('RdBu')(i / 4 + 0.5), edgecolor='gray', left=data_neg[order, :].sum(axis=1))

# ax.set_xlim([-0.5, 0.5])  # Set x limits.
# ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# # Add legend
# ax.legend(likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Assume we have 3 models and 5 bins representing a Likert scale.
# models = ["Model 1", "Model 2", "Model 3"]
# likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# # Randomly generated data for the sake of example.
# np.random.seed(0)
# data = np.random.rand(3, 5)

# # Normalization of data so that the sum of each row equals 1.
# data /= data.sum(axis=1, keepdims=True)

# # Split the neutral category into two.
# data[:, 2] /= 2

# # Prepare separate data for each category.
# data_strongly_neg = data[:, 0].copy()
# data_weakly_neg = data[:, 1].copy()
# data_neutral = data[:, 2].copy()
# data_weakly_pos = data[:, 3].copy()
# data_strongly_pos = data[:, 4].copy()

# fig, ax = plt.subplots()

# # Create bars for each category.
# ax.barh(models, -data_strongly_neg, color=plt.get_cmap('RdBu')(0), edgecolor='gray')
# ax.barh(models, -data_weakly_neg, color=plt.get_cmap('RdBu')(0.25), edgecolor='gray', left=-data_strongly_neg)
# ax.barh(models, -data_neutral/2, color=plt.get_cmap('RdBu')(0.5), edgecolor='gray', left=-data_strongly_neg-data_weakly_neg)
# ax.barh(models, data_neutral/2, color=plt.get_cmap('RdBu')(0.5), edgecolor='gray', left=data_strongly_neg+data_weakly_neg)
# ax.barh(models, data_weakly_pos, color=plt.get_cmap('RdBu')(0.75), edgecolor='gray', left=data_neutral/2)
# ax.barh(models, data_strongly_pos, color=plt.get_cmap('RdBu')(1), edgecolor='gray', left=data_neutral/2+data_weakly_pos)

# ax.set_xlim([-0.5, 0.5])  # Set x limits.
# ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# # Add legend
# ax.legend(likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Assume we have 3 models and 5 bins representing a Likert scale.
# models = ["Model 1", "Model 2", "Model 3"]
# likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# # Randomly generated data for the sake of example.
# np.random.seed(0)
# data = np.random.rand(3, 5)

# # Normalization of data so that the sum of each row equals 1.
# data /= data.sum(axis=1, keepdims=True)

# # Prepare separate data for each category.
# data_strongly_neg = data[:, 0].copy()
# data_weakly_neg = data[:, 1].copy()
# data_neutral = data[:, 2].copy()
# data_weakly_pos = data[:, 3].copy()
# data_strongly_pos = data[:, 4].copy()

# fig, ax = plt.subplots()

# # Create bars for each category.
# ax.barh(models, -data_strongly_neg, color=plt.get_cmap('RdBu')(0), edgecolor='gray')
# ax.barh(models, -data_weakly_neg, color=plt.get_cmap('RdBu')(0.25), edgecolor='gray', left=-data_strongly_neg)
# ax.barh(models, -data_neutral, color=plt.get_cmap('RdBu')(0.5), edgecolor='gray', left=-data_strongly_neg-data_weakly_neg)
# ax.barh(models, data_weakly_pos, color=plt.get_cmap('RdBu')(0.75), edgecolor='gray', left=0)
# ax.barh(models, data_strongly_pos, color=plt.get_cmap('RdBu')(1), edgecolor='gray', left=data_weakly_pos)

# ax.set_xlim([-0.5, 0.5])  # Set x limits.
# ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# # Add legend
# ax.legend(likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Assume we have 3 models and 5 bins representing a Likert scale.
# models = ["Model 1", "Model 2", "Model 3"]
# likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# # Randomly generated data for the sake of example.
# np.random.seed(0)
# data = np.random.rand(3, 5)

# # Normalization of data so that the sum of each row equals 1.
# data /= data.sum(axis=1, keepdims=True)

# # Prepare separate data for each category.
# data_strongly_neg = data[:, 0].copy()
# data_weakly_neg = data[:, 1].copy()
# data_neutral = data[:, 2].copy()
# data_weakly_pos = data[:, 3].copy()
# data_strongly_pos = data[:, 4].copy()

# fig, ax = plt.subplots()

# # Create bars for each category.
# ax.barh(models, -data_strongly_neg, color=plt.get_cmap('viridis')(0), edgecolor='gray')
# ax.barh(models, -data_weakly_neg, color=plt.get_cmap('viridis')(0.25), edgecolor='gray', left=-data_strongly_neg)
# ax.barh(models, -data_neutral, color=plt.get_cmap('viridis')(0.5), edgecolor='gray', left=-data_strongly_neg-data_weakly_neg)
# ax.barh(models, data_weakly_pos, color=plt.get_cmap('viridis')(0.75), edgecolor='gray', left=0)
# ax.barh(models, data_strongly_pos, color=plt.get_cmap('viridis')(1), edgecolor='gray', left=data_weakly_pos)

# ax.set_xlim([-0.5, 0.5])  # Set x limits.
# ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# # Create legend with matching colors
# handles = [plt.Rectangle((0,0),1,1, color=plt.get_cmap('viridis')(i)) for i in np.linspace(0, 1, 5)]
# ax.legend(handles, likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assume we have 3 models and 5 bins representing a Likert scale.
models = ["Model 1", "Model 2", "Model 3"]
likert_scale = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# Randomly generated data for the sake of example.
np.random.seed(0)
data = np.random.rand(3, 5)

# Normalization of data so that the sum of each row equals 1.
data /= data.sum(axis=1, keepdims=True)

# Prepare separate data for each category.
data_strongly_neg = data[:, 0].copy()
data_weakly_neg = data[:, 1].copy()
data_neutral = data[:, 2].copy()
data_weakly_pos = data[:, 3].copy()
data_strongly_pos = data[:, 4].copy()

fig, ax = plt.subplots()

# Create bars for each category.
ax.barh(models, -data_strongly_neg, color='purple', edgecolor='gray')
ax.barh(models, -data_weakly_neg, color='blue', edgecolor='gray', left=-data_strongly_neg)
ax.barh(models, -data_neutral, color='gray', edgecolor='gray', left=-data_strongly_neg-data_weakly_neg)
ax.barh(models, data_weakly_pos, color='lightgreen', edgecolor='gray', left=0)
ax.barh(models, data_strongly_pos, color='green', edgecolor='gray', left=data_weakly_pos)

ax.set_xlim([-0.5, 0.5])  # Set x limits.
ax.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at x=0.

# Create legend with matching colors
legend_colors = ['purple', 'blue', 'gray', 'lightgreen', 'green']
handles = [plt.Rectangle((0,0),1,1, color=color) for color in legend_colors]
ax.legend(handles, likert_scale, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
