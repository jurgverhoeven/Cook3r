from Food_recognition import Food_recognition
import Pan
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import Food
import os
from Fetch_data import fetch_data

food_path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans"


if __name__ == "__main__":

    data_path = 'C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Machine Learning report/Masked_dataset'

    # Fetch the data
    foods = fetch_data(food_path)

    # encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(foods.target)

    print(coded_labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(foods.data, coded_labels,
	test_size=0.20, stratify=foods.target)

    # Data preparation (note that a pipeline  would help here)
    # trainX = StandardScaler().fit_transform(trainX)

    # show target distribution
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticklabels(foods.unique_targets)
    ax.set_title(data_path + ' count')
    plt.tight_layout()

    
    plt.savefig('distribution.svg')

    # show histograms of first 4 features
    fig0, ax0 = plt.subplots(2, 2)
    sns.histplot(trainX[:,0], color="skyblue", bins=10, ax=ax0[0,0])
    sns.histplot(trainX[:,1], color="olive", bins=10, ax=ax0[0,1])#, axlabel=gestures.feature_names[1])
    sns.histplot(trainX[:,2], color="gold", bins=10, ax=ax0[1,0])#, axlabel=gestures.feature_names[2])
    sns.histplot(trainX[:,3], color="teal", bins=10, ax=ax0[1,1])#, axlabel=gestures.feature_names[3])
    ax0[0,0].set_xlabel(foods.feature_names[0])
    ax0[0,1].set_xlabel(foods.feature_names[1])
    ax0[1,0].set_xlabel(foods.feature_names[2])
    ax0[1,1].set_xlabel(foods.feature_names[3])
    plt.tight_layout()
    
    plt.savefig('histogram.svg')

    # show scatter plot of features a and b
    a, b = 0, 1
    fig1 = plt.figure()
    ax1 = sns.scatterplot(trainX[:,a], trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Scatter plot Area Perimeter")
    ax1.set_xlabel(foods.feature_names[a])
    ax1.set_ylabel(foods.feature_names[b])
    plt.tight_layout()
    
    plt.savefig('scatter_plot_area_perimeter.svg')

    # show scatter plot of features a and b
    a, b = 0, 2
    fig1 = plt.figure()
    ax1 = sns.scatterplot(trainX[:,a], trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Scatter plot Area prominent_hue")
    ax1.set_xlabel(foods.feature_names[a])
    ax1.set_ylabel(foods.feature_names[b])
    plt.tight_layout()
    
    plt.savefig('scatter_plot_area_prominent_hue.svg')

    # show scatter plot of features a and b
    a, b = 0, 3
    fig1 = plt.figure()
    ax1 = sns.scatterplot(trainX[:,a], trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Scatter plot Area shape")
    ax1.set_xlabel(foods.feature_names[a])
    ax1.set_ylabel(foods.feature_names[b])
    plt.tight_layout()
    
    plt.savefig('scatter_plot_area_shape.svg')

    # show scatter plot of features a and b
    a, b = 1, 2
    fig1 = plt.figure()
    ax1 = sns.scatterplot(trainX[:,a], trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Scatter plot Perimeter prominent_hue")
    ax1.set_xlabel(foods.feature_names[a])
    ax1.set_ylabel(foods.feature_names[b])
    plt.tight_layout()
    
    plt.savefig('scatter_plot_perimeter_prominent_hue.svg')

    # show scatter plot of features a and b
    a, b = 1, 3
    fig1 = plt.figure()
    ax1 = sns.scatterplot(trainX[:,a], trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Scatter plot Perimeter shape")
    ax1.set_xlabel(foods.feature_names[a])
    ax1.set_ylabel(foods.feature_names[b])
    plt.tight_layout()
    
    plt.savefig('scatter_plot_perimeter_shape.svg')

    # show scatter plot of features a and b
    a, b = 2, 3
    fig1 = plt.figure()
    ax1 = sns.scatterplot(trainX[:,a], trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Scatter plot Prominent_hue shape")
    ax1.set_xlabel(foods.feature_names[a])
    ax1.set_ylabel(foods.feature_names[b])
    plt.tight_layout()
    
    plt.savefig('scatter_plot_prominent_hue_shape.svg')

    # show boxplot for a single feature
    a = 0
    plt.figure()
    ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,a])
    ax3.set_title(foods.feature_names[a])
    ax3.set_ylabel(foods.feature_names[a])
    plt.tight_layout()
    
    plt.savefig('boxplot_area.svg')

    # show boxplot for a single feature
    a = 1
    plt.figure()
    ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,a])
    ax3.set_title(foods.feature_names[a])
    ax3.set_ylabel(foods.feature_names[a])
    plt.tight_layout()
    
    plt.savefig('boxplot_perimeter.svg')

    # show boxplot for a single feature
    a = 2
    plt.figure()
    ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,a])
    ax3.set_title(foods.feature_names[a])
    ax3.set_ylabel(foods.feature_names[a])
    plt.tight_layout()
    
    plt.savefig('boxplot_prominent_hue.svg')

    # show boxplot for a single feature
    a = 3
    plt.figure()
    ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,a])
    ax3.set_title(foods.feature_names[a])
    ax3.set_ylabel(foods.feature_names[a])
    plt.tight_layout()
    
    plt.savefig('boxplot_shape.svg')

    # show feature correlation heatmap
    plt.figure()
    corr = np.corrcoef(trainX, rowvar=False)
    ax4 = sns.heatmap(corr, annot=True, xticklabels=foods.feature_names, yticklabels=foods.feature_names)
    plt.tight_layout()    
    
    plt.savefig('correlation_heatmap.svg')
    
    plt.show(block=False)





