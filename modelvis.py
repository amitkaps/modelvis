"""
Visualising Machine Learning Models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from IPython.display import SVG
import requests

__version__ = "0.1.2"
__author__ = "Amit Kapoor <amitkaps@gmail.com>"

def plot_classifier_2d(clf, data, target, probability = False, alpha = 0.9):
    x_min, x_max = data.iloc[:,0].min(), data.iloc[:,0].max()
    y_min, y_max = data.iloc[:,1].min(), data.iloc[:,1].max()
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, (x_max - x_min)/100),
        np.arange(y_min, y_max, (y_max - y_min)/100))
    if probability == False:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,0]
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap="viridis", alpha = 0.5)
    plt.colorbar(cs)
    plt.scatter(x = data.iloc[:,0], y = data.iloc[:,1], c = target, s = 100, cmap="magma", alpha = alpha)

def render_tree(decision_tree, feature_names=None, class_names=None, **kwargs):
    """Returns a PIL image to visualize given DecisionTree model.

    This exports the decision_tree as graphviz file and sends it to a service that
    renders graphviz as an SVG.

    Sample Usage:

        modelvis.render_tree(
            decision_tree=model,
            class_names=['no', 'yes'],
            feature_names=['age', 'amount', 'ownership'])

    :param decision_tree: a DecisionTree model
    :param feature_names: names of the features used in building the model
    :param class_names: the class labels
    :return: visualization of the tree as an SVG
    """
    headers = {
        'Content-Type': 'text/vnd.graphviz',
        'Accept': 'image/svg+xml'
    }
    kwargs.setdefault("filled", True)
    kwargs.setdefault("rounded", True)
    kwargs.setdefault("special_characters", True)

    dot = export_graphviz(
            decision_tree=decision_tree,
            feature_names=feature_names,
            class_names=class_names,
            out_file=None,
            **kwargs)
    url = "http://gvaas.herokuapp.com/dot"
    svgtext = requests.post(url, data=dot, headers=headers).text
    return SVG(svgtext)

if __name__ == "__main__":
    print("welcome to model visualisation")
