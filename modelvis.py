"""
Visualising Machine Learning Models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import SVG
import requests
import seaborn as sns
import io

__version__ = "0.1.2"
__author__ = "Amit Kapoor <amitkaps@gmail.com>"

def plot_decision_boundaries(model, X, y,
    probability=False, show_input=False, alpha=0.5):
    """Plots the decision boundaries for a classifier with 2 features.

    This is good way to visualize the decision boundaries of various classifiers
    to build intution about them.

    It is assumed that the model is already trained.

    :param model: the classification model
    :param X: the training input samples
    :param y: the target values
    :param probability: flag to indicate whether to plot class predictions or probabilities
    :param show_input: flag to indicate whether or not to show input points
    :param alpha: the alpha value for the plots
    """

    x_min, x_max = X.iloc[:,0].min(), X.iloc[:,0].max()
    y_min, y_max = X.iloc[:,1].min(), X.iloc[:,1].max()

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, (x_max - x_min)/100),
        np.arange(y_min, y_max, (y_max - y_min)/100))

    mesh = np.c_[xx.ravel(), yy.ravel()]

    if probability:
        Z = model.predict_proba(mesh)[:,0]
    else:
        Z = model.predict(mesh)

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap="viridis", alpha=alpha)

    plt.colorbar(cs)
    if show_input:
        plt.scatter(x = X.iloc[:,0], y = X.iloc[:,1], c=y,
            s=20, cmap="magma", alpha=alpha)

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

def plot_probabilities(model, X, y, class_names=None):
    """Plots the probabilities of each class using kdeplot.

    It is assumed that the model is already trained and works only
    for 2-classes.

    :param model: the ML model
    :param X: the input data
    :param y: the classes
    :param class_names: optional class labels
    """
    y = np.array(y)
    proba = model.predict_proba(X)[:,0]

    def plot(index):
        p = proba[y == index]
        label = class_names and class_names[index] or index
        sns.kdeplot(p, shade=True, label=label)

    plot(0)
    plot(1)

def render_tree_as_code(decision_tree):
    """Converts a decision tree to equivalant python code.

    Useful to intutively understand how decision trees work.

    :param decision_tree: the decision tree model
    :return: equivalant python code
    """
    buf = io.StringIO()
    def xprint(*args, depth=0):
        indent = "    " * depth
        print(indent, *args, file=buf)

    def print_node(tree, node, depth):
        col = tree.feature[node]
        counts = tree.value[node][0]
        int_counts = [int(x) for x in counts]
        samples = int(counts.sum())
        class_ = tree.value[node].argmax()
        threshold = tree.threshold[node]
        left = tree.children_left[node]
        right = tree.children_right[node]

        xprint("# {} samples; value={}; class={}".format(samples, int_counts, class_), depth=depth)
        if col == -2:
            xprint("return", class_, depth=depth)
            return

        xprint('if row[{col}] < {threshold}:'.format(col=col, threshold=threshold), depth=depth)
        print_node(tree, left, depth+1)
        xprint('else:', depth=depth)
        print_node(tree, right, depth+1)

    tree = decision_tree.tree_
    xprint("def predict(row):")
    xprint('    """Your decision-tree model wrote this code."""')
    print_node(tree, 0, 1)
    return buf.getvalue()

def print_tree_as_code(decision_tree):
    """Prints the python code equivalant of the decision tree model.

    :param decision_tree: the decision tree model
    """
    code = render_tree_as_code(decision_tree)
    print(code)

if __name__ == "__main__":
    print("welcome to model visualisation")
