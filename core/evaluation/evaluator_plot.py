import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes):

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.tight_layout()

    return fig



def plot_roc_curve(roc_data, figsize=(6, 6)):

    """
    Parameters
    ----------
    roc_data : dict
        Format:
        {
            class_name: {
                "fpr": np.ndarray,
                "tpr": np.ndarray,
                "auc": float,
            }
        }
    """

    fig, ax = plt.subplots(figsize=figsize)

    for class_name, values in roc_data.items():

        ax.plot(
            values["fpr"],
            values["tpr"],
            label=f"{class_name} (AUC = {values['auc']:.2f})"
        )

    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend(loc="lower right")

    fig.tight_layout()

    return fig


def plot_pr_curve(pr_data, figsize=(6, 6)):

    """
    Parameters
    ----------
    pr_data : dict
        Format:
        {
            class_name: {
                "precision": np.ndarray,
                "recall": np.ndarray,
                "auc": float,
            }
        }
    """

    fig, ax = plt.subplots(figsize=figsize)

    for class_name, values in pr_data.items():

        ax.plot(
            values["recall"],
            values["precision"],
            label=f"{class_name} (AP = {values['auc']:.2f})"
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    ax.legend(loc="best")

    fig.tight_layout()

    return fig