import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(
    cm,
    class_names=("High", "Low"),
    title="Confusion Matrix - XGB - Quartiles model",
    cmap_color="Greens",
    fontsize_title=18,
    fontsize_labels=16,
    fontsize_ticks=14,
    fontsize_values=16,
    figsize=(7, 6),
    save_path=None,
):
    """
    Grafica una matriz de confusión con estilo similar a la imagen de referencia,
    usando tipografía tipo IEEE (Times New Roman / serif) y letras más grandes.

    Parámetros
    ----------
    cm : array-like (2x2 o NxN)
        Matriz de confusión, por ejemplo [[163, 19], [20, 161]].
    class_names : tuple/list
        Nombres de las clases, en el mismo orden que la matriz.
    title : str
        Título del gráfico.
    cmap_color : str
        Nombre del colormap de matplotlib (por defecto "Greens", igual al original).
    fontsize_title, fontsize_labels, fontsize_ticks, fontsize_values : int
        Tamaños de fuente para título, etiquetas de ejes, ticks y valores dentro de las celdas.
    figsize : tuple
        Tamaño de la figura.
    save_path : str o None
        Si se especifica, guarda la figura en esa ruta (por ejemplo "cm.png" o "cm.pdf").
    """

    # Configuración de fuente estilo IEEE (serif, tipo Times New Roman)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap_color)

    # Barra de color
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize_ticks)

    # Título
    ax.set_title(title, fontsize=fontsize_title, pad=15, fontweight="bold")

    # Ticks y etiquetas de clases
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=fontsize_ticks, rotation=45)
    ax.set_yticklabels(class_names, fontsize=fontsize_ticks)

    ax.set_xlabel("Predicted label", fontsize=fontsize_labels, labelpad=10)
    ax.set_ylabel("True label", fontsize=fontsize_labels, labelpad=10)

    # Umbral para decidir color de texto (blanco/negro) según intensidad de la celda
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > thresh else "black"
            ax.text(
                j,
                i,
                format(value, "d"),
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize_values,
                fontweight="bold",
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax


if __name__ == "__main__":
    # Ejemplo de uso con los mismos valores de la imagen de referencia
    cm = [[163, 19], [20, 161]]

    # PNG (raster, buena para vista previa / presentaciones)
    plot_confusion_matrix(
        cm,
        class_names=("High", "Low"),
        title="Confusion Matrix - XGB - Quartiles model",
        save_path="confusion_matrix_ieee_binary.png",
    )

    # PDF (vectorial, ideal para LaTeX / artículos IEEE)
    plot_confusion_matrix(
        cm,
        class_names=("High", "Low"),
        title="Confusion Matrix - XGB - Quartiles model",
        save_path="confusion_matrix_ieee_binary.pdf",
    )

    # SVG (vectorial, editable en Illustrator/Inkscape)
    plot_confusion_matrix(
        cm,
        class_names=("High", "Low"),
        title="Confusion Matrix - XGB - Quartiles model",
        save_path="confusion_matrix_ieee_binary.svg",
    )