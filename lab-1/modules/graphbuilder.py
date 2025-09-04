import matplotlib.pyplot as plt


def build_plot(x, graph_data):
    """
    Builds a plot of graphs
    :param x:
    1-D array of values for x-axi of any graphs
    :param graph_data:
    a dictionary, based on:
    data = {
        [number, starting with 1]: {
            'title': [any title],
            'lines': [
                {
                'y': [array of values of function],
                ('label'): [any label],
                ('color'): [any color],
                ('linestyle'): [any linestyle],
                ('marker'): [any marker],

                }
            ]
        },
        ...
    }
    :return: the picture of plot
    """

    # Quantity of cols and rows needed for a plot
    num_cols = len(graph_data)
    num_rows = 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6))

    # If axes is 1-D array, make it 2-D for easier handling
    if num_cols == 1:
        axes = [axes]

    # Getting index of ax and ax itself
    for i, ax in enumerate(axes):
        graph = graph_data[i + 1]  # Getting an info about graph
        ax.set_title(graph['title'])  # Setting a title for ax

        for line in graph['lines']:
            y = line['y']
            line_label = line.get('label', None)
            line_color = line.get('color', None)
            line_linestyle = line.get('linestyle', None)
            line_marker = line.get('marker', None)

            ax.plot(x,
                    y,
                    label=line_label,
                    color=line_color,
                    linestyle=line_linestyle,
                    marker=line_marker)
            ax.grid(True)

            if line_label is not None:  # If we have set up a label, then add legend
                ax.legend()

    plt.show()
