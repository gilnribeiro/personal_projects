from IPython.display import *
from ipywidgets import *

import Graph_modules.bool_cat as bc
import Graph_modules.continuous as cont
import Graph_modules.correlation as cor

def one_variable_graphs(df):
    """
        Generates a jupyter-notebook widget with 2 dropdown lists.
        - Variables:
        - Graphs:
        And generates graphs according to the choice of variables

        Parameters
        ----------
        df: pandas.DataFrame to be analysed

        returns graphs from dskc-visualization
        """
    # Create Dataset Columns list + All columns
    list_columns = list(df.columns)
    list_columns.append('All')

    output = widgets.Output()

    dropdown_columns = widgets.Dropdown(options=list_columns, description='Variables:')
    dropdown_graphs = widgets.Dropdown(description='Graphs:')

    def update(*args):
        bool_cat_one = ['Bar Chart', 'Frequency Bar Chart', 'Pie Chart', 'Donut Chart']
        continuous_one = ['Box Plot', 'Histogram -> No Outliers', 'Histogram', 'Histogram (10 - Bins)',
                          'KDE -> No Outliers', 'KDE Plot', 'Violin Plot', 'Strip Plot', 'Autocorrelation Plot',
                          'Correlation Heatmap']
        pairplot = ['Pairplot']
        continuous = list(df.select_dtypes(exclude=[object, bool, 'category']).columns)
        if dropdown_columns.value in continuous:
            dropdown_graphs.options = continuous_one
        elif dropdown_columns.value == 'All':
            dropdown_graphs.options = pairplot
        else:
            dropdown_graphs.options = bool_cat_one

    def graphs_function(df, graph=None, column=None):
        output.clear_output()
        with output:
            if column != None:
                if graph == 'Box Plot':
                    display(cont.box_plotly(df, column))
                elif graph == 'Histogram -> No Outliers':
                    display(cont.histogram_no_outliers(df, column))
                elif graph == 'Histogram':
                    display(cont.histogram_plotly(df, column))
                elif graph == 'Histogram (10 - Bins)':
                    display(cont.histogram_plotly(df, column, nbins=10))
                elif graph == 'KDE -> No Outliers':
                    display(cont.kde_univariate_no_outliers(df, column, fill=True))
                elif graph == 'KDE Plot':
                    display(cont.kde_univariate(df, column, fill=True))
                elif graph == 'Violin Plot':
                    display(cont.violin_plot_plotly(df, column, points='all'))
                elif graph == 'Pairplot' and column == 'All':
                    display(cont.pairplot(df, kind='scatter', diag_kind='kde'))
                elif graph == 'Strip Plot':
                    display(cont.strip_plot(df, column))
                elif graph == 'Autocorrelation Plot':
                    display(
                        cont.autocorrelation(df, column, timeframe='(Insert relevant Time-Frame)', figsize=(10, 6.18)))
                elif graph == 'Bar Chart':
                    display(bc.bar_chart_plotly(df, column, y=None, barmode='relative'))
                elif graph == 'Frequency Bar Chart':
                    display(bc.bar_freq_chart(df, column, figsize=(10, 8), fontsize=15, color='c'))
                elif graph == 'Pie Chart':
                    display(bc.pie_chart_plotly(df, column, numeric=None))
                elif graph == 'Donut Chart':
                    display(bc.pie_chart_plotly(df, column, numeric=None, hole=0.5))
                elif graph == 'Correlation Heatmap':
                    display(cor.variable_correlation_heatmap(df, column))
                else:
                    print('No graph was chosen')
            else:
                print('No column is active')

    def dropdown_columns_eventhandler(change):
        graphs_function(df, dropdown_graphs.value, change.new)
        update()

    def dropdown_graphs_eventhandler(change):
        graphs_function(df, change.new, dropdown_columns.value)
        update()

    dropdown_columns.observe(dropdown_columns_eventhandler, names='value')
    dropdown_graphs.observe(dropdown_graphs_eventhandler, names='value')

    display(dropdown_columns)
    display(dropdown_graphs)
    display(output)


def two_variable_graphs(df):
    """
    Generates a jupyter-notebook widget with 3 dropdown lists.
    - Variable 1:
    - Variable 2:
    - Graphs:
    And generates graphs according to the choice of variables

    Parameters
    ----------
    df: pandas.DataFrame to be analysed

    returns graphs from dskc-visualization
    """
    # Handy variables
    continuous = list(df.select_dtypes(exclude=[object, bool, 'category']).columns)
    all_columns = list(df.columns)
    num_num = ['Scatter Plot', 'Line Plot (x = Time(Number), y=Number)']
    num_cat = ['KDE (variable 1 by Category)', 'Histogram', 'Histogram (10 bins)', 'Violin Plot', 'Strip Plot',
               'Strip Plot (Overlay)']
    cat_num = ['Bar Bins', 'Bar (x = Category, y = Number)', 'Pie Chart', 'Donut Chart']
    cat_cat = ['Bar (x,y) = (Category,Category)', 'Mosaic Chart', 'KDE (variable 1 by Category)',
               'Histogram (variable 1 by Category)']

    output_2 = widgets.Output()

    dropdown_columns_1 = widgets.Dropdown(options=all_columns, description='Variables (1):')
    dropdown_columns_2 = widgets.Dropdown(description='Variables (2):')
    dropdown_graphs_2 = widgets.Dropdown(description='Graphs:')

    def update_col_2(*args):
        new_col = all_columns.copy()
        new_col.remove(dropdown_columns_1.value)
        dropdown_columns_2.options = new_col

    def update_graphs(*args):
        if dropdown_columns_1.value != [] and dropdown_columns_2.value != []:
            if dropdown_columns_1.value in continuous:
                if dropdown_columns_2.value in continuous:
                    # (Num, Num)
                    dropdown_graphs_2.options = num_num
                else:
                    # (Num, Cat)
                    dropdown_graphs_2.options = num_cat
            else:
                if dropdown_columns_2.value in continuous:
                    # (Cat, Num)
                    dropdown_graphs_2.options = cat_num
                else:
                    # (Cat, Cat)
                    dropdown_graphs_2.options = cat_cat
        else:
            dropdown_graphs_2.options = []

    def graphs_function(df, graph=None, column1=None, column2=None):
        output_2.clear_output()
        with output_2:
            if graph == 'Scatter Plot':
                display(cont.scatter_plotly(df, column1, column2))
            elif graph == 'Bar (x,y) = (Category,Category)':
                display(bc.bar_cat_cat(df, column1, column2, height=5, aspect=1, fontsize=20, palette='pastel'))
            elif graph == 'Line Plot (x = Time(Number), y=Number)':
                display(
                    cont.lineplot_time_num_cat(df, column1, column2, category=None, figsize=(10, 6.18), fontsize=10))
            elif graph == 'Bar Bins':
                display(cont.bar_bins(df, column1, column2, stacked=True, bar_type='bar', bins=10))
            elif graph == 'Bar (x = Category, y = Number)':
                display(bc.bar_cat_num_hue(df, column1, column2, height=6, aspect=2, fontsize=15, palette='pastel'))
            elif graph == 'Pie Chart':
                display(bc.pie_chart_plotly(df, column1, column2))
            elif graph == 'Donut Chart':
                display(bc.pie_chart_plotly(df, column1, column2, hole=0.5))
            elif graph == 'Mosaic Chart':
                display(bc.mosaic_graph(df, column1, column2, figsize=(16, 9.888), fontsize=20))
            elif graph == 'KDE (variable 1 by Category)':
                display(bc.kde_col_cat(df, column1, column2))
            elif graph == 'Histogram (variable 1 by Category)':
                display(bc.hist_col_cat(df, column1, column2))
            elif graph == 'Histogram':
                display(cont.histogram_plotly(df, column1, y=None, color=column2))
            elif graph == 'Histogram (10 bins)':
                display(cont.histogram_plotly(df, column1, y=None, color=column2, nbins=10))
            elif graph == 'Violin Plot':
                display(cont.violin_plot_plotly(df, column1, x=column2, color=column2, points='all'))
            elif graph == 'Strip Plot':
                display(cont.strip_plot(df, column1, x=column2, color=column2))
            elif graph == 'Strip Plot (Overlay)':
                display(cont.strip_plot(df, column1, x=None, color=column2, stripmode='overlay'))
            else:
                print('No graph was chosen')

    def dropdown_columns_eventhandler(change):
        graphs_function(df, dropdown_graphs_2.value, dropdown_columns_1.value, dropdown_columns_2.value)
        update_col_2()

    def dropdown_columns_2_eventhandler(change):
        graphs_function(df, dropdown_graphs_2.value, dropdown_columns_1.value, dropdown_columns_2.value)
        update_graphs()

    def dropdown_graphs_eventhandler(change):
        graphs_function(df, dropdown_graphs_2.value, dropdown_columns_1.value, dropdown_columns_2.value)

    dropdown_columns_1.observe(dropdown_columns_eventhandler, names='value')
    dropdown_columns_2.observe(dropdown_columns_2_eventhandler, names='value')
    dropdown_graphs_2.observe(dropdown_graphs_eventhandler, names='value')

    display(dropdown_columns_1)
    display(dropdown_columns_2)
    display(dropdown_graphs_2)
    display(output_2)

