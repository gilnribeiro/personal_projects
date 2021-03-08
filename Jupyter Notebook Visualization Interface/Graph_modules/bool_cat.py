import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic
import pandas as pd
import plotly.express as px


def bar_cat_cat(df, column1, column2, height=5, aspect=1, fontsize=20, title=None, palette='pastel'):
    """
    :parameter: df (pandas DataFrame)
    :parameter: column1 (specify DataFrame Column - category)
    :parameter: column2 (specify DataFrame Column - category)
    :parameter: height - size of the figure in inches - Default = 5
    :parameter: aspect - size of the figure in inches - Default = 1
    :parameter: fontsize - size of the font for the axis and title - Default = 20
    :parameter: title - String with the desired title
    :parameter: palette - choose the color set (palette) - Default = 'pastel'
    :function: create a bar chart

    """
    sns.catplot(x=column1, hue=column2, kind="count", palette=palette,
                edgecolor=".6", data=df, height=height, aspect=aspect)

    if title is None:
        plt.title(f'For each "{column1}" category, we are counting the number of "{column2}"', fontsize=fontsize - 5)
    else:
        plt.title(title, fontsize=fontsize + 5)

    plt.show()


def bar_cat_num_hue(df, column1, column2, hue=None, height=5, aspect=1, fontsize=15, title=None, palette='pastel'):
    """
    :parameter: df (pandas DataFrame)
    :parameter: column1 (specify (x's) DataFrame Column)
    :parameter: column2 (specify (y's) DataFrame Column)
    - Either "column1" or "column2" parameter, has to be numerical
    :parameter: hue (specify a DataFrame Column)
    :parameter: height - size of the figure in inches - Default = 5
    :parameter: aspect - size of the figure in inches - Default = 1
    :parameter: fontsize - size of the font for the axis and title - Default = 15
    :parameter: title - String with the desired title
    :parameter: palette - choose the color set (palette) - Default = 'pastel'
    :function: create a bar chart
    """
    sns.catplot(x=column1, y=column2, hue=hue, palette=palette, kind="bar", data=df, height=height, aspect=aspect)
    if title is None:
        plt.title(f'Estimate of the central tendency of the "{hue}"\nvalues by "{column1}" against "{column2}"',
                  fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)

    plt.show()


def bar_chart_plotly(df, x, y, color=None, facet_row=None, facet_col=None,
              facet_col_wrap=0, facet_row_spacing=None, facet_col_spacing=None, hover_name=None,
              hover_data=None, custom_data=None, text=None, base=None, error_x=None,
              error_x_minus=None, error_y=None, error_y_minus=None, animation_frame=None,
              animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None,
              color_discrete_map=None, color_continuous_scale=None, range_color=None,
              color_continuous_midpoint=None, opacity=None, orientation=None,
              barmode='relative', log_x=False, log_y=False, range_x=None, range_y=None,
              title=None, template=None, width=None, height=None):
    """
    Parameters
    ----------
    df: DataFrame or array-like or dict
        This argument needs to be passed for column names (and not keyword
        names) to be used. Array-like and dict are tranformed internally to a
        pandas DataFrame. Optional: if missing, a DataFrame gets constructed
        under the hood using the other arguments.
    x: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the x axis in cartesian coordinates. Either `x` or
        `y` can optionally be a list of column references or array_likes,  in
        which case the data will be treated as if it were 'wide' rather than
        'long'.
    y: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the y axis in cartesian coordinates. Either `x` or
        `y` can optionally be a list of column references or array_likes,  in
        which case the data will be treated as if it were 'wide' rather than
        'long'.
    color: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    facet_row: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to facetted subplots in the vertical direction.
    facet_col: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to facetted subplots in the horizontal direction.
    facet_col_wrap: int
        Maximum number of facet columns. Wraps the column variable at this
        width, so that the column facets span multiple rows. Ignored if 0, and
        forced to 0 if `facet_row` or a `marginal` is set.
    facet_row_spacing: float between 0 and 1
        Spacing between facet rows, in paper units. Default is 0.03 or 0.0.7
        when facet_col_wrap is used.
    facet_col_spacing: float between 0 and 1
        Spacing between facet columns, in paper units Default is 0.02.
    hover_name: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in bold
        in the hover tooltip.
    hover_data: list of str or int, or Series or array-like, or dict
        Either a list of names of columns in `data_frame`, or pandas Series, or
        array_like objects or a dict with column names as keys, with values
        True (for default formatting) False (in order to remove this column
        from hover information), or a formatting string, for example ':.3f' or
        '|%a' or list-like data to appear in the hover tooltip or tuples with a
        bool or formatting string as first element, and list-like data to
        appear in hover as second element Values from these columns appear as
        extra data in the hover tooltip.
    custom_data: list of str or int, or Series or array-like
        Either names of columns in `data_frame`, or pandas Series, or
        array_like objects Values from these columns are extra data, to be used
        in widgets or Dash callbacks for example. This data is not user-visible
        but is included in events emitted by the figure (lasso selection etc.)
    text: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in the
        figure as text labels.
    base: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position the base of the bar.
    error_x: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        size x-axis error bars. If `error_x_minus` is `None`, error bars will
        be symmetrical, otherwise `error_x` is used for the positive direction
        only.
    error_x_minus: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        size x-axis error bars in the negative direction. Ignored if `error_x`
        is `None`.
    error_y: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        size y-axis error bars. If `error_y_minus` is `None`, error bars will
        be symmetrical, otherwise `error_y` is used for the positive direction
        only.
    error_y_minus: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        size y-axis error bars in the negative direction. Ignored if `error_y`
        is `None`.
    animation_frame: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to animation frames.
    animation_group: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        provide object-constancy across animation frames: rows with matching
        `animation_group`s will be treated as if they describe the same object
        in each frame.
    category_orders: dict with str keys and list of str values (default `{}`)
        By default, in Python 3.6+, the order of categorical values in axes,
        legends and facets depends on the order in which these values are first
        encountered in `data_frame` (and no order is guaranteed by default in
        Python below 3.6). This parameter is used to force a specific ordering
        of values per column. The keys of this dict should correspond to column
        names, and the values should be lists of strings corresponding to the
        specific display order desired.
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in the figure for axis titles, legend
        entries and hovers. This parameter allows this to be overridden. The
        keys of this dict should correspond to column names, and the values
        should correspond to the desired label to be displayed.
    color_discrete_sequence: list of str
        Strings should define valid CSS-colors. When `color` is set and the
        values in the corresponding column are not numeric, values in that
        column are assigned colors by cycling through `color_discrete_sequence`
        in the order described in `category_orders`, unless the value of
        `color` is a key in `color_discrete_map`. Various useful color
        sequences are available in the `plotly.express.colors` submodules,
        specifically `plotly.express.colors.qualitative`.
    color_discrete_map: dict with str keys and str values (default `{}`)
        String values should define valid CSS-colors Used to override
        `color_discrete_sequence` to assign a specific colors to marks
        corresponding with specific values. Keys in `color_discrete_map` should
        be values in the column denoted by `color`. Alternatively, if the
        values of `color` are valid colors, the string `'identity'` may be
        passed to cause them to be used directly.
    color_continuous_scale: list of str
        Strings should define valid CSS-colors This list is used to build a
        continuous color scale when the column denoted by `color` contains
        numeric data. Various useful color scales are available in the
        `plotly.express.colors` submodules, specifically
        `plotly.express.colors.sequential`, `plotly.express.colors.diverging`
        and `plotly.express.colors.cyclical`.
    range_color: list of two numbers
        If provided, overrides auto-scaling on the continuous color scale.
    color_continuous_midpoint: number (default `None`)
        If set, computes the bounds of the continuous color scale to have the
        desired midpoint. Setting this value is recommended when using
        `plotly.express.colors.diverging` color scales as the inputs to
        `color_continuous_scale`.
    opacity: float
        Value between 0 and 1. Sets the opacity for markers.
    orientation: str, one of `'h'` for horizontal or `'v'` for vertical.
        (default `'v'` if `x` and `y` are provided and both continous or both
        categorical,  otherwise `'v'`(`'h'`) if `x`(`y`) is categorical and
        `y`(`x`) is continuous,  otherwise `'v'`(`'h'`) if only `x`(`y`) is
        provided)
    barmode: str (default `'relative'`)
        One of `'group'`, `'overlay'` or `'relative'` In `'relative'` mode,
        bars are stacked above zero for positive values and below zero for
        negative values. In `'overlay'` mode, bars are drawn on top of one
        another. In `'group'` mode, bars are placed beside each other.
    log_x: boolean (default `False`)
        If `True`, the x-axis is log-scaled in cartesian coordinates.
    log_y: boolean (default `False`)
        If `True`, the y-axis is log-scaled in cartesian coordinates.
    range_x: list of two numbers
        If provided, overrides auto-scaling on the x-axis in cartesian
        coordinates.
    range_y: list of two numbers
        If provided, overrides auto-scaling on the y-axis in cartesian
        coordinates.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.
        """

    if y is None and color is None and df[x].dtypes not in [float, int]:
        color = x
    elif x is None and color is None and df[y].dtypes not in [float, int]:
        color = y

    fig = px.bar(df, x, y, color=color, facet_row=facet_row, facet_col=facet_col,
                 facet_col_wrap=facet_col_wrap, facet_row_spacing=facet_row_spacing,
                 facet_col_spacing=facet_col_spacing,
                 hover_name=hover_name, hover_data=hover_data, custom_data=custom_data, text=text, base=base,
                 error_x=error_x,
                 error_x_minus=error_x_minus, error_y=error_y, error_y_minus=error_y_minus,
                 animation_frame=animation_frame,
                 animation_group=animation_group, category_orders=category_orders, labels=labels,
                 color_discrete_sequence=color_discrete_sequence,
                 color_discrete_map=color_discrete_map, color_continuous_scale=color_continuous_scale,
                 range_color=range_color,
                 color_continuous_midpoint=color_continuous_midpoint, opacity=opacity, orientation=orientation,
                 barmode=barmode, log_x=log_x, log_y=log_y, range_x=range_x, range_y=range_y,
                 title=title, template=template, width=width, height=height)

    if title is None and y is None:
        fig.update_layout(title=f'{x.capitalize()} Bar Chart', height=height, width=width)
    elif title is None and x is None:
        fig.update_layout(title=f'{y.capitalize()} Bar Chart', height=height, width=width)
    elif title is None:
        fig.update_layout(title=f'Bar Chart of {x.capitalize()} against {y.capitalize()}', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()


def bar_freq_chart(df, column, figsize=(30, 25), fontsize=30, title=None, color='blue', xlim=None, ylim=(0, 100)):
    """
    :parameter: df (pandas DataFrame)
    :parameter: column (specify DataFrame Column)
    :parameter: figsize - tuple by default = (30,25)
    :parameter: fontsize - size of the font for the axis and title - Default = 30
    :parameter: title - String with the desired title
    :parameter: color - list with the desired colors
    :parameter: xlim - tuple (a,b) to adjust the size of the x axis
    :parameter: ylim - tuple (a,b) to adjust the size of the y axis - Default = (0,100)
    :function: create a frequency bar chart

    """
    # Data to Plot
    labels = list(df[column].dropna().unique())
    x_size = np.arange(len(labels))
    hist_column = df[column].value_counts()

    # Plot
    fig, sp = plt.subplots(figsize=figsize)
    (100 * hist_column / len(df)).plot(kind='bar', color=color)

    # Labels
    for x in range(len(hist_column)):
        plt.text(x, (100 * hist_column[x] / len(df) + 2), hist_column[x], fontsize=fontsize - 5)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xticks(x_size, labels, rotation=0, fontsize=fontsize - 5)
    plt.yticks(fontsize=fontsize - 5)
    plt.ylabel(f"% {column} ", fontsize=fontsize - 5)
    if title is None:
        plt.title(f"Frequency of {column}", fontsize=fontsize + 5)

    else:
        plt.title(title, fontsize=fontsize + 5)

    plt.tight_layout()

    # Show
    plt.show()


def hist_col_cat(df, column, category, stacked=False, figsize=(46, 30), fontsize=40, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: column - Category, binary or numeric - (specify DataFrame Column)
    :parameter: category - (category or boolean), will be the defined legend (avoid numeric) (specify Dataframe Column)
    :parameter: stacked - True, False - Default = False
    :parameter: figsize - tuple by default = (46,30)
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: title - Title of the graph
    :function: create a histogram plot with stacked option
    """
    df_count = df[[column, category]].groupby([column, category]).size().reset_index(name='counts')
    df_count.set_index(column, inplace=True)
    df_count = df_count.pivot(columns=category, values="counts")

    # Ordering from lowest to highest
    df_count.columns = df_count.columns.astype(str)
    df_count['total'] = df_count.sum(axis=1)
    df_count.sort_values(by='total', inplace=True)
    df_count.drop(columns='total', inplace=True)

    df_count.plot(kind='hist', stacked=stacked, figsize=figsize)

    # Labels
    plt.xlabel(f'{column}', fontsize=fontsize)
    plt.ylabel(f'Frequency of {category} on {column}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 10)
    plt.yticks(fontsize=fontsize - 10)

    if title is None:
        plt.title(f"Histogram distribution of '{category}' over '{column}'",
                  fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize)

    plt.legend(fontsize=fontsize, title=category, title_fontsize=fontsize)

    # Plot
    plt.show()


def kde_col_cat(df, column, category, stacked=False, figsize=(46, 30), fontsize=40, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: column - Category, binary or numeric - (specify DataFrame Column)
    :parameter: category - (category or boolean), will be the defined legend (avoid numeric) (specify Dataframe Column)
    :parameter: figsize - tuple by default = (46,30)
    :parameter: stacked - True, False - Default = False
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: title - Title of the graph
    :function: create a kde plot with stacked option
    """
    df_count = df[[column, category]].groupby([column, category]).size().reset_index(name='counts')
    df_count.set_index(column, inplace=True)
    df_count = df_count.pivot(columns=category, values="counts")

    # Ordering from lowest to highest
    df_count.columns = df_count.columns.astype(str)
    df_count['total'] = df_count.sum(axis=1)
    df_count.sort_values(by='total', inplace=True)
    df_count.drop(columns='total', inplace=True)

    df_count.plot(kind='kde', stacked=stacked, figsize=figsize)

    # Labels
    plt.xlabel(f'{column}', fontsize=fontsize)
    plt.ylabel(f'Frequency of {category} on {column}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 10)
    plt.yticks(fontsize=fontsize - 10)
    if title is None:
        plt.title(f"Kde distribution of '{category}' over '{column}'",
                  fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize)

    plt.legend(fontsize=fontsize, title=category, title_fontsize=fontsize)

    # Plot
    plt.show()


def mosaic_graph(df, column1, column2, figsize=(16, 16 * 0.618), fontsize=20, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: column1 (specify DataFrame Column - category)
    :parameter: column2 (specify DataFrame Column - category)
    :parameter: figsize - tuple for size of the figure - Default = (16,16*0.618)
    :parameter: fontsize - size of the font for the axis and title - Default = 12
    :function: create a mosaic chart, these charts are good to find the largest and smallest categories

    """
    fig, ax = plt.subplots(figsize=figsize)
    mosaic(df, [column1, column2], horizontal=False, ax=ax)
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.color"] = "k"
    if title is None:
        ax.set_title(f"{column1} vs {column2}", fontsize=fontsize + 10)
    else:
        ax.set_title(title, fontsize=fontsize + 5)
    ax.set_xlabel(column2)
    ax.set_ylabel(column1)
    plt.tight_layout()
    # Plot
    plt.show()


def pie_chart_plotly(df, category, numeric, color=None, color_discrete_sequence=None,
                    color_discrete_map=None, hover_name=None, hover_data=None, custom_data=None, labels=None,
                    title=None, template=None, width=None, height=None, opacity=None, hole=None):
    """
    Parameters
    ----------
    df: DataFrame or array-like or dict
        This argument needs to be passed for column names (and not keyword
        names) to be used. Array-like and dict are tranformed internally to a
        pandas DataFrame. Optional: if missing, a DataFrame gets constructed
        under the hood using the other arguments.
    category: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used as
        labels for sectors.
    numeric: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        set values associated to sectors.
    color: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    color_discrete_sequence: list of str
        Strings should define valid CSS-colors. When `color` is set and the
        values in the corresponding column are not numeric, values in that
        column are assigned colors by cycling through `color_discrete_sequence`
        in the order described in `category_orders`, unless the value of
        `color` is a key in `color_discrete_map`. Various useful color
        sequences are available in the `plotly.express.colors` submodules,
        specifically `plotly.express.colors.qualitative`.
    color_discrete_map: dict with str keys and str values (default `{}`)
        String values should define valid CSS-colors Used to override
        `color_discrete_sequence` to assign a specific colors to marks
        corresponding with specific values. Keys in `color_discrete_map` should
        be values in the column denoted by `color`. Alternatively, if the
        values of `color` are valid colors, the string `'identity'` may be
        passed to cause them to be used directly.
    hover_name: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in bold
        in the hover tooltip.
    hover_data: list of str or int, or Series or array-like, or dict
        Either a list of names of columns in `data_frame`, or pandas Series, or
        array_like objects or a dict with column names as keys, with values
        True (for default formatting) False (in order to remove this column
        from hover information), or a formatting string, for example ':.3f' or
        '|%a' or list-like data to appear in the hover tooltip or tuples with a
        bool or formatting string as first element, and list-like data to
        appear in hover as second element Values from these columns appear as
        extra data in the hover tooltip.
    custom_data: list of str or int, or Series or array-like
        Either names of columns in `data_frame`, or pandas Series, or
        array_like objects Values from these columns are extra data, to be used
        in widgets or Dash callbacks for example. This data is not user-visible
        but is included in events emitted by the figure (lasso selection etc.)
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in the figure for axis titles, legend
        entries and hovers. This parameter allows this to be overridden. The
        keys of this dict should correspond to column names, and the values
        should correspond to the desired label to be displayed.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.
    opacity: float
        Value between 0 and 1. Sets the opacity for markers.
    hole: float
        Sets the fraction of the radius to cut out of the pie.Use this to make
        a donut chart.

    Returns
    -------
        plotly.graph_objects.Figure
        """
    fig = px.pie(data_frame=df, names=category, values=numeric, color=color,
                 color_discrete_sequence=color_discrete_sequence,
                 color_discrete_map=color_discrete_map, hover_name=hover_name, hover_data=hover_data,
                 custom_data=custom_data,
                 labels=labels, title=title, template=template, width=width, height=height, opacity=opacity, hole=hole)

    if title is None and hole is None:
        fig.update_layout(title=f'Pie Chart', height=height, width=width)
    elif title is None and hole is not None:
        fig.update_layout(title=f'Donut Chart', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()


def stacked_bar_cat_cat(df, category1, category2, figsize=(46, 30), fontsize=40, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: category1 - categorical (specify DataFrame Column)
    :parameter: category2 - categorical (legend) (specify Dataframe Column)
    :parameter: figsize - tuple by default = (46,30)
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: title - title of the plot
    :function: create a stacked bar plot with two categories
    """
    df_count = df[[category1, category2]].groupby([category1, category2]).size().reset_index(name='counts')
    df_count.set_index(category1, inplace=True)
    df_count = df_count.pivot(columns=category2, values="counts")

    # Ordering from lowest to highest
    df_count.columns = df_count.columns.astype(str)
    df_count['total'] = df_count.sum(axis=1)
    df_count.sort_values(by='total', inplace=True)
    df_count.drop(columns='total', inplace=True)

    fig, ax = plt.subplots()
    df_count.plot(kind='bar', stacked=True, figsize=figsize, ax=ax)

    # Labels
    plt.xlabel(f'{category1}', fontsize=fontsize)
    plt.ylabel(f'Number of {category2} per {category1}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 10)
    plt.yticks(fontsize=fontsize - 10)
    plt.legend(fontsize=fontsize, title=category2, title_fontsize=fontsize)
    if title is None:
        plt.title(f"Stacked bar plot '{category2}' over '{category1}'",
                  fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize)

    y_values = df_count.sum(axis=1)
    for i in ax.get_xticks():
        plt.text(i - 0.05, y_values[i] + 0.5, y_values[i], fontsize=fontsize)

    # Plot
    plt.show()


def violin_plot_plotly(df, y, x=None, color=None, facet_row=None, facet_col=None,
                facet_col_wrap=0, facet_row_spacing=None, facet_col_spacing=None, hover_name=None,
                hover_data=None, custom_data=None, animation_frame=None, animation_group=None,
                category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None,
                orientation=None, violinmode=None, log_x=False, log_y=False, range_x=None, range_y=None,
                points=None, box=False, title=None, template=None, width=None, height=None):
    """
    Parameters
    ----------
    df: DataFrame or array-like or dict
        This argument needs to be passed for column names (and not keyword
        names) to be used. Array-like and dict are tranformed internally to a
        pandas DataFrame. Optional: if missing, a DataFrame gets constructed
        under the hood using the other arguments.
    x: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the x axis in cartesian coordinates. Either `x` or
        `y` can optionally be a list of column references or array_likes,  in
        which case the data will be treated as if it were 'wide' rather than
        'long'.
    y: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the y axis in cartesian coordinates. Either `x` or
        `y` can optionally be a list of column references or array_likes,  in
        which case the data will be treated as if it were 'wide' rather than
        'long'.
    color: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    facet_row: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to facetted subplots in the vertical direction.
    facet_col: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to facetted subplots in the horizontal direction.
    facet_col_wrap: int
        Maximum number of facet columns. Wraps the column variable at this
        width, so that the column facets span multiple rows. Ignored if 0, and
        forced to 0 if `facet_row` or a `marginal` is set.
    facet_row_spacing: float between 0 and 1
        Spacing between facet rows, in paper units. Default is 0.03 or 0.0.7
        when facet_col_wrap is used.
    facet_col_spacing: float between 0 and 1
        Spacing between facet columns, in paper units Default is 0.02.
    hover_name: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in bold
        in the hover tooltip.
    hover_data: list of str or int, or Series or array-like, or dict
        Either a list of names of columns in `data_frame`, or pandas Series, or
        array_like objects or a dict with column names as keys, with values
        True (for default formatting) False (in order to remove this column
        from hover information), or a formatting string, for example ':.3f' or
        '|%a' or list-like data to appear in the hover tooltip or tuples with a
        bool or formatting string as first element, and list-like data to
        appear in hover as second element Values from these columns appear as
        extra data in the hover tooltip.
    custom_data: list of str or int, or Series or array-like
        Either names of columns in `data_frame`, or pandas Series, or
        array_like objects Values from these columns are extra data, to be used
        in widgets or Dash callbacks for example. This data is not user-visible
        but is included in events emitted by the figure (lasso selection etc.)
    animation_frame: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to animation frames.
    animation_group: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        provide object-constancy across animation frames: rows with matching
        `animation_group`s will be treated as if they describe the same object
        in each frame.
    category_orders: dict with str keys and list of str values (default `{}`)
        By default, in Python 3.6+, the order of categorical values in axes,
        legends and facets depends on the order in which these values are first
        encountered in `data_frame` (and no order is guaranteed by default in
        Python below 3.6). This parameter is used to force a specific ordering
        of values per column. The keys of this dict should correspond to column
        names, and the values should be lists of strings corresponding to the
        specific display order desired.
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in the figure for axis titles, legend
        entries and hovers. This parameter allows this to be overridden. The
        keys of this dict should correspond to column names, and the values
        should correspond to the desired label to be displayed.
    color_discrete_sequence: list of str
        Strings should define valid CSS-colors. When `color` is set and the
        values in the corresponding column are not numeric, values in that
        column are assigned colors by cycling through `color_discrete_sequence`
        in the order described in `category_orders`, unless the value of
        `color` is a key in `color_discrete_map`. Various useful color
        sequences are available in the `plotly.express.colors` submodules,
        specifically `plotly.express.colors.qualitative`.
    color_discrete_map: dict with str keys and str values (default `{}`)
        String values should define valid CSS-colors Used to override
        `color_discrete_sequence` to assign a specific colors to marks
        corresponding with specific values. Keys in `color_discrete_map` should
        be values in the column denoted by `color`. Alternatively, if the
        values of `color` are valid colors, the string `'identity'` may be
        passed to cause them to be used directly.
    orientation: str, one of `'h'` for horizontal or `'v'` for vertical.
        (default `'v'` if `x` and `y` are provided and both continous or both
        categorical,  otherwise `'v'`(`'h'`) if `x`(`y`) is categorical and
        `y`(`x`) is continuous,  otherwise `'v'`(`'h'`) if only `x`(`y`) is
        provided)
    violinmode: str (default `'group'`)
        One of `'group'` or `'overlay'` In `'overlay'` mode, violins are on
        drawn top of one another. In `'group'` mode, violins are placed beside
        each other.
    log_x: boolean (default `False`)
        If `True`, the x-axis is log-scaled in cartesian coordinates.
    log_y: boolean (default `False`)
        If `True`, the y-axis is log-scaled in cartesian coordinates.
    range_x: list of two numbers
        If provided, overrides auto-scaling on the x-axis in cartesian
        coordinates.
    range_y: list of two numbers
        If provided, overrides auto-scaling on the y-axis in cartesian
        coordinates.
    points: str or boolean (default `'outliers'`)
        One of `'outliers'`, `'suspectedoutliers'`, `'all'`, or `False`. If
        `'outliers'`, only the sample points lying outside the whiskers are
        shown. If `'suspectedoutliers'`, all outlier points are shown and those
        less than 4*Q1-3*Q3 or greater than 4*Q3-3*Q1 are highlighted with the
        marker's `'outliercolor'`. If `'outliers'`, only the sample points
        lying outside the whiskers are shown. If `'all'`, all sample points are
        shown. If `False`, no sample points are shown and the whiskers extend
        to the full range of the sample.
    box: boolean (default `False`)
        If `True`, boxes are drawn inside the violins.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.
        """

    if y is None and color is None and df[x].dtypes not in [float, int]:
        color = x
    elif x is None and color is None and df[y].dtypes not in [float, int]:
        color = y
    elif x != None and y != None and color is None:
        if df[x].dtypes not in [float, int]:
            color = x
        elif df[y].dtypes not in [float, int]:
            color = y

    fig = px.violin(df, y=y, x=x, color=color, facet_row=facet_row, facet_col=facet_col,
                    facet_col_wrap=facet_col_wrap, facet_row_spacing=facet_row_spacing,
                    facet_col_spacing=facet_col_spacing,
                    hover_name=hover_name, hover_data=hover_data, custom_data=custom_data,
                    animation_frame=animation_frame,
                    animation_group=animation_group, category_orders=category_orders, labels=labels,
                    color_discrete_sequence=color_discrete_sequence, color_discrete_map=color_discrete_map,
                    orientation=orientation, violinmode=violinmode, log_x=log_x, log_y=log_y, range_x=range_x,
                    range_y=range_y,
                    points=points, box=box, title=title, template=template, width=width, height=height)

    if title is None and y is None:
        fig.update_layout(title=f'{x.capitalize()} Violin Plot', height=height, width=width)
    elif title is None and x is None:
        fig.update_layout(title=f'{y.capitalize()} Violin Plot', height=height, width=width)
    elif title is None and color is not None and x != color and y != color:
        fig.update_layout(title=f'Bar Chart of {x.capitalize()} against {y.capitalize()} by {color.capitalize()}',
                          height=height, width=width)
    elif title is None:
        fig.update_layout(title=f'Bar Chart of {x.capitalize()} against {y.capitalize()}', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()
