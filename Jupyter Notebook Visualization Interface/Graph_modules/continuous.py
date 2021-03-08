import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots


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

def autocorrelation(df, numeric, timeframe='days', lags=None, alpha=0.05, obs_threshold='...',
                    figsize=(10, 10 * 0.618), fontsize=5, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: numeric - (y axis) - (specify Numeric DataFrame Column)
    :parameter: timeframe - Insert adequate time frame e.g(day, year, month) - Default = 'days'
    :parameter: lags - size of the lag - Default = None
    :parameter: alpha - error size (relates to confidence intervals) - Default = 0.05
    :parameter: obs_threshold - Insert the observed threshold to appear on the title - Default = '...'
    :parameter: figsize - tuple - Default = (10,10*0.618)
    :parameter: fontsize - size of the font for the title - Default = 10
    :parameter: title - graph title (str) - Default = None
    :function: create an Autocorrelation plot

    """

    # Autocorrelation Plot
    fig, ax = plt.subplots(figsize=figsize)
    tsaplots.plot_acf(df[numeric].dropna(), lags=lags, alpha=alpha, ax=ax)
    plt.xlabel(timeframe)
    plt.ylabel("Autocorrelation")
    if title is None:
        plt.title(
            f"""I could predict new values for my time\n 
            series with {obs_threshold} {timeframe} in advance for an alpha of {alpha}""",
            fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize + 10)

    plt.show()


def bar_bins(df, category, numeric, stacked=True, bar_type='bar', bins=10, figsize=(46, 30), fontsize=40, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: category - (y axis), (specify Categorical Dataframe Column)
    :parameter: numeric - legend - (specify Numeric DataFrame Column)
    :parameter: stacked - boolean value for stacking the graph - Default = True
    :parameter: bar_type - orientation of the bar graph 'bar' (vertical), 'barh' (horizontal) - Default = 'bar'
    :parameter: bins - Insert the number of bins - Default = 10
    :parameter: figsize - tuple by default = (46,30)
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: title - graph title (str)
    :function: create a horizontal stacked bar plot with x axis being the number of categorical observations
    per numeric bin
    """
    # creating bins for the numeric variable
    df_count = df[[category, numeric]].copy()
    df_count['bins'] = pd.cut(df_count[numeric], bins=bins)

    # counting elements in a subset of columns and add it as a columns
    df_count = df_count.groupby([category, 'bins']).size().reset_index(name='counts')

    # setting Category as Index
    df_count.set_index(category, inplace=True)

    # setting the values of Numeric as new columns and counts as the values in the table
    df_count = df_count.pivot(columns='bins', values="counts")

    # add a new column with the total of the counted numerical variable

    df_count.columns = df_count.columns.add_categories('total')
    df_count['total'] = df_count.sum(axis=1)

    # sort values acording to the total
    df_count.sort_values(by="total", ascending=False, inplace=True)

    total = df_count["total"]
    df_count.drop(columns=["total"], inplace=True)

    # Labels
    df_count.plot(kind=bar_type, stacked=stacked, figsize=figsize)
    plt.ylabel(f"{category}", fontsize=fontsize)
    plt.xlabel(f'Count of "{numeric}" observations', fontsize=fontsize)

    if stacked is True and bar_type == 'bar':
        [plt.text(i, total[i] * 1.01, str(total[i]), fontsize=fontsize - 10) for i in range(len(df_count))]
    if stacked is True and bar_type == 'barh':
        [plt.text(total[i] * 1.01, i, str(total[i]), fontsize=fontsize - 10) for i in range(len(df_count))]

    plt.legend(title=numeric, title_fontsize=fontsize, bbox_to_anchor=(1., 0., 0.1, 0.5), fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 10, rotation=0)
    plt.yticks(fontsize=fontsize - 10)
    if title is None:
        plt.title(f'Number of observations of "{category}" per "{numeric}"', fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize + 10)

    # Plot
    plt.show()


def box_plotly(df, y, x=None, color=None, hover_name=None, hover_data=None, custom_data=None,
               animation_frame=None, animation_group=None, category_orders=None, labels=None,
               color_discrete_sequence=None,
               color_discrete_map=None, orientation=None, boxmode=None, log_x=False, log_y=False, range_x=None,
               range_y=None, points=None, notched=False, title=None, template=None, width=None, height=None):
    """
    :function: Box Plot
        In a box plot, rows of `data_frame` are grouped together into a
        box-and-whisker mark to visualize their distribution.

        Each box spans from quartile 1 (Q1) to quartile 3 (Q3). The second
        quartile (Q2) is marked by a line inside the box. By default, the
        whiskers correspond to the box' edges +/- 1.5 times the interquartile
        range (IQR: Q3-Q1), see "points" for other options.

    Parameters
    ----------
    df: DataFrame or array-like or dict
        This argument needs to be passed for column names (and not keyword
        names) to be used. Array-like and dict are tranformed internally to a
        pandas DataFrame. Optional: if missing, a DataFrame gets constructed
        under the hood using the other arguments.
    y: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the y axis in cartesian coordinates. Either `x` or
        `y` can optionally be a list of column references or array_likes,  in
        which case the data will be treated as if it were 'wide' rather than
        'long'.
    x: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the x axis in cartesian coordinates. Either `x` or
        `y` can optionally be a list of column references or array_likes,  in
        which case the data will be treated as if it were 'wide' rather than
        'long'.
    color: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
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
    boxmode: str (default `'group'`)
        One of `'group'` or `'overlay'` In `'overlay'` mode, boxes are on drawn
        top of one another. In `'group'` mode, baxes are placed beside each
        other.
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
    notched: boolean (default `False`)
        If `True`, boxes are drawn with notches.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.

    Returns
    -------
        plotly.graph_objects.Figure
    """
    fig = px.box(df, x=x, y=y, color=color, hover_name=hover_name, hover_data=hover_data, custom_data=custom_data,
                 animation_frame=animation_frame, animation_group=animation_group, category_orders=category_orders,
                 labels=labels, color_discrete_sequence=color_discrete_sequence, color_discrete_map=color_discrete_map,
                 orientation=orientation, boxmode=boxmode, log_x=log_x, log_y=log_y, range_x=range_x, range_y=range_y,
                 points=points, notched=notched, template=template)

    if title is None:
        fig.update_layout(title=f'Box Plot', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()


def reject_outliers(df, x, m=2):
    """
    :function: Rejecting outliers
    :parameters: df (pandas data frame)
    :parameters: x - dataframe column
    :parameters: m - standard deviation multiplier - default = 2

    """
    return df[x][abs(df[x] - np.mean(df[x])) < m * np.std(df[x])]


def histogram_no_outliers(df, x, y=None, color=None, histnorm=None, log_x=False, log_y=False, range_x=None, range_y=None,
                         histfunc=None, cumulative=None, nbins=None, title=None, template=None, width=None,
                         height=None):
    """

    :function: Plot Histogram
        In a histogram, rows of `data_frame` are grouped together into a
        rectangular mark to visualize the 1D distribution of an aggregate
        function `histfunc`

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
        position marks along the x axis in cartesian coordinates. If
        `orientation` is `'h'`, these values are used as inputs to `histfunc`.
        Either `x` or `y` can optionally be a list of column references or
        array_likes,  in which case the data will be treated as if it were
        'wide' rather than 'long'.
    y: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the y axis in cartesian coordinates. If
        `orientation` is `'v'`, these values are used as inputs to `histfunc`.
        Either `x` or `y` can optionally be a list of column references or
        array_likes,  in which case the data will be treated as if it were
        'wide' rather than 'long'.
    color: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    histnorm: str (default `None`)
        One of `'percent'`, `'probability'`, `'density'`, or `'probability
        density'` If `None`, the output of `histfunc` is used as is. If
        `'probability'`, the output of `histfunc` for a given bin is divided by
        the sum of the output of `histfunc` for all bins. If `'percent'`, the
        output of `histfunc` for a given bin is divided by the sum of the
        output of `histfunc` for all bins and multiplied by 100. If
        `'density'`, the output of `histfunc` for a given bin is divided by the
        size of the bin. If `'probability density'`, the output of `histfunc`
        for a given bin is normalized such that it corresponds to the
        probability that a random event whose distribution is described by the
        output of `histfunc` will fall into that bin.
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
    histfunc: str (default `'count'` if no arguments are provided, else `'sum'`)
        One of `'count'`, `'sum'`, `'avg'`, `'min'`, or `'max'`.Function used
        to aggregate values for summarization (note: can be normalized with
        `histnorm`). The arguments to this function are the values of `y`(`x`)
        if `orientation` is `'v'`(`'h'`).
    cumulative: boolean (default `False`)
        If `True`, histogram values are cumulative.
    nbins: int
        Positive integer. Sets the number of bins.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.

    Returns
    -------
        plotly.graph_objects.Figure
    """

    def reject_outliers(df, x, m=2):
        """
        :function: Rejecting outliers
        :parameters: df (pandas data frame)
        :parameters: x - dataframe column
        :parameters: m - standard deviation multiplier - default = 2

        """
        return df[x][abs(df[x] - np.mean(df[x])) < m * np.std(df[x])]

    filtered_df = reject_outliers(df, x)

    fig = px.histogram(filtered_df, x=x, y=y, color=color, histnorm=histnorm, log_x=log_x, log_y=log_y, range_x=range_x,
                       range_y=range_y, histfunc=histfunc, cumulative=cumulative, nbins=nbins, template=template)
    if title is None:
        fig.update_layout(title=f'Histogram', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()


def histogram_plotly(df, x, y=None, color=None, histnorm=None, log_x=False, log_y=False, range_x=None, range_y=None,
                     histfunc=None, cumulative=None, nbins=None, title=None, template=None, width=None, height=None):
    """

    :function: Plot Histogram filtering out the outliers
        In a histogram, rows of `data_frame` are grouped together into a
        rectangular mark to visualize the 1D distribution of an aggregate
        function `histfunc`

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
        position marks along the x axis in cartesian coordinates. If
        `orientation` is `'h'`, these values are used as inputs to `histfunc`.
        Either `x` or `y` can optionally be a list of column references or
        array_likes,  in which case the data will be treated as if it were
        'wide' rather than 'long'.
    y: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the y axis in cartesian coordinates. If
        `orientation` is `'v'`, these values are used as inputs to `histfunc`.
        Either `x` or `y` can optionally be a list of column references or
        array_likes,  in which case the data will be treated as if it were
        'wide' rather than 'long'.
    color: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    histnorm: str (default `None`)
        One of `'percent'`, `'probability'`, `'density'`, or `'probability
        density'` If `None`, the output of `histfunc` is used as is. If
        `'probability'`, the output of `histfunc` for a given bin is divided by
        the sum of the output of `histfunc` for all bins. If `'percent'`, the
        output of `histfunc` for a given bin is divided by the sum of the
        output of `histfunc` for all bins and multiplied by 100. If
        `'density'`, the output of `histfunc` for a given bin is divided by the
        size of the bin. If `'probability density'`, the output of `histfunc`
        for a given bin is normalized such that it corresponds to the
        probability that a random event whose distribution is described by the
        output of `histfunc` will fall into that bin.
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
    histfunc: str (default `'count'` if no arguments are provided, else `'sum'`)
        One of `'count'`, `'sum'`, `'avg'`, `'min'`, or `'max'`.Function used
        to aggregate values for summarization (note: can be normalized with
        `histnorm`). The arguments to this function are the values of `y`(`x`)
        if `orientation` is `'v'`(`'h'`).
    cumulative: boolean (default `False`)
        If `True`, histogram values are cumulative.
    nbins: int
        Positive integer. Sets the number of bins.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.

    Returns
    -------
        plotly.graph_objects.Figure
    """
    fig = px.histogram(df, x=x, y=y, color=color, histnorm=histnorm, log_x=log_x, log_y=log_y, range_x=range_x,
                       range_y=range_y, histfunc=histfunc, cumulative=cumulative, nbins=nbins, template=template)
    if title is None:
        fig.update_layout(title=f'Histogram', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()


def lineplot_time_num_cat(df, time, numeric, category=None, figsize=(10, 10 * 0.618), fontsize=10, title=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: time - day, year, month, date
    :parameter: numeric - (y axis) - (specify Numeric DataFrame Column)
    :parameter: category - legend (specify Categorical Dataframe Column) - default = None
    :parameter: figsize - tuple - Default = (10,10*0.618)
    :parameter: fontsize - size of the font for the title - Default = 10
    :parameter: title - graph title (str) - Default = None
    :function: create a time series graph with y axis (numeric), time on (x axis) and a optional legend

    """
    # Initiate Figure
    fig, ax = plt.subplots(figsize=figsize)

    if category is None:
        sns.lineplot(data=df, x=time, y=numeric, ax=ax)
        if title is None:
            plt.title(f'Evolution of "{numeric}" per "{time}"', fontsize=fontsize + 10)
        else:
            plt.title(title, fontsize=fontsize + 10)
    else:
        # get the first and last day in the dataset
        unique_values = list(df[category].unique())
        last_day = [df[df[category] == i][time].max() for i in unique_values]
        last_day_values = [df[(df[time] == last_day[i]) & (df[category] == unique_values[i])] for i in
                           range(len(unique_values))]

        # Labels & Plot
        sns.lineplot(data=df, x=time, y=numeric, hue=category, ax=ax)
        if title is None:
            plt.title(f'Evolution of "{numeric}" in each "{category}" per "{time}"', fontsize=fontsize + 10)
        else:
            plt.title(title, fontsize=fontsize + 10)

        [plt.text(last_day_values[i][time], last_day_values[i][numeric], unique_values[i]) if unique_values[
                                                                                                  i] != "NaN" else None
         for i in range(len(unique_values))]

    # Plot
    plt.show()


def pairplot(df, kind='scatter', diag_kind='kde', height=2.5, aspect=1, corner=True, dropna=False, fontsize=18,
             hue=None, palette=None, x_vars=None, y_vars=None, title=None):
    """

    :Function: Plot pairwise relationships in a dataset.
        By default, this function will create a grid of Axes such that each numeric
        variable in ``data`` will by shared across the y-axes across a single row and
        the x-axes across a single column. The diagonal plots are treated
        differently: a univariate distribution plot is drawn to show the marginal
        distribution of the data in each column.

    :parameters:
    df : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : name of variable in ``data``
        Variable in ``data`` to map plot aspects to different colors.
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    {x, y}_vars : lists of variable names
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    kind : {'scatter', 'kde', 'hist', 'reg'}
        Kind of plot to make.
    diag_kind : {'auto', 'hist', 'kde', None}
        Kind of plot for the diagonal subplots. If 'auto', choose based on
        whether or not ``hue`` is used.
    height : scalar
        Height (in inches) of each facet.
    aspect : scalar
        Aspect * height gives the width (in inches) of each facet.
    corner : bool
        If True, don't add axes to the upper (off-diagonal) triangle of the
        grid, making this a "corner" plot.
    dropna : boolean
        Drop missing values from the data before plotting.
    title: str
        The figure title.
    """
    g = sns.pairplot(data=df, hue=hue,
                     palette=palette, x_vars=x_vars, y_vars=y_vars, kind=kind, diag_kind=diag_kind,
                     height=height, aspect=aspect, corner=corner, dropna=dropna)
    if title is None:
        g.fig.suptitle("Pair Plot", y=0.98, fontsize=fontsize)
    else:
        g.fig.suptitle(title, y=0.98)

    plt.show()


def scatter_dynamic_plotly(df, x, y, animation_frame, animation_group=None, size=None, color_col=None, hover_name=None,
                           labels=None, title=None, height=None, width=None, range_y=None, range_x=None, log_x=None,
                           log_y=None):
    """

    :function: Scatter Plot with positional argument of animation frame
        In a scatter plot, each row of `data_frame` is represented by a symbol
        mark in 2D space.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
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
    animation_frame: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to animation frames.
    animation_group: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        provide object-constancy across animation frames: rows with matching
        `animation_group`s
    color_col : str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    size: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign mark sizes.
    hover_name: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in bold
        in the hover tooltip.
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in the figure for axis titles, legend
        entries and hovers. This parameter allows this to be overridden. The
        keys of this dict should correspond to column names, and the values
        should correspond to the desired label to be displayed.
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
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.

    Returns
    -------
        plotly.graph_objects.Figure
    """
    if range_y is None:
        range_y = [df[y].min(), df[y].max() * 1.1]

    fig = px.scatter(df, x=x, y=y, animation_frame=animation_frame, animation_group=animation_group,
                     size=size, color=color_col, hover_name=hover_name, labels=labels, range_y=range_y,
                     range_x=range_x, log_x=log_x, log_y=log_y)
    if title is None:
        fig.update_layout(title=f'Scatter Plot between {x} and {y}', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    # Plot
    fig.show()


def scatter_plotly(df, x, y, size=None, color_col=None, hover_name=None, labels=None, title=None,
                   height=None, width=None, range_x=None, range_y=None, log_x=False, log_y=False):
    """

    :function: Scatter Plot
        In a scatter plot, each row of `data_frame` is represented by a symbol
        mark in 2D space.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
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
    color_col : str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign color to marks.
    size: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign mark sizes.
    hover_name: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in bold
        in the hover tooltip.
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in the figure for axis titles, legend
        entries and hovers. This parameter allows this to be overridden. The
        keys of this dict should correspond to column names, and the values
        should correspond to the desired label to be displayed.
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
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.

    Returns
    -------
        plotly.graph_objects.Figure
    """
    fig = px.scatter(df, x=x, y=y, size=size, color=color_col, hover_name=hover_name,
                     labels=labels, range_x=range_x, range_y=range_y, log_x=log_x, log_y=log_y)
    if title is None:
        fig.update_layout(title=f'Scatter Plot between {x} and {y}', height=height, width=width)

    else:
        fig.update_layout(title=title, height=height, width=width)

    # Plot
    fig.show()


def violin_plot(df, category, binary, numeric, figsize=(46, 30), fontsize=40, bw=0.5, operator=None, oper_to=None):
    """
    :parameter: df (pandas DataFrame)
    :parameter: category (x axis) - (specify Categorical DataFrame Column)
    :parameter: binary - (specify Binary Dataframe Column)
    :parameter: numeric (y axis) - (specify Numerical Dataframe Column)
    :parameter: figsize - tuple by default = (46,30)
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: bw - bw is the window size used for computing the kernel - Default = 0.5
    :parameter: operator - operator in "> , < , >= , <= , == , !=" - Default = None
    :parameter: oper_to - value in the binary variable to be compared to with the selected operator - Default = None
    :function: create a violin plot
    """

    # Relevant variables from original dataset
    df_copy = df[[category, binary, numeric]].copy()
    if numeric == binary:
        print('ERROR\n :parameter: binary cannot be the same as :parameter: numeric')
        return

    # Drop all NA values
    df_copy.dropna(inplace=True)

    # convert numeric to float
    pd.to_numeric(df_copy[numeric])

    # Check if inserted binary :parameter: is binary
    if len(df_copy[binary].unique()) > 2 and operator is None and oper_to is None:
        print(
            """ERROR\n :parameter: binary is not a binary variable\n 
            To use set :parameter: operator to "> , < , >= , <= , == , !="\n 
            and :parameter: oper_to to value to be compared""")
        return
    elif len(df_copy[binary].unique()) == 2 and operator is not None and oper_to is not None:
        print('ERROR\n :parameter: binary is already a binary variable\nDo not select :parameter: operator and oper_to')
        return

    # Transform non-binary variable into a binary comparizon
    if operator == '>':
        df_copy[binary] = df_copy[binary] > oper_to
    elif operator == '<':
        df_copy[binary] = df_copy[binary] < oper_to
    elif operator == '>=':
        df_copy[binary] = df_copy[binary] >= oper_to
    elif operator == '<=':
        df_copy[binary] = df_copy[binary] <= oper_to
    elif operator == '==':
        df_copy[binary] = df_copy[binary] == oper_to
    elif operator == '!=':
        df_copy[binary] = df_copy[binary] != oper_to
    elif operator is None:
        df_copy[binary] = df[binary].astype(bool)

    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(x=category, y=numeric, hue=binary,
                   data=df_copy, palette="viridis",
                   split=True, bw=bw, ax=ax)
    # Labels
    plt.xlabel(f'{category}', fontsize=fontsize)
    plt.ylabel(f'{numeric} distribution per "{category}"', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 10)
    plt.yticks(fontsize=fontsize - 10)
    plt.legend(title=binary, title_fontsize=fontsize, fontsize=fontsize)
    if (operator or oper_to) is not None:
        plt.title(f'Distribution of "{numeric}" by "{category}" splitted by "{binary}" {operator} "{oper_to}"',
                  fontsize=fontsize + 10)
    else:
        plt.title(f'Distribution of "{numeric}" by "{category}" splitted by "{binary}"',
                  fontsize=fontsize + 10)

    # Plot
    plt.show()


def kde_univariate(df, x, y=None, shade=None, palette=None, log_scale=None, fill=None,
                   figsize=(46, 30), fontsize=40, title=None):
    """
    :function: KDE plot
        Plot univariate or bivariate distributions using kernel density estimation.

        A kernel density estimate (KDE) plot is a method for visualizing the
        distribution of observations in a dataset, analagous to a histogram. KDE
        represents the data using a continuous probability density curve in one or
        more dimensions.

    Parameters
    ----------
    x, y : vectors or keys in ``data``
        Variables that specify positions on the x and y axes.
    shade : bool
        Alias for ``fill``. Using ``fill`` is recommended.
    palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
        Method for choosing the colors to use when mapping the ``hue`` semantic.
        String values are passed to :func:`color_palette`. List or dict values
        imply categorical mapping, while a colormap object implies numeric mapping.
    log_scale : bool or number, or pair of bools or numbers
        Set a log scale on the data axis (or axes, with bivariate data) with the
        given base (default 10), and evaluate the KDE in log space.
    fill : bool or None
        If True, fill in the area under univariate density curves or between
        bivariate contours. If None, the default depends on ``multiple``.
    :parameter: figsize - tuple by default = (46,30)
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: title - Title of the graph
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(data=df, x=x, y=y, palette=palette, shade=shade, log_scale=log_scale, ax=ax, fill=fill)

    plt.xticks(fontsize=fontsize - 10)
    plt.yticks(fontsize=fontsize - 10)
    plt.xlabel(f'{x}', fontsize=fontsize - 10)
    if y is not None:
        plt.ylabel(f'{y}', fontsize=fontsize - 10)

    if title is None:
        plt.title(f"Kernal Density Plot of {x}",
                  fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize)

    plt.show()


def kde_univariate_no_outliers(df, x, shade=None, palette=None, log_scale=None, fill=None,
                   figsize=(46, 30), fontsize=40, title=None):
    """
    :function: KDE plot No Outliers
        Plot univariate or bivariate distributions using kernel density estimation.

        A kernel density estimate (KDE) plot is a method for visualizing the
        distribution of observations in a dataset, analagous to a histogram. KDE
        represents the data using a continuous probability density curve in one or
        more dimensions.

    Parameters
    ----------
    df : pandas.DataFrame
    x : vectors or keys in ``data``
        Variables that specify positions on the x and y axes.
    shade : bool
        Alias for ``fill``. Using ``fill`` is recommended.
    palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
        Method for choosing the colors to use when mapping the ``hue`` semantic.
        String values are passed to :func:`color_palette`. List or dict values
        imply categorical mapping, while a colormap object implies numeric mapping.
    log_scale : bool or number, or pair of bools or numbers
        Set a log scale on the data axis (or axes, with bivariate data) with the
        given base (default 10), and evaluate the KDE in log space.
    fill : bool or None
        If True, fill in the area under univariate density curves or between
        bivariate contours. If None, the default depends on ``multiple``.
    :parameter: figsize - tuple by default = (46,30)
    :parameter: fontsize - size of the font for the axis and title - Default = 40
    :parameter: title - Title of the graph
    """
    def reject_outliers(df, x, m=2):
        """
        :function: Rejecting outliers
        :parameters: df (pandas data frame)
        :parameters: x - dataframe column
        :parameters: m - standard deviation multiplier - default = 2

        """
        return df[x][abs(df[x] - np.mean(df[x])) < m * np.std(df[x])]

    filtered_x = reject_outliers(df, x)

    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(x=filtered_x, palette=palette, shade=shade, log_scale=log_scale, ax=ax, fill=fill)

    plt.xticks(fontsize=fontsize - 10)
    plt.yticks(fontsize=fontsize - 10)
    plt.xlabel(f'{x}', fontsize=fontsize - 10)

    if title is None:
        plt.title(f"Kernal Density Plot of {x}",
                  fontsize=fontsize + 10)
    else:
        plt.title(title, fontsize=fontsize)

    plt.show()


def strip_plot(df, y, x=None, color=None, facet_row=None, facet_col=None, facet_col_wrap=0,
               facet_row_spacing=None, facet_col_spacing=None, hover_name=None, hover_data=None, custom_data=None,
               animation_frame=None, animation_group=None, category_orders=None, labels=None,
               color_discrete_sequence=None, color_discrete_map=None, orientation=None, stripmode=None,
               log_x=False, log_y=False, range_x=None, range_y=None, title=None, template=None, width=None,
               height=None):
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
    stripmode: str (default `'group'`)
        One of `'group'` or `'overlay'` In `'overlay'` mode, strips are on
        drawn top of one another. In `'group'` mode, strips are placed beside
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
    if x != None and df[x].dtypes not in [float, int]:
        color = x

    fig = px.strip(df, y=y, x=x, color=color, facet_row=facet_row, facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                   facet_row_spacing=facet_row_spacing, facet_col_spacing=facet_col_spacing, hover_name=hover_name,
                   hover_data=hover_data, custom_data=custom_data, animation_frame=animation_frame,
                   animation_group=animation_group, category_orders=category_orders, labels=labels,
                   color_discrete_sequence=color_discrete_sequence, color_discrete_map=color_discrete_map,
                   orientation=orientation, stripmode=stripmode, log_x=log_x, log_y=log_y, range_x=range_x,
                   range_y=range_y, title=title, template=template, width=width, height=height)

    if title is None:
        fig.update_layout(title=f'Strip Plot', height=height, width=width)
    else:
        fig.update_layout(title=title, height=height, width=width)

    fig.show()


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
