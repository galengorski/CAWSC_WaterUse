import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
import math
import copy


class ReportSection(object):
    def __init__(
        self, title="", section="", description="", include_plotlyjs=False
    ):
        self.title = title
        self.section = section
        self.description = description
        self.include_plotlyjs = include_plotlyjs
        self._json = {}
        self._plots = {}
        self._n = 0

    def compile_html(self, shape=(1, 1), layout={}, **kwargs):
        """
        Method to compile a html section with shapely figures.

        Parameters
        ----------
        shape : tuple
            number of figure rows and columns
        layout : dict
            dictionary of layout kwargs
        kwargs :
            shapely keyword arguments

        """
        rows, cols = shape

        if rows == -1 and cols == -1:
            rows = len(self._plots)
            cols = 1

        elif rows == -1:
            rows = math.ceil(len(self._plots) / cols)

        elif cols == -1:
            cols = math.ceil(len(self._plots) / rows)

        else:
            pass

        if "specs" not in kwargs:
            s = [
                None,
            ] * cols
            specs = [copy.copy(s) for _ in range(rows)]

            i = 0
            j = 0
            for k, d in self._json.items():
                specs[i][j] = {"type": d["type"]}
                if j + 1 == cols:
                    i += 1
                    j = 0
                else:
                    j += 1

        else:
            specs = kwargs.pop("specs")

        fig = make_subplots(rows=rows, cols=cols, specs=specs, **kwargs)

        r, c = 1, 1
        for _, tr in self._plots.items():
            if isinstance(tr, list):
                fig.add_traces(tr, rows=r, cols=c)
            else:
                fig.add_trace(tr, row=r, col=c)

            if c == cols:
                c = 1
                r += 1
            else:
                c += 1

        if layout:
            fig.update_layout(**layout)

        plot_html = to_html(
            fig, include_plotlyjs=self.include_plotlyjs, full_html=True
        )

        html = []
        if self.title:
            html.append("<h1>{}</h1>\n".format(self.title))

        if self.section:
            html.append("<h2>{}</h2>\n".format(self.section))

        if self.description:
            html.append("<p>{}</p>\n".format(self.description))

        html.append("{}\n".format(plot_html))

        html = "".join(html)

        return html

    def pandas_to_scatter(self, df, xfield, **kwargs):
        """

        Parameters
        ----------

        df : pd.DataFrame
        xfield : str
            x axis data field
        **kwargs :
            plotly scatter keyword arguments

        """
        d = dict()
        cols = [i for i in list(df) if i.lower() != "date"]

        for ix, col in enumerate(cols):
            t = dict(
                x=df[xfield].tolist(), y=df[col].tolist(), name=col, **kwargs
            )
            d[ix] = t

        self._json[self._n] = {"type": "scatter", "data": d}

        traces = []

        for _, v in d.items():
            traces.append(go.Scatter(**v))
        self._plots[self._n] = traces

        self._n += 1

    def pandas_to_table(self, df, **kwargs):
        """

        Parameters
        ----------
        df : pd.DataFrame
        kwargs :
            plotly table keyword arguments

        """
        if "header" in kwargs:
            header = kwargs.pop("header")
        else:
            header = dict(values=list(df), font=dict(size=10), align="left")

        if "cells" in kwargs:
            cells = kwargs.pop("cells")
        else:
            cells = dict(
                values=[df[k].tolist() for k in df.columns], align="left"
            )

        d = dict(header=header, cells=cells, **kwargs)

        self._json[self._n] = {"type": "table", "data": d}

        trace = go.Table(**d)
        self._plots[self._n] = trace

        self._n += 1

    def geojson_to_heatmap(
        self, geojson, featureidkey, locations, z, **kwargs
    ):
        """

        :param geojson:
        :return:
        """
        d = dict(
            geojson=geojson,
            featureidkey=featureidkey,
            locations=locations,
            z=z,
            **kwargs
        )

        self._json[self._n] = {"type": "choroplethmapbox", "data": d}

        trace = go.Choroplethmapbox(**d)
        self._plots[self._n] = trace

        self._n += 1
