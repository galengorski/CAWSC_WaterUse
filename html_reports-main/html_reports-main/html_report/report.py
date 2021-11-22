import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

# todo: this module is defunct. Plans are to make this a wrapper that can
# todo: "snap" together ReportSections and provide read/write capabilities

class Report(object):
    """
    Class to handle reporting for Machine learning processing

    Parameters
    ----------
    name : str


    """

    def __init__(self, name=""):

        self._name = name
        self._metadata = {"title":
                          "Machine Learning report for {}".format(name)}
        self.__idf = None

    @property
    def metadata(self):
        """
        Property that returns a copy of the metadata dictionary

        """
        return self._metadata

    @property
    def initial_dataframe(self):
        """
        Propertry method that returns a copy of the initial dataframe

        """
        return self.__idf

    def set_initial_dataframe(self, df):
        """
        Method to set the initial dataframe

        Parameters
        ----------
        df : pandas DataFrame

        """
        if self.__idf is None:
            self.__idf = df
            table = self._df_to_metadata_table(df)
            scatter = self._df_to_metadata_scatter(df, "lines+markers")
            self._metadata[0] = dict(title="Initial data",
                                     table=table,
                                     scatter=scatter)

        else:
            raise AttributeError("Initial dataframe cannot be reset!")

    def _df_to_metadata_table(self, df):
        """
        General method to convert a df to a plotly table

        Parameters
        ----------
        df : pd.Dataframe

        Returns
        -------
            dict

        """
        header = dict(values=list(df),
                      font=dict(size=10),
                      align="left")
        cells=dict(values=[df[k].tolist() for k in df.columns],
                   align="left")
        d = dict(header=header,
                 cells=cells)
        return d

    def _df_to_metadata_scatter(self, df, mode="markers"):
        """
        Method to convert a df into a metadata scatter plot for later use
        by plotly

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
            dict
        """
        d = dict()
        cols = [i for i in list(df) if i.lower() != "date"]

        for ix, col in enumerate(cols):
            t = dict(x=df.date.tolist(), y=df[col].tolist(),
                     name=col,
                     mode=mode)
            d[ix] = t

        return d

    def _get_report_spec(self):
        """
        Method to create a spec listing for plotly figures

        Returns
        -------
            list
        """
        spec = []
        for k, metadata in self._metadata.items():
            if k == "title":
                continue

            spc = []
            nitems = len(metadata) - 1
            for ptype, meta in metadata.items():
                if ptype == 'title':
                    continue

                else:
                    if nitems == 1:
                        spc.append(dict(type=ptype,
                                        colspan=2))
                        spc.append(None)

                    else:
                        spc.append(dict(type=ptype))

            spec.append(spc)

        return spec

    def _get_report_titles(self):
        """
        Method to generate plot titles for the report

        Returns
        -------
            list
        """
        titles = []
        for k, metadata in self._metadata.items():
            if k == "title":
                continue
            else:
                for ptype, meta in metadata.items():
                    if ptype == "title":
                        continue
                    else:
                        if ptype == "table":
                            titles.append("{} of {}".format(ptype,
                                                            metadata['title']))
                        else:
                            titles.append("{} plot of {}".format(
                                ptype,
                                metadata['title']))

        return titles

    def to_html(self, f):
        """
        Method to create a html report of all metadata

        """
        nrow = len(self._metadata) - 1
        if nrow < 1:
            raise Exception("Not enough data in metadata to create a report")

        height = 800 * nrow
        spec = self._get_report_spec()
        subplot_titles = self._get_report_titles()

        fig = make_subplots(rows=nrow, cols=2, specs=spec,
                            subplot_titles=subplot_titles)

        for k, metadata in self._metadata.items():
            if k == "title":
                continue

            itm = 1
            for ptype, meta in metadata.items():
                if ptype == 'title':
                    continue
                else:
                    if ptype == "table":
                        trace = go.Table(**meta)

                    elif ptype == "scatter":
                        trace = [go.Scatter(**t) for kk, t in meta.items()]

                    else:
                        trace = None

                if isinstance(trace, list):
                    fig.add_traces(trace, rows=k+1, cols=itm)

                elif trace is not None:
                    fig.add_trace(trace, row=k+1, col=itm)

                else:
                    raise NotImplementedError()

                itm += 1

        fig.update_layout(height=height,
                          title_text=self._metadata['title'])
        x = 'cdn'
        html = to_html(fig, include_plotlyjs=True, full_html=True)
        html = """<h1>{}</h1>\n<h2>Example section 1 </h2>\n<p>Here is some text that can be used for describing the data!</p>\n{}\n<h2>Example section 2</h2>""".format(self.metadata['title'], html)
        html2 = to_html(fig, include_plotlyjs=False, full_html=True)
        html += "\n{}\n".format(html2)
        with open('reporting_class_development_with_sections.html', 'w') as foo:
            foo.write(html)

        fig.write_html(f)

    def to_plot(self):
        """
        Method to create a static report of all metadata

        """
        return


