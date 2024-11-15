# @Author   : Alejandro Benavente Álvarez
# @FileName : visualization_utils.py
# @Version  : 1.0
# @IDE      : VSCode
# @Github   : https://github.com/

import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

import folium
import mapclassify
import folium.plugins


def get_geoData():
      # load the low resolution world map
      url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
      geoData = gpd.read_file(url)

      geoData = geoData[(geoData.POP_EST>0) & (geoData.NAME!="Antarctica")]
      geoData = geoData[['ISO_A3_EH', 'geometry']]
      geoData.rename(columns={"ISO_A3_EH": "Alpha-3 code", "geometry": "geometry"}, inplace = True)

      return geoData


def process_df(df, auxiliar_data):
      country_list = auxiliar_data['Alpha-3 code'].loc[:].to_list()
      columns = [f"{i}-{j}" for i in country_list for j in country_list if i != j]
      columns.sort()
      df.columns = columns

      return df


def interactive_map(geoData, auxiliar_data, tooltip=['Country'], width=1400, height=600):
      # Create the figure
      f = folium.Figure(width=width, height=height)

      #
      df = pd.merge(auxiliar_data, geoData, on="Alpha-3 code")
      world_map = gpd.GeoDataFrame(df, geometry=df.geometry)

      # Create the geometry (tiles=folium.TileLayer(no_wrap=True))
      m = world_map.explore(tooltip=tooltip, popup=False, location=[37, 0], zoom_start=2, style_kwds=dict(opacity=0.3, fillColor="lightgreen", color="lightgreen"), name="Countries").add_to(f)

      # tiles=folium.TileLayer(no_wrap=True)
      folium.plugins.ScrollZoomToggler().add_to(m)

      # Add CartoDB layer to the figure
      folium.TileLayer("CartoDB positron", show=False).add_to(m)

      # Add LayerControl item to the figure (upper right corner)
      folium.LayerControl().add_to(m)

      # Return the map in html format
      return m._repr_html_()


def simple_map(location_code, geoData):
      fig, ax = plt.subplots(figsize =(20, 5), layout="constrained")
      simple_map_body(ax, location_code, geoData)
      plt.show


def simple_map_body(ax, location_code, geoData):
      geoData.plot(
            ax=ax,
            color="gray",
            edgecolor="black",
            alpha=0.5
      )

      location = geoData[geoData["Alpha-3 code"] == location_code]
      location.plot(ax=ax, color='lightgreen', alpha=0.5)


def basic_plot(location, predictions, auxiliar_data):
      fig, ax = plt.subplots(figsize =(14, 6))
      basic_plot_body(ax, location, predictions, auxiliar_data)
      plt.show


def basic_plot_body(ax, location, predictions, auxiliar_data):
      country_list = auxiliar_data['Alpha-3 code'].loc[:].to_list()

      exports_columns = [f"{location}-{partner}"  for partner in country_list if location != partner]
      exports = get_sum_rows(predictions, exports_columns)

      imports_columns = [f"{partner}-{location}"  for partner in country_list if location != partner]
      imports = get_sum_rows(predictions, imports_columns)

      x_axis = []
      for i in range(len(exports)):
            time_step = f"Time step {str(i + 1)}"
            x_axis.append(time_step)

      ax.plot(x_axis, exports, label='Exports')
      ax.plot(x_axis, imports, label='Imports')

      ax.legend()


def get_sum_rows(df, subset_columns):
      subset = df[subset_columns]
      sum_values = subset.sum(axis=1)

      return sum_values.T.to_list()


def plot_trade_partner(location, partner, predictions):
      fig, ax = plt.subplots(figsize =(14, 6))
      plot_trade_partner_body(ax, location, partner, predictions)
      plt.show


def plot_trade_partner_body(ax, location, partner, predictions):

      exports = predictions[f"{location}-{partner}"]
      imports = predictions[f"{partner}-{location}"]

      x_axis = []
      for i in range(len(exports)):
            time_step = f"Time step {str(i + 1)}"
            x_axis.append(time_step)

      ax.plot(x_axis, exports, label='Exports')
      ax.plot(x_axis, imports, label='Imports')

      ax.legend()


def bar_plot_exports(location_code, predictions, auxiliar_data):
      fig, ax = plt.subplots(figsize =(30, 6))
      bar_plot_exports_body(ax, location_code, predictions, auxiliar_data)
      plt.show()


def bar_plot_exports_body(ax, location, predictions, auxiliar_data, n=20):
      country_list = auxiliar_data['Alpha-3 code'].loc[:].to_list()

      # Get columns
      location_exports_columns = [f"{location}-{partner}"  for partner in country_list if location != partner]
      location_exports = predictions[location_exports_columns]
      location_exports.reset_index(drop=True, inplace = True)

      # sort values
      index = location_exports.mean().sort_values(ascending=False).index
      location_exports = location_exports.reindex(index, axis=1)
      y = location_exports[location_exports.columns[:n]]

      # Create the x axis labels
      x_axis = []
      for pair in location_exports.columns:
            _, partner = pair.split("-")
            x_axis.append(partner)

      barWidth = 0.15
      colors = plt.cm.BuPu(np.linspace(0.3, 0.6, len(y)))

      bar_index = [np.arange(len(y.loc[0]))]
      for i in range(len(y) - 1):
            bar = [x + barWidth for x in bar_index[i]]
            bar_index.append(bar)

      for i in range(len(y)):
            time_step = f"Time step {str(i + 1)}"
            ax.bar(bar_index[i], y.loc[i], color=colors[i], width=barWidth, edgecolor='w', label=time_step)

      plt.xticks([r + barWidth for r in range(len(y.loc[0]))], x_axis[:n])
      ax.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.5)
      ax.legend(loc="upper right")


def pie_plot_exports(location_code, predictions, auxiliar_data):
    fig, ax = plt.subplots(figsize =(10, 8))
    pie_plot_exports_body(ax, location_code, predictions, auxiliar_data)
    plt.show()


def pie_plot_exports_body(ax, location, predictions, auxiliar_data, n=5):
    country_list = auxiliar_data['Alpha-3 code'].loc[:].to_list()

    # Get
    location_exports_index = [f"{location}-{partner}"  for partner in country_list if location != partner]
    location_exports = predictions[location_exports_index]

    # sort values
    location_exports.sort_values(ascending=False, inplace=True)
    y = location_exports[location_exports.index[:n]]

    # create a element for the rest
    rest = location_exports[location_exports.index[n:]]
    sum_value = rest.sum()
    y['Rest of countries'] = sum_value

    # Create the x axis labels
    countries = []
    for pair in location_exports.index:
        _, partner = pair.split("-")
        country = auxiliar_data['Country'].loc[auxiliar_data['Alpha-3 code'] == partner]
        countries.append(country.values[0])

    x_axis = countries[:n]
    x_axis.append("Rest of countries")

    size = 0.2
    colors = plt.cm.BuPu(np.linspace(0.3, 1, n+1))

    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%"

    wedges, text, autotexts = ax.pie(y, wedgeprops=dict(width=1, edgecolor='w'), colors=colors, autopct=lambda pct: func(pct, y))

    ax.legend(wedges, x_axis,
          loc="upper left",
          bbox_to_anchor=(1.1, 0, 0.5, 1))

    plt.setp(autotexts, size=9, color='w')


def basic_dashboard(location, geoData, predictions, auxiliar_data):
      fig = plt.figure(figsize=(25, 12))
      spec = fig.add_gridspec(2, 2)

      ax0 = fig.add_subplot(spec[0, 0])
      ax1 = fig.add_subplot(spec[0, 1])
      ax2 = fig.add_subplot(spec[1, :])

      simple_map_body(ax0, location, geoData)
      basic_plot_body(ax1, location, predictions, auxiliar_data)
      bar_plot_exports_body(ax2, location, predictions, auxiliar_data, n=20)

      plt.show()