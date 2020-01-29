import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt

# adjust pandas display options to make improve dataframe printouts
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class STC:
    """
    Analyze the spacio-temporal consistency of bower building.

    Attributes:
         trial (str): the identifier for the trial being analyzed
         pixel_size (float): pixel to cm conversion factor
         cluster_file (str): path to AllClusterData.csv
         df (pandas.DataFrame): data-frame of the cluster data used for analysis
    """

    def __init__(self, trial, cluster_file):
        """
        Construct spacio-temporal consistency (STC) class

        Parameters:
            trial (str): the identifier for the trial being analyzed
            cluster_file (str): path to AllClusterData.csv
        """
        self.trial = trial
        self.pixel_size = 0.1030168618  # cm/pixel
        self.cluster_file = cluster_file
        self.df = self._prep_cluster_data()

    def plot_progression(self, min_min_gap=0.1, max_min_gap=600.0, iterations=50):
        """
        Plot various bout metrics against the minimum gap between bouts.

        To split the data into bouts (temporal groupings of scoops and spits), a series containing the timestamps of
        every spit and scoop event is split wherever the time-delta between two successive events is greater than
        some minimum gap (in seconds). Small minimum gap values result in a large number of short bouts, while large
        values lead to a small number of long bouts. Next, for each individual bout, the coordinates of the contained
        scoop and spit events are used to calculate bout centroids for scoops and spits respectively. Next, the distance
        of each event from its respective bout centroid is calculated, and these values are averaged to get the
        'avg_dist_to_bout_centroid' for a given minimum gap. In addition, the average time between events within bouts,
        as well as the average number of events per bout, are calculated. This process is repeated for numerous values
        of minimum gap (ranging from min_min_gap to max_min_gap, logarithmically), and the results plotted.

        Parameters:
            min_min_gap (float): smallest value (in seconds) to be used for the minimum gap between bouts
            max_min_gap (float): largest number (in seconds) to be used for the minimum gap between bouts
            iterations (int): number of data points to use. Higher numbers give higher resolution but decrease
                performance

        Returns:
            df (pandas.DataFrame): dataframe of the numerical data used in plotting. Can be stored in a variable
                and used for further analysis if desired
        """
        min_min_gap = math.log(min_min_gap, 10)
        max_min_gap = math.log(max_min_gap, 10)
        df = []
        for i, min_gap in enumerate(np.logspace(min_min_gap, max_min_gap, iterations)):
            print('Iterating:{}/{}'.format(i + 1, iterations), end='\r')
            df.append(self._return_bout_summary(min_gap))
        df = pd.concat(df)
        df.reset_index(inplace=True)
        fig, axes = plt.subplots(3, 1, sharex='all')
        sns.lineplot(x='min_gap_between_bouts', y='avg_dist_to_bout_centroid', hue='bid', data=df, ax=axes[0])
        sns.lineplot(x='min_gap_between_bouts', y='avg_time_gap_within_bout', hue='bid', data=df, ax=axes[1])
        sns.lineplot(x='min_gap_between_bouts', y='avg_num_events_per_bout', hue='bid', data=df, ax=axes[2])
        plt.show()
        return df

    def _prep_cluster_data(self):
        df = pd.read_csv(self.cluster_file, index_col='TimeStamp', parse_dates=True, infer_datetime_format=True,
                         usecols=['TimeStamp', 'modelAll_18_pred', 'X_depth', 'Y_depth'], skip_blank_lines=True)
        df.rename(columns={'modelAll_18_pred': 'bid', 'X_depth': 'X', 'Y_depth': 'Y'}, inplace=True)
        df = df[(df.bid == 'c') | (df.bid == 'p')].sort_index()
        df['elapsed_time'] = (df.index - df.index.min()).total_seconds()
        df.reset_index(inplace=True, drop=True)
        df['time_gap'] = df.elapsed_time.diff()
        df['position'] = list(zip(df.X, df.Y))
        df.drop(columns=['X', 'Y', 'elapsed_time'])
        df = df[1:]
        return df

    def _return_bout_summary(self, min_gap):
        df = self.df.copy()
        splits = df.index[(df.time_gap > min_gap) | (df.index == df.index[0])]
        labels = range(len(splits))[1:]
        df['bout_number'] = pd.cut(df.index, bins=splits, labels=labels, include_lowest=True, right=False).astype(float)
        df.fillna({'bout_number': (labels[-1] + 1)}, inplace=True)
        df.bout_number = df.bout_number.astype(int)
        df['dist_to_bout_centroid'] = df.position.groupby([df.bout_number, df.bid]).transform(self._return_dist_to_centroid)
        df.dropna(subset=['dist_to_bout_centroid'], inplace=True)
        df.loc[df.index.isin(splits.to_list()), 'time_gap'] = np.NaN
        summary = df.groupby('bid').agg({'dist_to_bout_centroid': np.nanmean, 'time_gap': np.nanmean})
        summary.columns = ['avg_dist_to_bout_centroid', 'avg_time_gap_within_bout']
        summary['min_gap_between_bouts'] = min_gap
        summary.avg_dist_to_bout_centroid = summary.avg_dist_to_bout_centroid * self.pixel_size
        summary['avg_num_events_per_bout'] = df.X.groupby([df.bout_number, df.bid]).count().groupby('bid').mean()
        return summary

    def _return_dist_to_centroid(self, xy_series):
        if len(xy_series) < 3:
            return np.NaN
        else:
            centroid = np.nanmean([xy[0] for xy in xy_series]), np.nanmean([xy[1] for xy in xy_series])
            dist_to_centroid = [math.sqrt((xy[0] - centroid[0]) ** 2 + (xy[1] - centroid[1]) ** 2) for xy in xy_series]
            return dist_to_centroid


trial_ = 'MC16_2'
cluster_file_ = '/home/tlancaster6/PycharmProjects/cichlid-lab/data/{}/AllClusterData.csv'.format(trial_)
stc_obj = STC(trial_, cluster_file_)
data = stc_obj.plot_progression(iterations=10)
