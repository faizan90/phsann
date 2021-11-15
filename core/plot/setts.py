'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import matplotlib.pyplot as plt


class PlotSettings:

    '''For internal use

    Description of settings at:
    https://matplotlib.org/tutorials/introductory/customizing.html
    '''

    def __init__(self, figsize, dpi, fontsize):

        assert isinstance(figsize, tuple), 'figsize not a tuple!'

        assert len(figsize) == 2, 'Only two items allowed in figsize!'

        assert all([isinstance(size, (int, float)) for size in figsize]), (
            'Items in figsize can either be integers or floats!')

        assert all([size > 0 for size in figsize]), (
            'items in figsize cannot be <= 0!')

        assert isinstance(dpi, int), 'dpi not an integer!'

        assert dpi > 0, 'dpi should be > 0!'

        assert isinstance(fontsize, int), 'fontsize not an integer!'

        assert fontsize > 0

        self.prms_dict = {
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'font.size': fontsize,
            }

        return


class PlotLineSettings(PlotSettings):

    '''For internal use'''

    def __init__(
            self,
            figsize,
            dpi,
            fontsize,
            alpha_1,
            alpha_2,
            lw_1,
            lw_2,
            lc_1,
            lc_2):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha can only be 0 <= alpha_2 <= 1!'

        assert isinstance(lw_1, (int, float)), (
            'lw_1 not an integer or a float!')

        assert lw_1 > 0, 'lw_1 must be > 0!'

        assert isinstance(lw_2, (int, float)), (
            'lw_2 not an integer or a float!')

        assert lw_2 > 0, 'lw_2 must be > 0!'

        assert isinstance(lc_1, (str, tuple, hex)), 'Invalid lc_1!'
        assert isinstance(lc_2, (str, tuple, hex)), 'Invalid lc_2!'

        self.prms_dict.update({
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.lw_1 = lw_1
        self.lw_2 = lw_2

        self.lc_1 = lc_1
        self.lc_2 = lc_2
        return


class PlotImageSettings(PlotSettings):

    '''For internal use'''

    def __init__(self, figsize, dpi, fontsize, alpha_1, alpha_2, cmap):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(cmap, str), 'cmap not a string!'

        assert cmap in plt.colormaps(), 'Unknown cmap!'

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha_text can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha_text can only be 0 <= alpha_2 <= 1!'

        self.prms_dict.update({
            'image.cmap': cmap,
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        return


class PlotScatterSettings(PlotSettings):

    '''For internal use'''

    def __init__(self, figsize, dpi, fontsize, alpha_1, alpha_2, color):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(color, (str, tuple, hex)), 'Invalid color!'

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha_text can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha_text can only be 0 <= alpha_2 <= 1!'

        self.prms_dict.update({
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.color = color
        return


def get_mpl_prms(keys):

    prms_dict = {key: plt.rcParams[key] for key in keys}

    return prms_dict


def set_mpl_prms(prms_dict):

    plt.rcParams.update(prms_dict)

    return
