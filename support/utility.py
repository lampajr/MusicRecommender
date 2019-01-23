import numpy as np
import pandas as pd
import scipy.sparse as sp


###############################################
################# UTILITY #####################
###############################################


def read_data(filename, subfolder='/input/'):

    """
    Read the csv file given the its path
    :param filename: name of the csv file
    :param subfolder: subfolder of the file
    :return: Dataframe class
    """
    return pd.read_csv(filepath_or_buffer=subfolder+filename, engine='python')


def create_sparse(data, row, col, format='csr'):

    """
    Generates a sparse matrix given a pandas dataframe
    :param data: dataframe
    :param row: name of the attribute to be considered as rows
    :param col: name of the attribute to be considered as columns
    :param format: which sparse matrix generate
    :return: the scipy sparse matrix
    """

    ratings = [1] * data[row].size  # number of interactions
    rows = data[row]
    columns = data[col]

    if format == 'csr':
        return sp.csr_matrix((ratings, (rows, columns)))
    elif format == 'csc':
        return sp.csc_matrix((ratings, (rows, columns)))
    else:
        print('format not supported!')
        return


def get_tops(ratings, k):

    """ Returns an array of k best tracks according to the ratings provided """

    return np.flip(np.argsort(ratings))[0:k]


def check_matrix(data, format='csc', dtype=np.float32):

    """ Checks the matrix format, if different convert the current matrix
        into the specified format and return it """

    if format == 'csc' and not isinstance(data, sp.csc_matrix):
        return data.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(data, sp.csr_matrix):
        return data.tocsr().astype(dtype)
    elif format == 'lil' and not isinstance(data, sp.lil_matrix):
        return data.tolil().astype(dtype)
    else:
        return data.astype(dtype)


def create_csv(data, pathname, columns=('playlist_id', 'track_ids')):
    """
    Creates the csv file from a given pandas dataframe
    :param data: dataframe containing the data results
    :param pathname: path where to save the csv file
    :param columns: name of the two columns
    """

    data.to_csv(path_or_buf=pathname, columns=columns, index=False)


