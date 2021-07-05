
class BaseEstimator:
    def fit(self, X):
        raise NotImplementedError()

    def query(self, Y):
        raise NotImplementedError()

    def set_query_param(self, args):
        raise NotImplementedError()

    def __str__(self):
        raise Exception("No representation given")
