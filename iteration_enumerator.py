

#Convenience class for switching between trk_algo indices and step names
# class _Enum:
#     def __init__(self, **values):
#         self._reverse = {}
#         for key, value in values.items():
#             setattr(self, key, value)
#             if value in self._reverse:
#                 raise Exception("Value %s is already used for a key %s, tried to re-add it for key %s" % (
#                 value, self._reverse[value], key))
#             self._reverse[value] = key
#
#     def toString(self, val):
#         return self._reverse[val]
class TwoWayDict(dict):
    def __init__(self, list_of_tuples):
        self.add(list_of_tuples)
    def add(self, list_of_tuples):
        for tuple in list_of_tuples:
            self[tuple[0]] = tuple[1]
            self[tuple[1]] = tuple[0]

iteration_enumerator = TwoWayDict([
    ("InitialStep", 4),
    ("LowPtTripletStep", 5),
    ("PixelPairStep", 6),
    ("DetachedTripletStep", 7),
    ("MixedTripletStep", 8),
    ("PixelLessStep", 9),
    ("TobTecStep", 10),
    ("JetCoreRegionalStep", 11),
    # Phase1
    ("HighPtTripletStep", 22),
    ("LowPtQuadStep", 23),
    ("DetachedQuadStep", 24)
])