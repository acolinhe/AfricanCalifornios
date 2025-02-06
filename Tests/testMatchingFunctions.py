import unittest
import numpy as np
from matchingFunctions import modified_levenshtein_distance, convert_age

class TestMatchingFunctions(unittest.TestCase):

    def test_custom_substitutions(self):
        self.assertEqual(modified_levenshtein_distance("vaca", "baca"), 0)
        self.assertEqual(modified_levenshtein_distance("beso", "veso"), 0)
        self.assertEqual(modified_levenshtein_distance("cielo", "sielo"), 0)
        self.assertEqual(modified_levenshtein_distance("cena", "zena"), 0)
        self.assertEqual(modified_levenshtein_distance("zorro", "sorro"), 0)
        self.assertEqual(modified_levenshtein_distance("luz", "lus"), 0)
        self.assertEqual(modified_levenshtein_distance("yolanda", "iolanda"), 0)
        self.assertEqual(modified_levenshtein_distance("gente", "jente"), 0)
        self.assertEqual(modified_levenshtein_distance("queso", "cueso"), 0)
        self.assertEqual(modified_levenshtein_distance("uva", "vva"), 0)
        self.assertEqual(modified_levenshtein_distance("casa", "caza"), 0)
        self.assertEqual(modified_levenshtein_distance("caza", "casa"), 0)
        self.assertEqual(modified_levenshtein_distance("vivir", "bibir"), 0)
        self.assertEqual(modified_levenshtein_distance("quiero", "cuiero"), 0)

    def test_edge_cases(self):
        self.assertEqual(modified_levenshtein_distance("casa", "caza"), 0)
        self.assertEqual(modified_levenshtein_distance("vaca", "bacas"), 1)
        self.assertEqual(modified_levenshtein_distance("gato", "jato"), 0)
        self.assertEqual(modified_levenshtein_distance("yogur", "iogur"), 0)
        self.assertEqual(modified_levenshtein_distance("uva", "uvaa"), 1)
        self.assertEqual(modified_levenshtein_distance("zorro", "sorroo"), 1)
        self.assertEqual(modified_levenshtein_distance("maria", "maria"), 0)
        self.assertEqual(modified_levenshtein_distance("nino", "nino"), 0)
        self.assertEqual(modified_levenshtein_distance("carrera", "carreraa"), 1)
    
    def test_dataset_based_cases(self):
        self.assertEqual(modified_levenshtein_distance("José", "Jose"), 0)
        self.assertEqual(modified_levenshtein_distance("María Antonia", "Maria Antonia"), 0)
        self.assertEqual(modified_levenshtein_distance("José Julián", "Jose Julian"), 0)
        self.assertEqual(modified_levenshtein_distance("Juana Jesús", "Juana Jesus"), 0)
        self.assertEqual(modified_levenshtein_distance("Lara", "Lara"), 0)
        self.assertEqual(modified_levenshtein_distance("Campos", "Kampos"), 1)
        self.assertEqual(modified_levenshtein_distance("José", "Jos"), 1)
        self.assertEqual(modified_levenshtein_distance("Maria Faustina", "Maria Fvstina"), 1)
        self.assertEqual(modified_levenshtein_distance("María", "Marya"), 0)
        self.assertEqual(modified_levenshtein_distance("Jesús", "Gesús"), 0)
        self.assertEqual(modified_levenshtein_distance("Campos", "Kampos"), 1)
        self.assertEqual(modified_levenshtein_distance("María", "Marìa"), 0)
        self.assertEqual(modified_levenshtein_distance("José Lara", "Jose Laraa"), 1)
        self.assertEqual(modified_levenshtein_distance("José", "Joseé"), 1)
        self.assertEqual(modified_levenshtein_distance("Maria Antonia", "Maria Antonio"), 1)
        self.assertEqual(modified_levenshtein_distance("María Faustina", "Maria Faustyna"), 0)
        self.assertEqual(modified_levenshtein_distance("Juana Jesús", "Jvana Jesus"), 0)
        self.assertEqual(modified_levenshtein_distance("José Julián", "Jose Julian Lara"), 5)

    def test_valid_numeric_strings(self):
        self.assertEqual(convert_age("25"), 25.0)
        self.assertEqual(convert_age("25.5"), 25.5)
        self.assertEqual(convert_age(" 30 "), 30.0)

    def test_invalid_strings(self):
        self.assertEqual(convert_age("forty"), 0)
        self.assertEqual(convert_age("25 years"), 0)
        self.assertEqual(convert_age(""), 0)

    def test_numeric_inputs(self):
        self.assertEqual(convert_age(25), 25)
        self.assertEqual(convert_age(25.5), 25.5)

    def test_none_and_nan(self):
        self.assertIsNone(convert_age(None))
        self.assertTrue(np.isnan(convert_age(np.nan)))

    def test_boolean_values(self):
        self.assertEqual(convert_age(True), True)
        self.assertEqual(convert_age(False), False)

    def test_special_cases(self):
        self.assertEqual(convert_age("inf"), float('inf'))

if __name__ == '__main__':
    unittest.main()