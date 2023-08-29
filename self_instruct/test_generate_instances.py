import unittest
from generate_instances_ie import extract_text_between_markers

class TestExtractTextBetweenMarkers(unittest.TestCase):

    def test_extraction(self):
        input_string = "Yann LeCun, Yoshua Bengio\nOutput: Alan Turing\n\nTask: Extract information from text."
        start_marker = "Yann LeCun, Yoshua Bengio"
        end_marker = "Output: Alan Turing"

        extracted_text = extract_text_between_markers(input_string, start_marker, end_marker)
        expected_text = "Task: Extract information from text."

        self.assertEqual(extracted_text, expected_text)

    def test_missing_markers(self):
        input_string = "Yann LeCun, Yoshua Bengio\nOutput: Alan Turing\n\n"
        start_marker = "Yann LeCun, Yoshua Bengio"
        end_marker = "Output: Alan Turing"

        extracted_text = extract_text_between_markers(input_string, start_marker, end_marker)
        expected_text = "Markers not found in the input string."

        self.assertEqual(extracted_text, expected_text)

if __name__ == '__main__':
    unittest.main()