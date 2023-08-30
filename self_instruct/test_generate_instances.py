import unittest
from generate_instances_ie import remove_prefix_markers

class TestRemovePrefixMarkers(unittest.TestCase):

    def test_remove_prefix(self):
        input_string = "Output: Alan Turing\n\nTask: Extract information from text."
        end_marker = "Output: Alan Turing"

        extracted_text = remove_prefix_markers(input_string, end_marker)
        expected_text = "Task: Extract information from text."

        self.assertEqual(extracted_text, expected_text)

    def test_missing_marker(self):
        input_string = "Output: Alan Turing\n\nTask: Extract information from text."
        end_marker = "Markers not found"

        extracted_text = remove_prefix_markers(input_string, end_marker)
        expected_text = "Markers not found in the input string."

        self.assertEqual(extracted_text, expected_text)

if __name__ == '__main__':
    unittest.main()
