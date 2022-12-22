import unittest
import read_files
from helpers import read_json

class Test_read_files(unittest.TestCase):
    def test_init_data(self):
        properties = read_json('','properties.json')
        data = read_files.init_data(properties)
        self.assertEqual(type(data), dict)

    def test_scan_folder(self):
        properties = read_json('','properties.json')
        path = properties['root_path']
        data = read_files.scan_folder(path, properties)
        self.assertEqual(type(data), dict)

if __name__ == '__main__':
    unittest.main()